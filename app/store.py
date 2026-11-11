from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

import numpy as np

from app import embeddings
from app.bm25 import LexicalIndex, bm25_scores, rank_map, rrf
from app.chunking import chunk_text
from app.config import (
    CHUNK_CHARS,
    MIN_MERGE_CHARS,
    OVERLAP_CHARS,
    RETRIEVAL_MODE,
    STORE_DIR,
    TOP_K,
)
from app.schemas import DOC_ID_PATTERN

_SAFE_DOC_ID = re.compile(DOC_ID_PATTERN)
MAX_DOC_ID_CHARS = 64


def safe_doc_id(doc_id: str) -> str:
    """A doc id becomes a directory name, so it must never carry a path.

    The API schema already rejects unsafe ids; this is the second lock on
    the same door, so no caller can make the store touch a path outside
    its own root.
    """
    if (
        not isinstance(doc_id, str)
        or not 1 <= len(doc_id) <= MAX_DOC_ID_CHARS
        or _SAFE_DOC_ID.fullmatch(doc_id) is None
    ):
        raise ValueError("非法的文档编号：只允许字母、数字、下划线与连字符")
    return doc_id


RETRIEVAL_MODES = ("hybrid", "dense", "bm25")


def resolve_mode(mode: str | None) -> str:
    """Unknown or missing modes fall back to the configured default."""
    value = (mode or RETRIEVAL_MODE or "hybrid").strip().lower()
    return value if value in RETRIEVAL_MODES else "hybrid"


class DocStore:
    """In-memory matrices + JSON/NPY persistence under data/store/.

    Each document also carries a BM25 postings index. It is rebuilt from the
    chunk texts on load, so nothing new is written to disk.
    """

    def __init__(self, root: Path | None = None):
        self.root = Path(root) if root else STORE_DIR
        self.root.mkdir(parents=True, exist_ok=True)
        self.docs: dict[str, dict] = {}
        self.matrices: dict[str, np.ndarray] = {}
        self.lexical: dict[str, LexicalIndex] = {}
        self.lock = asyncio.Lock()

    def _doc_dir(self, doc_id: str) -> Path:
        return self.root / safe_doc_id(doc_id)

    def load_disk(self) -> int:
        loaded = 0
        if not self.root.exists():
            return 0
        for child in self.root.iterdir():
            if _SAFE_DOC_ID.fullmatch(child.name) is None:
                continue
            meta = child / "meta.json"
            chunks_path = child / "chunks.json"
            npy = child / "embeddings.npy"
            if not (meta.exists() and chunks_path.exists() and npy.exists()):
                continue
            try:
                info = json.loads(meta.read_text(encoding="utf-8"))
                chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
                matrix = np.load(npy)
                if matrix.ndim != 2 or matrix.shape[0] != len(chunks):
                    continue
                self.docs[child.name] = {
                    "name": info.get("name") or child.name,
                    "chunks": chunks,
                    "raw": info.get("raw", ""),
                }
                self.matrices[child.name] = matrix
                self.lexical[child.name] = LexicalIndex(chunks)
                loaded += 1
            except Exception:
                continue
        return loaded

    def _persist(self, doc_id: str) -> None:
        doc = self.docs[doc_id]
        folder = self._doc_dir(doc_id)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "meta.json").write_text(
            json.dumps({"name": doc["name"], "raw": doc.get("raw", "")}, ensure_ascii=False),
            encoding="utf-8",
        )
        (folder / "chunks.json").write_text(
            json.dumps(doc["chunks"], ensure_ascii=False),
            encoding="utf-8",
        )
        np.save(folder / "embeddings.npy", self.matrices[doc_id])

    async def add(self, doc_id: str, name: str, text: str) -> int:
        safe_doc_id(doc_id)
        chunks = chunk_text(text, CHUNK_CHARS, OVERLAP_CHARS, MIN_MERGE_CHARS)
        if not chunks:
            raise ValueError("No extractable text")
        loop = asyncio.get_event_loop()
        matrix = await embeddings.encode_async(loop, chunks)
        async with self.lock:
            self.docs[doc_id] = {"name": name, "chunks": chunks, "raw": text}
            self.matrices[doc_id] = matrix
            self.lexical[doc_id] = LexicalIndex(chunks)
            self._persist(doc_id)
        return len(chunks)

    async def remove(self, doc_id: str) -> None:
        folder = self._doc_dir(doc_id)
        async with self.lock:
            self.docs.pop(doc_id, None)
            self.matrices.pop(doc_id, None)
            self.lexical.pop(doc_id, None)
        if folder.is_dir():
            for p in folder.iterdir():
                p.unlink(missing_ok=True)
            folder.rmdir()

    def _overlap_terms(self, query: str, chunk: str, limit: int = 8) -> list[str]:
        q = set(query.replace("\n", " ").split())
        extra = [query[i : i + 2] for i in range(len(query) - 1)]
        hits = []
        for term in list(q) + extra:
            if len(term) < 2:
                continue
            if term in chunk and term not in hits:
                hits.append(term)
            if len(hits) >= limit:
                break
        return hits

    async def search(
        self,
        query: str,
        doc_ids: list[str],
        k: int = TOP_K,
        mode: str | None = None,
    ) -> list[dict]:
        """Top-k chunks for ``query`` across ``doc_ids``.

        ``mode`` picks the ranking: ``dense`` (BGE cosine only), ``bm25``
        (keyword only) or ``hybrid`` (reciprocal-rank fusion of the two).
        Whatever the mode, ``score`` stays the cosine similarity so scores
        remain comparable across queries; the fields beside it say how the
        final rank was reached.
        """
        mode = resolve_mode(mode)
        loop = asyncio.get_event_loop()
        q_vec = await embeddings.encode_async(loop, [query])
        q_vec = q_vec[0]
        cosine: dict[tuple[str, int], float] = {}
        async with self.lock:
            selected = [d for d in doc_ids if d in self.matrices and d in self.docs]
            for did in selected:
                for i, score in enumerate(np.dot(self.matrices[did], q_vec)):
                    cosine[(did, i)] = float(score)
            lexical = bm25_scores(
                query, {d: self.lexical[d] for d in selected if d in self.lexical}
            )
            dense_rank = rank_map(cosine)
            bm25_rank = rank_map(lexical)
            if mode == "dense":
                final = dict(cosine)
            elif mode == "bm25":
                final = dict(lexical)
            else:
                final = rrf([dense_rank, bm25_rank])
            # ties (and the keys RRF scores equally) fall back to cosine
            ordered = sorted(final, key=lambda key: (-final[key], -cosine[key], key))[:k]
            top: list[dict] = []
            for rank, key in enumerate(ordered, start=1):
                did, i = key
                doc = self.docs[did]
                text = doc["chunks"][i]
                item = {
                    "doc_id": did,
                    "doc_name": doc["name"],
                    "chunk_idx": i,
                    "text": text,
                    "score": cosine[key],
                    "overlap": self._overlap_terms(query, text),
                    "rank": rank,
                    "mode": mode,
                    "dense_rank": dense_rank[key],
                    "bm25_score": round(lexical.get(key, 0.0), 4),
                    "bm25_rank": bm25_rank.get(key),
                    "fused_score": round(final[key], 6),
                }
                item["why"] = self._why(item)
                top.append(item)
        return top

    @staticmethod
    def _why(item: dict) -> str:
        """One sentence a learner can check: how this chunk earned its rank."""
        overlap = f"与问句共享 {len(item['overlap'])} 处字词重叠"
        if item["mode"] == "dense":
            return f"余弦相似度 {item['score']:.3f}，全库第 {item['rank']} 名；{overlap}"
        if item["bm25_rank"] is None:
            lexical = "关键词未命中"
        else:
            lexical = f"关键词 BM25 {item['bm25_score']:.2f}（关键词第 {item['bm25_rank']} 名）"
        semantic = f"语义相似度 {item['score']:.3f}（语义第 {item['dense_rank']} 名）"
        if item["mode"] == "bm25":
            return f"关键词排序第 {item['rank']} 名：{lexical}；{semantic}；{overlap}"
        return f"融合排序第 {item['rank']} 名：{semantic}；{lexical}；{overlap}"

    async def get_chunk(self, doc_id: str, chunk_idx: int) -> dict | None:
        async with self.lock:
            doc = self.docs.get(doc_id)
            if not doc or chunk_idx < 0 or chunk_idx >= len(doc["chunks"]):
                return None
            return {
                "doc_id": doc_id,
                "doc_name": doc["name"],
                "chunk_idx": chunk_idx,
                "text": doc["chunks"][chunk_idx],
                "total": len(doc["chunks"]),
            }

    async def list_docs(self) -> list[dict]:
        async with self.lock:
            return [
                {"id": did, "name": d["name"], "chunks": len(d["chunks"])}
                for did, d in self.docs.items()
            ]
