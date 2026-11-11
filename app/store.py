from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

import numpy as np

from app import embeddings
from app.chunking import chunk_text
from app.config import CHUNK_CHARS, MIN_MERGE_CHARS, OVERLAP_CHARS, STORE_DIR, TOP_K
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


class DocStore:
    """In-memory matrices + JSON/NPY persistence under data/store/."""

    def __init__(self, root: Path | None = None):
        self.root = Path(root) if root else STORE_DIR
        self.root.mkdir(parents=True, exist_ok=True)
        self.docs: dict[str, dict] = {}
        self.matrices: dict[str, np.ndarray] = {}
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
            self._persist(doc_id)
        return len(chunks)

    async def remove(self, doc_id: str) -> None:
        folder = self._doc_dir(doc_id)
        async with self.lock:
            self.docs.pop(doc_id, None)
            self.matrices.pop(doc_id, None)
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
    ) -> list[dict]:
        loop = asyncio.get_event_loop()
        q_vec = await embeddings.encode_async(loop, [query])
        q_vec = q_vec[0]
        results: list[dict] = []
        async with self.lock:
            for did in doc_ids:
                if did not in self.matrices or did not in self.docs:
                    continue
                matrix = self.matrices[did]
                doc = self.docs[did]
                scores = np.dot(matrix, q_vec)
                for i, score in enumerate(scores):
                    text = doc["chunks"][i]
                    results.append(
                        {
                            "doc_id": did,
                            "doc_name": doc["name"],
                            "chunk_idx": i,
                            "text": text,
                            "score": float(score),
                            "overlap": self._overlap_terms(query, text),
                        }
                    )
        results.sort(key=lambda x: x["score"], reverse=True)
        top = results[:k]
        for rank, item in enumerate(top, start=1):
            item["rank"] = rank
            item["why"] = (
                f"余弦相似度 {item['score']:.3f}，全库第 {rank} 名；"
                f"与问句共享 {len(item['overlap'])} 处字词重叠"
            )
        return top

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
