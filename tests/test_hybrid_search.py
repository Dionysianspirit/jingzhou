"""BM25 + fusion, and the hybrid search contract of DocStore.

Pure Python: embeddings are replaced by a controllable fake so the tests can
state exactly what the dense side "believes" and check what fusion does.
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from app.bm25 import LexicalIndex, bm25_scores, rank_map, rrf, tokenize
from app.store import DocStore, resolve_mode


class TokenizeTests(unittest.TestCase):
    def test_cjk_runs_become_bigrams(self):
        self.assertEqual(tokenize("线性相关"), ["线性", "性相", "相关"])

    def test_single_cjk_char_is_kept(self):
        self.assertEqual(tokenize("秩"), ["秩"])

    def test_latin_and_numbers_are_whole_lowercased_tokens(self):
        self.assertEqual(tokenize("rank(A) + dim ker(A) = n"), ["rank", "a", "dim", "ker", "a", "n"])
        self.assertEqual(tokenize("准确率 0.83"), ["准确", "确率", "0.83"])

    def test_mixed_text_breaks_cjk_runs_at_other_scripts(self):
        self.assertEqual(tokenize("特征值λ与BGE模型"), ["特征", "征值", "λ", "与", "bge", "模型"])

    def test_punctuation_and_empty_input_yield_nothing(self):
        self.assertEqual(tokenize("，。！？ \n"), [])
        self.assertEqual(tokenize(""), [])
        self.assertEqual(tokenize(None), [])


class Bm25Tests(unittest.TestCase):
    CHUNKS = [
        "向量加法按分量进行，数乘把每个分量同乘一个标量。",
        "秩-零化度定理：rank(A) + dim ker(A) = n。",
        "特征值与特征向量刻画线性映射的不变方向。",
    ]

    def setUp(self):
        self.indexes = {"d1": LexicalIndex(self.CHUNKS)}

    def test_exact_term_wins(self):
        scores = bm25_scores("秩-零化度定理", self.indexes)
        best = max(scores, key=scores.get)
        self.assertEqual(best, ("d1", 1))

    def test_chunks_without_shared_terms_are_absent(self):
        scores = bm25_scores("rank", self.indexes)
        self.assertEqual(set(scores), {("d1", 1)})

    def test_no_overlap_gives_no_scores(self):
        self.assertEqual(bm25_scores("量子纠缠", self.indexes), {})
        self.assertEqual(bm25_scores("", self.indexes), {})

    def test_rare_term_outweighs_common_term(self):
        chunks = ["定理 甲", "定理 乙", "定理 丙 行列式"]
        scores = bm25_scores("定理 行列式", {"d": LexicalIndex(chunks)})
        self.assertGreater(scores[("d", 2)], scores[("d", 0)])

    def test_statistics_span_the_selected_documents(self):
        """idf must come from the searched set, not from one document alone."""
        alone = bm25_scores("特征", {"a": LexicalIndex(["特征值"])})
        together = bm25_scores(
            "特征", {"a": LexicalIndex(["特征值"]), "b": LexicalIndex(["向量", "矩阵", "行列式"])}
        )
        self.assertGreater(together[("a", 0)], alone[("a", 0)])

    def test_rank_map_is_one_based_and_deterministic_on_ties(self):
        ranks = rank_map({("d", 2): 1.0, ("d", 0): 1.0, ("d", 1): 3.0})
        self.assertEqual(ranks, {("d", 1): 1, ("d", 0): 2, ("d", 2): 3})

    def test_rrf_rewards_agreement_between_lists(self):
        dense = {"a": 1, "b": 2, "c": 3}
        lexical = {"b": 1, "c": 2}
        fused = rrf([dense, lexical])
        self.assertEqual(max(fused, key=fused.get), "b")  # 2nd + 1st beats 1st + absent
        self.assertGreater(fused["c"], fused["a"])  # 3rd + 2nd beats 1st alone


def encoder_for(vectors: dict[str, list[float]]):
    """Fake BGE: each known text maps to a chosen unit vector."""

    async def encode(loop, texts):
        rows = []
        for text in texts:
            vec = np.array(vectors[text], dtype=np.float32)
            rows.append(vec / np.linalg.norm(vec))
        return np.vstack(rows)

    return encode


class HybridStoreTests(unittest.IsolatedAsyncioTestCase):
    """Three one-paragraph chunks; the fake encoder decides what "semantic" means."""

    A = "向量加法按分量进行。" + "甲" * 300
    B = "秩-零化度定理：rank(A) + dim ker(A) = n。" + "乙" * 300
    C = "特征值刻画线性映射的不变方向。" + "丙" * 300
    QUERY = "秩-零化度定理说了什么"

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = DocStore(Path(self.tmp.name))
        # Dense side is "wrong" on purpose: it prefers C, then A, and puts the
        # chunk that literally contains the theorem last.
        self.vectors = {
            self.A: [0.6, 0.8, 0.0],
            self.B: [0.2, 0.0, 0.98],
            self.C: [1.0, 0.1, 0.0],
            self.QUERY: [1.0, 0.0, 0.0],
        }

    def tearDown(self):
        self.tmp.cleanup()

    async def index(self):
        text = "\n\n".join([self.A, self.B, self.C])
        with (
            patch("app.embeddings.encode_async", encoder_for(self.vectors)),
            patch("app.store.CHUNK_CHARS", 400),
            patch("app.store.MIN_MERGE_CHARS", 10),
        ):
            n = await self.store.add("d1", "线性代数", text)
        self.assertEqual(n, 3)

    async def search(self, mode=None, k=3):
        with patch("app.embeddings.encode_async", encoder_for(self.vectors)):
            return await self.store.search(self.QUERY, ["d1"], k=k, mode=mode)

    async def test_dense_mode_keeps_the_original_cosine_ranking(self):
        await self.index()
        hits = await self.search("dense")
        self.assertEqual([h["chunk_idx"] for h in hits], [2, 0, 1])
        self.assertTrue(all(h["mode"] == "dense" for h in hits))
        self.assertIn("余弦相似度", hits[0]["why"])
        self.assertAlmostEqual(hits[0]["score"], 0.995, places=2)

    async def test_bm25_mode_ranks_by_keywords_only(self):
        await self.index()
        hits = await self.search("bm25")
        self.assertEqual(hits[0]["chunk_idx"], 1)
        self.assertEqual(hits[0]["bm25_rank"], 1)
        self.assertGreater(hits[0]["bm25_score"], 0)
        for h in hits:  # keyword mode never returns a chunk without a keyword match
            self.assertIsNotNone(h["bm25_rank"])

    async def test_hybrid_lifts_the_exact_match_the_dense_side_missed(self):
        await self.index()
        dense_pos = [h["chunk_idx"] for h in await self.search("dense")].index(1)
        hybrid = await self.search("hybrid")
        hybrid_pos = [h["chunk_idx"] for h in hybrid].index(1)
        self.assertEqual(dense_pos, 2)
        self.assertLess(hybrid_pos, dense_pos)
        hit = next(h for h in hybrid if h["chunk_idx"] == 1)
        self.assertEqual(hit["dense_rank"], 3)
        self.assertEqual(hit["bm25_rank"], 1)
        self.assertIn("融合排序", hit["why"])
        self.assertIn("关键词 BM25", hit["why"])

    async def test_score_stays_cosine_in_every_mode(self):
        await self.index()
        by_mode = {}
        for mode in ("dense", "bm25", "hybrid"):
            by_mode[mode] = {h["chunk_idx"]: h["score"] for h in await self.search(mode)}
        for chunk_idx, score in by_mode["dense"].items():
            for mode in ("bm25", "hybrid"):
                if chunk_idx in by_mode[mode]:
                    self.assertAlmostEqual(by_mode[mode][chunk_idx], score, places=6)

    async def test_hits_explain_themselves(self):
        await self.index()
        for hit in await self.search("hybrid"):
            for field in ("score", "rank", "why", "overlap", "dense_rank", "bm25_score", "fused_score"):
                self.assertIn(field, hit)
            self.assertIn("bm25_rank", hit)
        ranks = [h["rank"] for h in await self.search("hybrid")]
        self.assertEqual(ranks, [1, 2, 3])

    async def test_chunk_without_keyword_match_says_so(self):
        await self.index()
        self.vectors["完全无关的问句xyz"] = [1.0, 0.0, 0.0]
        with patch("app.embeddings.encode_async", encoder_for(self.vectors)):
            hits = await self.store.search("完全无关的问句xyz", ["d1"], k=3, mode="hybrid")
        self.assertEqual(len(hits), 3)  # dense still ranks everything
        self.assertTrue(all(h["bm25_rank"] is None for h in hits))
        self.assertIn("关键词未命中", hits[0]["why"])

    async def test_k_limits_results_and_unknown_docs_are_skipped(self):
        await self.index()
        self.assertEqual(len(await self.search("hybrid", k=2)), 2)
        with patch("app.embeddings.encode_async", encoder_for(self.vectors)):
            self.assertEqual(await self.store.search(self.QUERY, ["ghost"], k=3), [])

    async def test_index_is_rebuilt_from_disk_and_dropped_on_remove(self):
        await self.index()
        reloaded = DocStore(Path(self.tmp.name))
        self.assertEqual(reloaded.load_disk(), 1)
        self.assertIn("d1", reloaded.lexical)
        with patch("app.embeddings.encode_async", encoder_for(self.vectors)):
            hits = await reloaded.search(self.QUERY, ["d1"], k=3, mode="bm25")
        self.assertEqual(hits[0]["chunk_idx"], 1)
        await reloaded.remove("d1")
        self.assertNotIn("d1", reloaded.lexical)

    def test_resolve_mode_falls_back_to_hybrid(self):
        with patch("app.store.RETRIEVAL_MODE", "hybrid"):
            self.assertEqual(resolve_mode(None), "hybrid")
            self.assertEqual(resolve_mode("DENSE"), "dense")
            self.assertEqual(resolve_mode("nonsense"), "hybrid")
        with patch("app.store.RETRIEVAL_MODE", "dense"):
            self.assertEqual(resolve_mode(None), "dense")


if __name__ == "__main__":
    unittest.main()
