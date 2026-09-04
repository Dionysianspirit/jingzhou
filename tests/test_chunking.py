import unittest

from app.chunking import chunk_text


class ChunkingTests(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(chunk_text(""), [])
        self.assertEqual(chunk_text("   \n\n  "), [])

    def test_keeps_short_doc(self):
        text = "向量加法按分量进行。"
        self.assertEqual(chunk_text(text), [text])

    def test_splits_on_paragraphs(self):
        paras = [f"段落{i}。" + ("字" * 80) for i in range(12)]
        chunks = chunk_text("\n\n".join(paras), chunk_chars=300, overlap_chars=40)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(c) <= 400 for c in chunks))

    def test_hard_split_long_paragraph(self):
        text = "。".join(["这是一句足够长的中文句子用于切分"] * 80)
        chunks = chunk_text(text, chunk_chars=200, overlap_chars=40)
        self.assertGreater(len(chunks), 2)
        self.assertTrue(any("这是一句" in c for c in chunks))

    def test_merges_tiny_tail(self):
        text = ("甲" * 180) + "\n\n尾部很短"
        chunks = chunk_text(text, chunk_chars=500, min_merge_chars=50)
        self.assertEqual(len(chunks), 1)
        self.assertIn("尾部很短", chunks[0])


if __name__ == "__main__":
    unittest.main()
