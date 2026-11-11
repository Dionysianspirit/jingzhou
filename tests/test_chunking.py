import json
import subprocess
import sys
import unittest
from pathlib import Path

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


class ChunkingTerminationTests(unittest.TestCase):
    """A hard-split paragraph must always move forward.

    These run in a child process with a deadline: the failure they guard
    against is an endless loop that eats memory, which must show up as a
    failed assertion rather than a hung test run.
    """

    def run_chunker(self, expr: str, timeout: float = 20.0) -> list[int]:
        code = (
            "import json, sys\n"
            "try:\n"
            "    import resource\n"
            "    cap = 1024 * 1024 * 1024\n"
            "    resource.setrlimit(resource.RLIMIT_AS, (cap, cap))\n"
            "except Exception:\n"
            "    pass\n"
            "from app.chunking import chunk_text\n"
            f"chunks = {expr}\n"
            "print(json.dumps([len(c) for c in chunks]))\n"
        )
        root = Path(__file__).resolve().parents[1]
        try:
            done = subprocess.run(
                [sys.executable, "-c", code],
                cwd=root,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            self.fail("chunk_text did not terminate")
        self.assertEqual(done.returncode, 0, done.stderr[-400:])
        return json.loads(done.stdout)

    def test_early_sentence_end_then_long_unpunctuated_run(self):
        """Default settings: one early 「。」 followed by a table/formula-like run."""
        sizes = self.run_chunker(
            'chunk_text("第一句话。" + "这是一段没有句号的长内容例如表格公式或英文 " * 200)'
        )
        self.assertGreater(len(sizes), 1)
        self.assertLess(len(sizes), 10)
        self.assertTrue(all(size <= 2048 for size in sizes))

    def test_overlap_not_smaller_than_window_is_clamped(self):
        """A misconfigured OVERLAP_CHARS >= CHUNK_CHARS must not hang indexing."""
        sizes = self.run_chunker('chunk_text("字" * 5000, chunk_chars=400, overlap_chars=512)')
        self.assertTrue(all(size <= 400 for size in sizes))
        self.assertLessEqual(len(sizes), 5000 // 200 + 2)

    def test_hard_split_keeps_every_character(self):
        text = "序。" + "".join(chr(0x4E00 + i % 500) for i in range(3000))
        chunks = chunk_text(text, chunk_chars=300, overlap_chars=80)
        rebuilt = chunks[0]
        for nxt in chunks[1:]:
            overlap = next(n for n in range(min(len(rebuilt), len(nxt)), -1, -1) if rebuilt.endswith(nxt[:n]))
            rebuilt += nxt[overlap:]
        self.assertEqual(rebuilt, text)
        self.assertTrue(all(len(c) <= 300 for c in chunks))


if __name__ == "__main__":
    unittest.main()
