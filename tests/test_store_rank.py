import unittest

from app.chunking import chunk_text


class RetrievalContractTests(unittest.TestCase):
    """Contracts interviewers can read without loading BGE."""

    def test_sample_contains_eigen_and_rank_theorems(self):
        from pathlib import Path

        text = (Path(__file__).resolve().parents[1] / "data/sample/线性代数导引.txt").read_text(
            encoding="utf-8"
        )
        chunks = chunk_text(text, chunk_chars=600, overlap_chars=80)
        joined = "\n".join(chunks)
        self.assertIn("特征值", joined)
        self.assertIn("秩-零化度定理", joined)
        self.assertGreaterEqual(len(chunks), 4)


if __name__ == "__main__":
    unittest.main()
