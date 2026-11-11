"""API-boundary safety: ids that reach the filesystem, and honest status codes.

No BGE, no network: the store is rooted in a temp dir and embeddings are faked.
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient

import app.main as main_mod
from app.store import DocStore


async def fake_encode(loop, texts):
    """Deterministic stand-in for BGE: one L2-normalized vector per text."""
    out = np.zeros((len(texts), 8), dtype=np.float32)
    for row, text in enumerate(texts):
        for ch in text:
            out[row, ord(ch) % 8] += 1.0
        norm = np.linalg.norm(out[row])
        if norm:
            out[row] /= norm
    return out


class RemoveTraversalTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.store = DocStore(self.root / "store")
        self.victim = self.root / "victim"
        self.victim.mkdir()
        (self.victim / "keep.txt").write_text("must survive", encoding="utf-8")
        self.patches = [
            patch.object(main_mod, "store", self.store),
            patch("app.embeddings.encode_async", fake_encode),
        ]
        for p in self.patches:
            p.start()
        self.client = TestClient(main_mod.app)

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def test_remove_rejects_parent_traversal(self):
        resp = self.client.post("/api/remove", json={"doc_id": "../victim"})
        self.assertEqual(resp.status_code, 422)
        self.assertTrue((self.victim / "keep.txt").exists())

    def test_remove_rejects_absolute_path(self):
        resp = self.client.post("/api/remove", json={"doc_id": str(self.victim)})
        self.assertEqual(resp.status_code, 422)
        self.assertTrue((self.victim / "keep.txt").exists())

    def test_store_refuses_unsafe_ids_even_without_the_api(self):
        """Defense in depth: the store itself never resolves an unsafe id to a path."""
        import asyncio

        for bad in ("../victim", str(self.victim), "a/b", "..", "", "x" * 65):
            with self.assertRaises(ValueError, msg=bad):
                asyncio.run(self.store.remove(bad))
        self.assertTrue((self.victim / "keep.txt").exists())

    def test_index_then_remove_roundtrip_still_works(self):
        resp = self.client.post(
            "/api/index", json={"doc_id": "doc-1", "name": "讲义", "text": "向量加法按分量进行。"}
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertTrue((self.root / "store" / "doc-1" / "meta.json").exists())
        resp = self.client.post("/api/remove", json={"doc_id": "doc-1"})
        self.assertEqual(resp.status_code, 200)
        self.assertFalse((self.root / "store" / "doc-1").exists())
        self.assertEqual(self.client.get("/api/docs").json(), [])

    def test_agent_session_id_is_validated(self):
        for path in ("/api/agent/..%2F..%2Fetc", "/api/agent/not-a-session-id"):
            resp = self.client.get(path)
            self.assertIn(resp.status_code, (404, 422), path)


class NoKeyStatusTests(unittest.TestCase):
    """README contract: without LLM_API_KEY, generation endpoints answer 503."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = DocStore(Path(self.tmp.name) / "store")
        self.patches = [
            patch.object(main_mod, "store", self.store),
            patch("app.embeddings.encode_async", fake_encode),
            patch("app.llm.llm", None),
        ]
        for p in self.patches:
            p.start()
        self.client = TestClient(main_mod.app)
        self.client.post(
            "/api/index", json={"doc_id": "doc-1", "name": "讲义", "text": "向量加法按分量进行。"}
        )

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def test_generation_endpoints_return_503_not_500(self):
        body = {"doc_ids": ["doc-1"]}
        for path in (
            "/api/flashcards",
            "/api/table",
            "/api/infographic",
            "/api/mindmap",
            "/api/report",
            "/api/study-guide",
        ):
            resp = self.client.post(path, json=body)
            self.assertEqual(resp.status_code, 503, f"{path}: {resp.status_code} {resp.text}")

    def test_quiz_falls_back_to_sample_without_key(self):
        resp = self.client.post("/api/quiz", json={"doc_ids": ["doc-1"]})
        self.assertEqual(resp.status_code, 200, resp.text)
        data = resp.json()
        self.assertTrue(data.get("fallback"))
        self.assertTrue(data["questions"])

    def test_search_works_without_key(self):
        resp = self.client.post("/api/search", json={"query": "向量加法", "doc_ids": ["doc-1"]})
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertEqual(len(resp.json()["hits"]), 1)


class CorsDefaultTests(unittest.TestCase):
    def test_cross_origin_is_not_allowed_by_default(self):
        client = TestClient(main_mod.app)
        resp = client.options(
            "/api/remove",
            headers={
                "Origin": "https://evil.example",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )
        self.assertNotIn("access-control-allow-origin", {k.lower() for k in resp.headers})


if __name__ == "__main__":
    unittest.main()
