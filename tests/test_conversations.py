"""Multi-turn chat: persistence, history-aware retrieval, and the API contract.

A scripted fake stands in for the OpenAI-compatible client and for the
DocStore, so the tests can assert exactly what was searched and what the
model was shown.
"""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

import app.main as main_mod
from app.conversations import (
    HISTORY_ASSISTANT_CHARS,
    HISTORY_TURNS,
    Conversation,
    ConversationStore,
    standalone_query,
)

HITS = [
    {
        "doc_id": "d1",
        "doc_name": "线性代数导引",
        "chunk_idx": 2,
        "text": "秩-零化度定理：rank(A) + dim ker(A) = n。",
        "score": 0.81,
        "why": "融合排序第 1 名",
        "overlap": ["定理"],
    }
]


class FakeStore:
    def __init__(self):
        self.queries: list[str] = []

    async def search(self, query, doc_ids, k=8, mode=None):
        self.queries.append(query)
        return HITS


class FakeLLM:
    """Streams scripted answers; non-stream calls return the scripted rewrite."""

    def __init__(self, answers, rewrite="秩-零化度定理与齐次方程组的解有什么关系？"):
        self.answers = list(answers)
        self.rewrite = rewrite
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    async def create(self, **kw):
        self.calls.append(kw)
        if not kw.get("stream"):
            if isinstance(self.rewrite, Exception):
                raise self.rewrite
            message = SimpleNamespace(content=self.rewrite)
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer

        async def stream():
            for piece in (answer[:4], answer[4:]):
                if piece:
                    delta = SimpleNamespace(content=piece)
                    yield SimpleNamespace(choices=[SimpleNamespace(delta=delta)])

        return stream()

    @property
    def stream_calls(self):
        return [c for c in self.calls if c.get("stream")]

    @property
    def rewrite_calls(self):
        return [c for c in self.calls if not c.get("stream")]


def events_of(resp):
    out = []
    for line in resp.text.splitlines():
        if line.startswith("data: ") and "[DONE]" not in line:
            out.append(json.loads(line[6:]))
    return out


def merged(events):
    data = {"text": ""}
    for e in events:
        if "text" in e:
            data["text"] += e["text"]
        else:
            data.update(e)
    return data


class ConversationStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = ConversationStore(Path(self.tmp.name))

    def tearDown(self):
        self.tmp.cleanup()

    def exchange(self, conv, q="秩-零化度定理在说什么？", a="它说 rank + nullity = n。"):
        conv.add_exchange(q, a, sources=[{"doc_id": "d1", "chunk": 2}], search_query=q, doc_ids=["d1"])

    def test_roundtrip_keeps_turns_sources_and_title(self):
        conv = Conversation()
        self.exchange(conv)
        self.store.save(conv)
        loaded = self.store.load(conv.conversation_id)
        self.assertEqual([t.role for t in loaded.turns], ["user", "assistant"])
        self.assertEqual(loaded.turns[1].sources, [{"doc_id": "d1", "chunk": 2}])
        self.assertEqual(loaded.title, "秩-零化度定理在说什么？")
        self.assertEqual(loaded.doc_ids, ["d1"])

    def test_long_first_question_is_clipped_into_the_title(self):
        conv = Conversation()
        self.exchange(conv, q="请" + "很长的问题" * 20)
        self.assertLessEqual(len(conv.title), 31)
        self.assertTrue(conv.title.endswith("…"))

    def test_list_is_newest_first_and_skips_corrupt_and_empty(self):
        old, new, empty = Conversation(), Conversation(), Conversation()
        self.exchange(old, q="旧问题")
        self.exchange(new, q="新问题")
        old.updated_at, new.updated_at = 100, 200
        for conv in (old, new, empty):
            self.store.save(conv)
        (Path(self.tmp.name) / "bad.json").write_text("not-json{", encoding="utf-8")
        items = self.store.list_conversations()
        self.assertEqual([i["title"] for i in items], ["新问题", "旧问题"])
        self.assertEqual(items[0]["exchanges"], 1)
        self.assertNotIn("turns", items[0])

    def test_delete_and_unsafe_ids(self):
        conv = Conversation()
        self.exchange(conv)
        self.store.save(conv)
        self.assertTrue(self.store.delete(conv.conversation_id))
        self.assertFalse(self.store.delete(conv.conversation_id))
        self.assertIsNone(self.store.load("../../etc/passwd"))
        self.assertFalse(self.store.delete("../escape"))
        with self.assertRaises(ValueError):
            self.store.save(Conversation(conversation_id="../escape"))

    def test_history_is_bounded_and_clipped(self):
        conv = Conversation()
        for i in range(5):
            self.exchange(conv, q=f"问题{i}", a="答" * 5000)
        messages = conv.history_messages()
        self.assertEqual(len(messages), HISTORY_TURNS)
        self.assertEqual(messages[-1]["role"], "assistant")
        self.assertLessEqual(len(messages[-1]["content"]), HISTORY_ASSISTANT_CHARS + 1)
        self.assertEqual(messages[-2]["content"], "问题4")


class StandaloneQueryTests(unittest.IsolatedAsyncioTestCase):
    def conversation(self):
        conv = Conversation()
        conv.add_exchange(
            "秩-零化度定理在说什么？", "rank + nullity = n。", sources=[], search_query="", doc_ids=["d1"]
        )
        return conv

    async def test_first_turn_is_not_rewritten_and_costs_no_llm_call(self):
        async def boom(*a, **kw):
            raise AssertionError("no LLM call expected on the first turn")

        with patch("app.llm.complete", boom):
            self.assertEqual(await standalone_query(None, " 什么是秩？ "), ("什么是秩？", False))
            self.assertEqual(await standalone_query(Conversation(), "什么是秩？"), ("什么是秩？", False))

    async def test_follow_up_is_rewritten_with_history_in_the_prompt(self):
        seen = {}

        async def fake(system, user, **kw):
            seen["user"] = user
            return "「秩-零化度定理与齐次方程组有什么关系？」\n"

        with patch("app.llm.complete", fake):
            query, rewritten = await standalone_query(self.conversation(), "那它和齐次方程有什么关系？")
        self.assertEqual(query, "秩-零化度定理与齐次方程组有什么关系？")  # quotes/newline stripped
        self.assertTrue(rewritten)
        self.assertIn("秩-零化度定理在说什么？", seen["user"])
        self.assertIn("那它和齐次方程有什么关系？", seen["user"])

    async def test_llm_failure_falls_back_to_previous_question_plus_follow_up(self):
        async def broken(*a, **kw):
            raise RuntimeError("network down")

        with patch("app.llm.complete", broken):
            query, rewritten = await standalone_query(self.conversation(), "再举个例子")
        self.assertEqual(query, "秩-零化度定理在说什么？ 再举个例子")
        self.assertTrue(rewritten)

    async def test_empty_or_runaway_rewrite_is_discarded(self):
        for junk in ("", "   ", "啰" * 400):

            async def fake(system, user, _junk=junk, **kw):
                return _junk

            with patch("app.llm.complete", fake):
                query, _ = await standalone_query(self.conversation(), "再举个例子")
            self.assertEqual(query, "秩-零化度定理在说什么？ 再举个例子")

    async def test_unchanged_rewrite_is_reported_as_not_rewritten(self):
        async def echo(system, user, **kw):
            return "特征值的定义是什么？"

        with patch("app.llm.complete", echo):
            self.assertEqual(
                await standalone_query(self.conversation(), "特征值的定义是什么？"),
                ("特征值的定义是什么？", False),
            )


class ChatApiTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.chat_store = ConversationStore(Path(self.tmp.name))
        self.store = FakeStore()
        self.llm = FakeLLM(["第一轮回答。[来源: 线性代数导引 片段2]", "第二轮回答，承接上文。"])
        self.patches = [
            patch.object(main_mod, "store", self.store),
            patch.object(main_mod, "chat_store", self.chat_store),
            patch("app.llm.llm", self.llm),
        ]
        for p in self.patches:
            p.start()
        self.client = TestClient(main_mod.app)

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def ask(self, query, **extra):
        return self.client.post("/api/chat", json={"query": query, "doc_ids": ["d1"], **extra})

    def test_stateless_call_behaves_as_before_and_stores_nothing(self):
        resp = self.ask("秩-零化度定理在说什么？")
        self.assertEqual(resp.status_code, 200)
        data = merged(events_of(resp))
        self.assertEqual(data["text"], "第一轮回答。[来源: 线性代数导引 片段2]")
        self.assertEqual(data["sources"][0]["chunk"], 2)
        self.assertNotIn("conversation_id", data)
        self.assertEqual(self.chat_store.list_conversations(), [])
        self.assertEqual(self.llm.rewrite_calls, [])
        self.assertTrue(resp.text.rstrip().endswith("data: [DONE]"))

    def test_remember_starts_a_conversation(self):
        data = merged(events_of(self.ask("秩-零化度定理在说什么？", remember=True)))
        cid = data["conversation_id"]
        self.assertFalse(data["rewritten"])
        self.assertEqual(data["search_query"], "秩-零化度定理在说什么？")
        stored = self.chat_store.load(cid)
        self.assertEqual([t.role for t in stored.turns], ["user", "assistant"])
        self.assertEqual(stored.turns[1].sources[0]["doc_id"], "d1")

    def test_follow_up_searches_the_rewritten_question_and_sends_history(self):
        cid = merged(events_of(self.ask("秩-零化度定理在说什么？", remember=True)))["conversation_id"]
        data = merged(events_of(self.ask("那它和齐次方程有什么关系？", conversation_id=cid)))

        self.assertEqual(self.store.queries[0], "秩-零化度定理在说什么？")
        self.assertEqual(self.store.queries[1], "秩-零化度定理与齐次方程组的解有什么关系？")
        self.assertTrue(data["rewritten"])
        self.assertEqual(data["search_query"], self.store.queries[1])
        self.assertEqual(data["conversation_id"], cid)

        first, second = self.llm.stream_calls
        self.assertEqual([m["role"] for m in first["messages"]], ["system", "user"])
        self.assertEqual(
            [m["role"] for m in second["messages"]], ["system", "user", "assistant", "user"]
        )
        self.assertEqual(second["messages"][1]["content"], "秩-零化度定理在说什么？")
        self.assertIn("第一轮回答", second["messages"][2]["content"])
        # the model sees the learner's own words, with this turn's retrieved context
        self.assertIn("用户问题：那它和齐次方程有什么关系？", second["messages"][3]["content"])
        self.assertIn("rank(A) + dim ker(A) = n", second["messages"][3]["content"])
        self.assertIn("对话历史只用于理解追问", second["messages"][0]["content"])
        self.assertNotIn("对话历史只用于理解追问", first["messages"][0]["content"])

        stored = self.chat_store.load(cid)
        self.assertEqual(len(stored.turns), 4)
        self.assertEqual(stored.turns[3].search_query, self.store.queries[1])
        self.assertEqual(stored.title, "秩-零化度定理在说什么？")

    def test_unknown_conversation_is_404_before_any_llm_call(self):
        resp = self.ask("追问", conversation_id="c0000000000")
        self.assertEqual(resp.status_code, 404)
        self.assertEqual(self.llm.calls, [])

    def test_unsafe_conversation_id_is_rejected(self):
        self.assertEqual(self.ask("追问", conversation_id="../../etc/passwd").status_code, 422)

    def test_failed_generation_reports_the_error_and_stores_nothing(self):
        self.llm.answers = [RuntimeError("upstream exploded")]
        data = merged(events_of(self.ask("秩-零化度定理在说什么？", remember=True)))
        self.assertIn("upstream exploded", data["error"])
        self.assertNotIn("conversation_id", data)
        self.assertEqual(self.chat_store.list_conversations(), [])

    def test_conversation_endpoints(self):
        cid = merged(events_of(self.ask("秩-零化度定理在说什么？", remember=True)))["conversation_id"]
        listing = self.client.get("/api/conversations").json()
        self.assertEqual([c["conversation_id"] for c in listing], [cid])
        self.assertEqual(listing[0]["exchanges"], 1)

        full = self.client.get(f"/api/conversations/{cid}").json()
        self.assertEqual(full["turns"][0]["content"], "秩-零化度定理在说什么？")
        self.assertEqual(full["turns"][1]["sources"][0]["chunk"], 2)

        self.assertEqual(self.client.delete(f"/api/conversations/{cid}").status_code, 200)
        self.assertEqual(self.client.get(f"/api/conversations/{cid}").status_code, 404)
        self.assertEqual(self.client.delete(f"/api/conversations/{cid}").status_code, 404)
        self.assertEqual(self.client.get("/api/conversations").json(), [])


if __name__ == "__main__":
    unittest.main()
