"""Agent loop tests with a scripted fake LLM and an in-memory store.

No network, no BGE: decisions come from a queue, quiz generation and
explanations are monkeypatched. Existing RAG behaviour is covered by
test_chunking / test_store_rank.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app import agent, features
from app.llm import parse_json
from app.memory import AgentMemory, AgentStore, WeakPoint

HITS = [
    {
        "doc_id": "d1",
        "doc_name": "线性代数导引",
        "chunk_idx": 0,
        "text": "线性映射把向量送到向量，核空间与像空间满足秩-零化度定理。",
        "score": 0.82,
    },
    {
        "doc_id": "d1",
        "doc_name": "线性代数导引",
        "chunk_idx": 1,
        "text": "特征值与特征向量刻画线性映射的不变方向。",
        "score": 0.64,
    },
]

QUIZ = [
    {
        "q": "秩-零化度定理的内容是？",
        "options": ["rank + nullity = dim", "rank = nullity", "无关", "未知"],
        "answer": 0,
        "explanation": "秩与零化度之和为定义域维数。",
        "wrong_analysis": {
            "1": {"type": "公式遗忘", "reason": "把加法记成相等。"},
            "2": {"type": "概念混淆", "reason": "混淆概念。"},
            "3": {"type": "审题偏移", "reason": "审题偏移。"},
        },
    },
    {
        "q": "特征向量满足？",
        "options": ["Av = λv", "Av = 0", "无关", "未知"],
        "answer": 0,
        "explanation": "定义式 Av = λv。",
        "wrong_analysis": {
            "1": {"type": "概念混淆", "reason": "与核向量混淆。"},
            "2": {"type": "公式遗忘", "reason": "公式遗忘。"},
            "3": {"type": "推导断裂", "reason": "推导断裂。"},
        },
    },
]


class FakeStore:
    def __init__(self, hits=None, docs=("d1",)):
        self._hits = hits if hits is not None else HITS
        self._docs = list(docs)

    async def search(self, query, doc_ids, k=8):
        return self._hits[:k]

    async def get_chunk(self, doc_id, chunk_idx):
        for h in self._hits:
            if h["doc_id"] == doc_id and h["chunk_idx"] == chunk_idx:
                return {**h, "total": len(self._hits)}
        return None

    async def list_docs(self):
        return [{"id": d, "name": d, "chunks": 1} for d in self._docs]


def decisions(*items):
    """Queue-backed complete_json fake; each item is a decision dict or exception."""
    queue = list(items)

    async def fake(system, user, **kw):
        if not queue:
            raise AssertionError("complete_json called more times than scripted")
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    return fake, queue


async def text_fake(*args, **kw):
    return "讲解与总结文本"


def parse_events(lines):
    events = []
    for line in lines:
        if line.startswith("data: ") and "[DONE]" not in line:
            events.append(json.loads(line[6:]))
    return events


def types_of(events):
    return [e["type"] for e in events]


async def collect(mem, store, persist=None, max_steps=None):
    return [
        line async for line in agent.run_session(mem, store, persist=persist, max_steps=max_steps)
    ]


def make_memory(**kw):
    base = {"goal": "复习前三章，找出薄弱点", "doc_ids": ["d1"]}
    base.update(kw)
    return AgentMemory(**base)


class AgentFlowTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.persist = AgentStore(Path(self.tmp.name))

    def tearDown(self):
        self.tmp.cleanup()

    async def test_search_then_quiz_pauses_for_answers(self):
        fake, _queue = decisions(
            {
                "stage": "检索",
                "reason": "先检索前三章内容。",
                "action": "search_knowledge",
                "args": {"query": "线性映射 秩 特征值"},
            },
            {
                "stage": "诊断",
                "reason": "检测到需定位薄弱点，生成诊断测验。",
                "action": "create_quiz",
                "args": {"topic": "前三章核心概念", "count": 4},
            },
        )

        async def fake_gen_quiz(store, doc_ids, query, count):
            return QUIZ

        with (
            patch("app.llm.complete_json", fake),
            patch("app.llm.complete", text_fake),
            patch("app.features.gen_quiz", fake_gen_quiz),
        ):
            mem = make_memory()
            lines = await collect(mem, FakeStore(), persist=self.persist)

        events = parse_events(lines)
        self.assertEqual(types_of(events)[0], "planning")
        self.assertIn("tool_call", types_of(events))
        calls = [e for e in events if e["type"] == "tool_call"]
        self.assertEqual([c["tool"] for c in calls], ["search_knowledge", "create_quiz"])
        self.assertIn("awaiting_input", types_of(events))
        quiz_evt = next(e for e in events if e["type"] == "awaiting_input")
        self.assertEqual(len(quiz_evt["quiz"]), 2)
        for q in quiz_evt["quiz"]:
            self.assertNotIn("answer", q)  # answers must not leak to the learner
            self.assertNotIn("wrong_analysis", q)
        self.assertEqual(mem.status, "awaiting_answers")
        self.assertEqual(len(mem.steps), 2)
        self.assertTrue(all(s.ok for s in mem.steps))
        self.assertTrue(mem.sources)  # citations stay verifiable
        reloaded = self.persist.load(mem.session_id)
        self.assertEqual(reloaded.status, "awaiting_answers")
        self.assertEqual(reloaded.goal, mem.goal)

    async def test_max_steps_budget(self):
        counter = {"n": 0}

        async def fake(system, user, **kw):
            counter["n"] += 1
            return {
                "stage": "检索",
                "reason": "继续检索。",
                "action": "search_knowledge",
                "args": {"query": f"查询{counter['n']}"},
            }

        with patch("app.llm.complete_json", fake), patch("app.llm.complete", text_fake):
            mem = make_memory()
            lines = await collect(mem, FakeStore(), persist=self.persist, max_steps=3)

        self.assertEqual(mem.status, "finished")
        self.assertEqual(mem.stop_reason, "max_steps")
        self.assertEqual(mem.step_count, 3)
        final = next(e for e in parse_events(lines) if e["type"] == "final")
        self.assertEqual(final["stop_reason"], "max_steps")

    async def test_unknown_tool_is_rejected(self):
        fake, _ = decisions(
            {"stage": "检索", "reason": "误选不存在的工具。", "action": "hack_tool", "args": {}},
            {"stage": "结束", "reason": "已完成。", "action": "finish", "args": {}},
        )
        with patch("app.llm.complete_json", fake), patch("app.llm.complete", text_fake):
            mem = make_memory()
            await collect(mem, FakeStore(), persist=self.persist)

        rejected = mem.steps[0]
        self.assertFalse(rejected.ok)
        self.assertIn("不存在", rejected.result_summary)
        self.assertEqual(mem.status, "finished")
        self.assertEqual(mem.stop_reason, "completed")

    async def test_duplicate_call_is_blocked(self):
        same = {
            "stage": "检索",
            "reason": "重复请求。",
            "action": "search_knowledge",
            "args": {"query": "完全相同的查询"},
        }
        fake, _ = decisions(same, same, {"action": "finish", "reason": "结束", "args": {}})
        with patch("app.llm.complete_json", fake), patch("app.llm.complete", text_fake):
            mem = make_memory()
            await collect(mem, FakeStore(), persist=self.persist)

        self.assertTrue(mem.steps[0].ok)
        self.assertFalse(mem.steps[1].ok)
        self.assertIn("重复", mem.steps[1].result_summary)

    async def test_tool_failure_circuit_breaker(self):
        bad = {
            "stage": "溯源",
            "reason": "读取片段。",
            "action": "read_source",
            "args": {"doc_id": "ghost", "chunk_idx": 99},
        }
        fake, _ = decisions(bad, bad)
        with patch("app.llm.complete_json", fake), patch("app.llm.complete", text_fake):
            mem = make_memory()
            await collect(mem, FakeStore(), persist=self.persist)

        self.assertEqual(mem.status, "finished")
        self.assertIn("tool_failures", mem.stop_reason)
        self.assertLessEqual(mem.step_count, 2)  # bounded, no infinite loop

    async def test_no_hits_finishes_honestly(self):
        fake, _ = decisions(
            {
                "stage": "检索",
                "reason": "检索目标内容。",
                "action": "search_knowledge",
                "args": {"query": "完全无关的查询xyz"},
            },
            {
                "stage": "结束",
                "reason": "资料中无相关内容，如实结束。",
                "action": "finish",
                "args": {},
            },
        )
        with patch("app.llm.complete_json", fake), patch("app.llm.complete", text_fake):
            mem = make_memory()
            lines = await collect(mem, FakeStore(hits=[]), persist=self.persist)

        result = next(e for e in parse_events(lines) if e["type"] == "tool_result")
        self.assertIn("无命中", result["summary"])
        self.assertEqual(mem.status, "finished")
        self.assertEqual(mem.stop_reason, "completed")

    async def test_missing_docs_stops_without_llm(self):
        counter = {"n": 0}

        async def fake(system, user, **kw):
            counter["n"] += 1
            raise AssertionError("LLM must not be called when no docs exist")

        with patch("app.llm.complete_json", fake), patch("app.llm.complete", fake):
            mem = make_memory(doc_ids=["ghost"])
            lines = await collect(mem, FakeStore(docs=[]), persist=self.persist)

        final = next(e for e in parse_events(lines) if e["type"] == "final")
        self.assertEqual(final["stop_reason"], "no_material")
        self.assertEqual(mem.status, "finished")
        self.assertEqual(counter["n"], 0)

    async def test_resume_scores_answers_then_remediates(self):
        fake, _ = decisions(
            {
                "stage": "讲解",
                "reason": "针对薄弱点回查讲解。",
                "action": "answer_question",
                "args": {"question": "秩-零化度定理和特征值的区别"},
            },
            {"stage": "结束", "reason": "薄弱点已覆盖，总结。", "action": "finish", "args": {}},
        )
        with patch("app.llm.complete_json", fake), patch("app.llm.complete", text_fake):
            mem = make_memory(quiz=[dict(q) for q in QUIZ], status="awaiting_answers")
            mem.pending_answers = [0, 1]  # q1 correct, q2 wrong (chooses option 1)
            mem.status = "running"
            lines = await collect(mem, FakeStore(), persist=self.persist)

        events = parse_events(lines)
        self.assertEqual(mem.steps[0].tool, "analyze_mistakes")
        self.assertTrue(mem.steps[0].ok)
        self.assertEqual(len(mem.weaknesses), 1)
        self.assertEqual(mem.weaknesses[0].error_type, "概念混淆")
        explain = next(
            e for e in events if e["type"] == "tool_result" and e["tool"] == "answer_question"
        )
        self.assertEqual(explain["data"]["answer"], "讲解与总结文本")
        final = next(e for e in events if e["type"] == "final")
        self.assertEqual(len(final["weaknesses"]), 1)
        self.assertEqual(mem.status, "finished")
        self.assertEqual(mem.stop_reason, "completed")

    async def test_llm_cannot_self_invoke_analyze(self):
        fake, _ = decisions(
            {
                "stage": "判分",
                "reason": "试图自行判分。",
                "action": "analyze_mistakes",
                "args": {"answers": [0, 0]},
            },
            {"stage": "结束", "reason": "结束。", "action": "finish", "args": {}},
        )
        with patch("app.llm.complete_json", fake), patch("app.llm.complete", text_fake):
            mem = make_memory()
            await collect(mem, FakeStore(), persist=self.persist)

        self.assertFalse(mem.steps[0].ok)
        self.assertIn("拒绝", mem.steps[0].result_summary)
        self.assertEqual(mem.weaknesses, [])

    async def test_decide_failure_ends_session_with_error(self):
        fake, _ = decisions(ValueError("bad json"), ValueError("bad json"))
        with patch("app.llm.complete_json", fake), patch("app.llm.complete", text_fake):
            mem = make_memory()
            lines = await collect(mem, FakeStore(), persist=self.persist)

        events = parse_events(lines)
        self.assertIn("error", types_of(events))
        self.assertEqual(mem.status, "finished")
        self.assertEqual(mem.stop_reason, "error")


class UnitTests(unittest.TestCase):
    def test_memory_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = AgentStore(Path(tmp))
            mem = make_memory()
            mem.record_step("search_knowledge", {"query": "x"}, "理由", True, "命中 2 条")
            mem.weaknesses.append(
                WeakPoint(topic="特征值", error_type="概念混淆", reason="混淆", question="q")
            )
            store.save(mem)
            loaded = store.load(mem.session_id)
            self.assertEqual(loaded.goal, mem.goal)
            self.assertEqual(loaded.step_count, 1)
            self.assertEqual(loaded.steps[0].tool, "search_knowledge")
            self.assertEqual(loaded.weaknesses[0].error_type, "概念混淆")
            self.assertIsNone(store.load("nonexistent"))

    def test_public_quiz_hides_answers(self):
        mem = make_memory(quiz=QUIZ)
        public = mem.public_quiz()
        self.assertEqual(public[0]["q"], QUIZ[0]["q"])
        self.assertNotIn("answer", public[0])

    def test_normalize_questions_cleans_junk(self):
        raw = [
            {
                "q": "好题",
                "options": ["甲", "乙"],
                "answer": "1",
                "wrong_analysis": {"0": {"type": "瞎编的类型", "reason": "r"}},
            },
            "not a dict",
            {"q": "缺选项", "options": []},
        ]
        out = features.normalize_questions(raw)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["answer"], 1)
        self.assertEqual(out[0]["wrong_analysis"]["0"]["type"], "概念混淆")

    def test_parse_json_strips_fences(self):
        self.assertEqual(parse_json('```json\n{"a": 1}\n```'), {"a": 1})

    def test_list_sessions_orders_and_skips_corrupt(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = AgentStore(Path(tmp))

            def write(sid, **fields):
                record = {"session_id": sid, "goal": sid, **fields}
                (Path(tmp) / f"{sid}.json").write_text(
                    json.dumps(record, ensure_ascii=False), encoding="utf-8"
                )

            write("aaa", status="awaiting_answers", updated_at=100, step_count=2)
            write("bbb", status="finished", updated_at=200, step_count=5)
            (Path(tmp) / "bad.json").write_text("not-json{", encoding="utf-8")

            items = store.list_sessions()
            self.assertEqual([i["session_id"] for i in items], ["bbb", "aaa"])
            self.assertEqual(items[0]["status"], "finished")
            self.assertEqual(items[0]["step_count"], 5)
            self.assertEqual(items[1]["weakness_count"], 0)

    def test_list_endpoint_returns_summaries(self):
        from fastapi.testclient import TestClient

        import app.main as main_mod

        with tempfile.TemporaryDirectory() as tmp:
            st = AgentStore(Path(tmp))
            mem = make_memory(goal="恢复用会话")
            mem.status = "awaiting_answers"
            mem.record_step("search_knowledge", {"query": "x"}, "理由", True, "命中 2 条")
            st.save(mem)

            with patch.object(main_mod, "agent_store", st):
                client = TestClient(main_mod.app)
                resp = client.get("/api/agent")
                self.assertEqual(resp.status_code, 200)
                body = resp.json()
                self.assertEqual(len(body), 1)
                self.assertEqual(body[0]["session_id"], mem.session_id)
                self.assertEqual(body[0]["status"], "awaiting_answers")
                self.assertEqual(body[0]["step_count"], 1)
                # summaries must not carry quiz payloads or answers
                self.assertNotIn("quiz", body[0])
                self.assertNotIn("steps", body[0])


if __name__ == "__main__":
    unittest.main()
