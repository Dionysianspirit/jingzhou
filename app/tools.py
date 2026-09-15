"""Agent tools: thin wrappers over the existing RAG/quiz/guide capabilities.

Every tool declares a Pydantic args model (whitelist + validation) and
returns a ToolResult with a short user-facing summary. No new retrieval,
embedding or generation logic lives here — search/read go straight to
DocStore, quiz/guide call app.features, mistake analysis is deterministic.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from app import features, llm
from app.memory import WeakPoint

LOW_SCORE_FLOOR = 0.2


class ToolResult(BaseModel):
    ok: bool
    summary: str
    data: Any = None


class ToolContext:
    """What a tool may touch: the shared DocStore plus the session memory."""

    def __init__(self, store, memory):
        self.store = store
        self.memory = memory


@dataclass
class Tool:
    name: str
    description: str
    args_model: type[BaseModel]
    func: Callable[[BaseModel, ToolContext], Awaitable[ToolResult]]

    def spec(self) -> str:
        props = self.args_model.model_json_schema().get("properties", {})
        fields = ", ".join(
            f"{name}:{info.get('type', 'string')}{'=可选' if 'default' in info else ''}"
            for name, info in props.items()
        )
        return f"- {self.name}({fields}): {self.description}"


class SearchArgs(BaseModel):
    query: str = Field(min_length=2, max_length=256)
    k: int = Field(default=8, ge=1, le=12)


class ReadSourceArgs(BaseModel):
    doc_id: str = Field(min_length=1, max_length=64)
    chunk_idx: int = Field(ge=0)


class CreateQuizArgs(BaseModel):
    topic: str = Field(min_length=2, max_length=256)
    count: int = Field(default=4, ge=2, le=6)


class AnalyzeMistakesArgs(BaseModel):
    answers: list[int] = Field(min_length=1)


class AnswerQuestionArgs(BaseModel):
    question: str = Field(min_length=2, max_length=512)


class CreateStudyGuideArgs(BaseModel):
    topic: str = Field(min_length=2, max_length=256)


async def search_knowledge(args: SearchArgs, ctx: ToolContext) -> ToolResult:
    hits = await ctx.store.search(args.query, ctx.memory.doc_ids, k=args.k)
    ctx.memory.add_sources(hits)
    if not hits:
        return ToolResult(
            ok=True,
            summary="检索无命中：所选资料中没有与该查询相关的内容。",
            data={"hits": [], "empty": True},
        )
    top = hits[0]
    low = all(h.get("score", 0.0) < LOW_SCORE_FLOOR for h in hits)
    summary = (
        f"命中 {len(hits)} 条，最相关：{top['doc_name']} 片段{top['chunk_idx']}"
        f"（相似度 {top['score']:.3f}）"
    )
    if low:
        summary += "。注意：所有命中相似度偏低，资料中可能没有真正相关的内容。"
    return ToolResult(
        ok=True,
        summary=summary,
        data={
            "hits": [
                {
                    "doc_id": h["doc_id"],
                    "doc_name": h["doc_name"],
                    "chunk_idx": h["chunk_idx"],
                    "score": round(float(h["score"]), 3),
                    "preview": h["text"][:200],
                }
                for h in hits
            ],
            "low_relevance": low,
        },
    )


async def read_source(args: ReadSourceArgs, ctx: ToolContext) -> ToolResult:
    item = await ctx.store.get_chunk(args.doc_id, args.chunk_idx)
    if item is None:
        return ToolResult(ok=False, summary=f"片段不存在：{args.doc_id}#{args.chunk_idx}")
    ctx.memory.add_sources(
        [
            {
                "doc_id": item["doc_id"],
                "doc_name": item["doc_name"],
                "chunk_idx": item["chunk_idx"],
                "score": 1.0,
            }
        ]
    )
    return ToolResult(
        ok=True,
        summary=f"已读取 {item['doc_name']} 片段{item['chunk_idx']}（共 {item['total']} 片段，{len(item['text'])} 字）",
        data={
            "doc_id": item["doc_id"],
            "doc_name": item["doc_name"],
            "chunk_idx": item["chunk_idx"],
            "text": item["text"][:1500],
        },
    )


async def create_quiz(args: CreateQuizArgs, ctx: ToolContext) -> ToolResult:
    questions = await features.gen_quiz(ctx.store, ctx.memory.doc_ids, args.topic, args.count)
    if not questions:
        return ToolResult(ok=False, summary="测验生成失败：未得到有效题目。")
    ctx.memory.quiz = questions
    return ToolResult(
        ok=True,
        summary=f"已生成 {len(questions)} 道诊断题（主题：{args.topic}）。",
        data={"questions": questions, "topic": args.topic},
    )


async def analyze_mistakes(args: AnalyzeMistakesArgs, ctx: ToolContext) -> ToolResult:
    quiz = ctx.memory.quiz or []
    if len(args.answers) != len(quiz):
        return ToolResult(
            ok=False,
            summary=f"答案数量不符：收到 {len(args.answers)} 份，题目为 {len(quiz)} 道。",
        )
    per_question = []
    score = 0
    for i, (q, ans) in enumerate(zip(quiz, args.answers)):
        ans = max(0, min(int(ans), len(q.get("options", [])) - 1))
        correct = ans == q.get("answer")
        if correct:
            score += 1
        else:
            wa = (q.get("wrong_analysis") or {}).get(str(ans)) or {}
            ctx.memory.weaknesses.append(
                WeakPoint(
                    topic=q.get("q", "")[:40],
                    error_type=wa.get("type", ""),
                    reason=wa.get("reason", "") or q.get("explanation", ""),
                    question=q.get("q", ""),
                )
            )
        per_question.append({"index": i, "correct": correct, "chosen": ans})
    ctx.memory.pending_answers = None
    return ToolResult(
        ok=True,
        summary=f"判分完成：{score}/{len(quiz)} 正确，累计薄弱点 {len(ctx.memory.weaknesses)} 个。",
        data={
            "score": score,
            "total": len(quiz),
            "per_question": per_question,
            "weaknesses": [w.model_dump() for w in ctx.memory.weaknesses],
        },
    )


async def answer_question(args: AnswerQuestionArgs, ctx: ToolContext) -> ToolResult:
    results = await ctx.store.search(args.question, ctx.memory.doc_ids, k=6)
    if not results:
        return ToolResult(
            ok=True,
            summary="检索无命中，无法基于资料回答该问题。",
            data={"answer": "", "empty": True},
        )
    ctx.memory.add_sources(results)
    system = (
        "你是学习助手，负责针对性讲解。仅基于提供的文档片段回答，不要使用外部知识；"
        "如果片段中没有相关信息，诚实说明。回答末尾列出引用，格式为 [来源: 文档名 片段N]。"
    )
    answer = await llm.complete(
        system,
        f"学习者的问题：{args.question}\n\n相关薄弱背景："
        f"{'; '.join(w.topic for w in ctx.memory.weaknesses[:5]) or '无'}\n\n文档内容：\n\n"
        + llm.build_ctx(results),
        temperature=0.3,
        max_tokens=1536,
    )
    return ToolResult(
        ok=True,
        summary=f"已生成针对性讲解（{len(results)} 条来源支撑）。",
        data={
            "answer": answer,
            "sources": [
                {"doc_id": r["doc_id"], "name": r["doc_name"], "chunk": r["chunk_idx"]}
                for r in results
            ],
        },
    )


async def create_study_guide(args: CreateStudyGuideArgs, ctx: ToolContext) -> ToolResult:
    guide = await features.gen_guide(ctx.store, ctx.memory.doc_ids, args.topic)
    points = guide.get("key_points") or []
    for p in points:
        if isinstance(p, str) and p not in ctx.memory.findings:
            ctx.memory.findings.append(p)
    ctx.memory.add_sources(guide.get("sources") or [])
    return ToolResult(
        ok=True,
        summary=f"已提炼学习指南：「{guide.get('title') or args.topic}」，共 {len(points)} 个要点。",
        data=guide,
    )


def build_tools(store) -> dict[str, Tool]:
    """Whitelist of tools bound to one DocStore (the app's global store)."""
    return {
        "search_knowledge": Tool(
            "search_knowledge",
            "在所选教材中语义检索，返回命中片段、相似度与预览",
            SearchArgs,
            search_knowledge,
        ),
        "read_source": Tool(
            "read_source",
            "读取指定文档片段的原文（用于核对出处）",
            ReadSourceArgs,
            read_source,
        ),
        "create_study_guide": Tool(
            "create_study_guide",
            "提炼学习指南：要点、易错点、思考题（用于梳理知识结构）",
            CreateStudyGuideArgs,
            create_study_guide,
        ),
        "create_quiz": Tool(
            "create_quiz",
            "针对主题生成诊断选择题（含错因分类），生成后将交给学习者作答",
            CreateQuizArgs,
            create_quiz,
        ),
        "analyze_mistakes": Tool(
            "analyze_mistakes",
            "判分并分析答题结果，形成薄弱点与错因列表（需要学习者的答案）",
            AnalyzeMistakesArgs,
            analyze_mistakes,
        ),
        "answer_question": Tool(
            "answer_question",
            "基于教材检索结果生成针对性讲解（引用来源）",
            AnswerQuestionArgs,
            answer_question,
        ),
    }
