"""Single controlled learning agent.

Loop: observe memory state -> LLM decides one action -> execute tool ->
record -> repeat until finish / quiz delivery / budget exhausted.
Two transitions are pinned deterministically (never left to the LLM):
quiz delivery pauses the session for learner answers, and submitted
answers are scored by the analyze_mistakes tool before the LLM resumes.

Guarantees: step budget, tool whitelist + pydantic arg validation,
duplicate-call interception, circuit breaker on repeated failures,
honest termination when no material is available.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

from app import llm
from app.config import AGENT_MAX_STEPS
from app.memory import AgentMemory, AgentStore
from app.tools import ToolContext, ToolResult, build_tools

DECIDE_SYSTEM = (
    "你是「径舟」的自主学习调度器。学习者给出学习目标，你通过调用工具规划并执行多步骤学习任务，"
    "典型流程：检索教材 → 提炼要点 → 出诊断题（交给学习者作答）→ 判分定位薄弱点 → 针对性讲解 → 总结；"
    "判分后若仍有薄弱点且剩余步数充足，可只针对薄弱主题再出一轮小测并再判分，如此多轮直到步数用尽"
    "或最近一轮全对，最后讲解剩余薄弱点并总结。\n"
    "规则：\n"
    "1. 每次只决定下一个动作，返回严格 JSON，不要输出任何其他内容。\n"
    '2. action 只能是工具列表中的工具名，或 "finish"。\n'
    "3. args 必须符合各工具参数要求；检索无命中或相关性低时不要编造，选择换关键词再检索一次或直接 finish。\n"
    "4. 出题可多轮：第一轮全面诊断，其后每轮只针对尚未掌握的薄弱主题；最近一轮全对时不要再出题，转为讲解与总结。\n"
    "5. reason 用一句简短中文说明为什么选这一步（面向用户，不暴露内部推理）；stage 用不超过 6 个字概括当前阶段。\n"
    '返回格式：{"stage":"...","reason":"...","action":"工具名或finish","args":{...}}'
)

SUMMARY_SYSTEM = (
    "你是学习助手。根据一次自主学习过程的记录，用中文 Markdown 生成最终复习总结（400 字内）："
    "目标回顾、掌握情况、薄弱点与错因、针对性建议、复习出处（引用文档名与片段号）。"
    "只依据记录中的信息，不要编造。"
)


class AgentError(Exception):
    pass


def sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _state_view(mem: AgentMemory) -> dict:
    return {
        "stage": mem.stage,
        "step": mem.step_count,
        "findings": mem.findings[:6],
        "weaknesses": [w.topic for w in mem.weaknesses[:6]],
        "status": mem.status,
    }


def _event_data(tool: str, result: ToolResult):
    """Trim tool payloads to what the UI genuinely renders."""
    data = result.data if isinstance(result.data, dict) else {}
    if tool == "search_knowledge":
        hits = data.get("hits") or []
        return {
            "hits": [
                {
                    "doc_id": h["doc_id"],
                    "doc_name": h["doc_name"],
                    "chunk_idx": h["chunk_idx"],
                    "score": h["score"],
                }
                for h in hits[:8]
            ],
            "empty": data.get("empty", False),
            "low_relevance": data.get("low_relevance", False),
        }
    if tool == "read_source":
        return {
            "doc_id": data.get("doc_id"),
            "doc_name": data.get("doc_name"),
            "chunk_idx": data.get("chunk_idx"),
            "preview": str(data.get("text", ""))[:200],
        }
    if tool == "create_quiz":
        return {"count": len(data.get("questions") or []), "topic": data.get("topic", "")}
    if tool == "create_study_guide":
        return {"title": data.get("title", ""), "key_points": (data.get("key_points") or [])[:8]}
    if tool == "answer_question":
        return data
    return data


def _quiz_rounds_done(mem: AgentMemory) -> int:
    return sum(1 for s in mem.steps if s.tool == "create_quiz" and s.ok)


def _decision_view(decision: dict) -> dict:
    return {
        "stage": str(decision.get("stage") or "")[:12],
        "reason": str(decision.get("reason") or "")[:160],
        "action": str(decision.get("action") or "").strip(),
        "args": decision.get("args") if isinstance(decision.get("args"), dict) else {},
    }


def _state_prompt(mem: AgentMemory, tools: dict, doc_brief: str, max_steps: int) -> str:
    lines = [f"【学习目标】{mem.goal}", f"【资料范围】{doc_brief}", "【工具列表】"]
    lines += [t.spec() for t in tools.values()]
    if mem.steps:
        lines.append("【已完成步骤】")
        lines += [
            f"{s.step}. {s.tool}({json.dumps(s.args, ensure_ascii=False)}) → "
            f"{'成功' if s.ok else '失败'}：{s.result_summary}"
            for s in mem.steps[-8:]
        ]
    if mem.findings:
        lines.append("【已提炼要点】\n" + "\n".join(f"- {f}" for f in mem.findings[:8]))
    if mem.weaknesses:
        lines.append(
            "【薄弱点】\n" + "\n".join(f"- {w.topic}（{w.error_type}）" for w in mem.weaknesses[:8])
        )
    quiz_rounds = _quiz_rounds_done(mem)
    if quiz_rounds:
        recent = (
            "最近一轮全对" if mem.last_quiz_wrong == 0 else f"最近一轮错 {mem.last_quiz_wrong} 题"
        )
        quiz_state = f"已出题 {quiz_rounds} 轮，累计薄弱点 {len(mem.weaknesses)} 处，{recent}"
    else:
        quiz_state = "尚未出题"
    lines.append(f"【测验状态】{quiz_state}")
    lines.append(f"【剩余步数】{max_steps - mem.step_count}")
    lines.append("请决定下一步，只输出 JSON。")
    return "\n".join(lines)


async def decide(mem: AgentMemory, tools: dict, doc_brief: str, max_steps: int) -> dict:
    """One LLM decision. Invalid JSON raises AgentError after one retry;
    unknown tool names are returned as-is and rejected by the executor."""
    prompt = _state_prompt(mem, tools, doc_brief, max_steps)
    last_err: Exception | None = None
    for _ in range(2):
        try:
            raw = await llm.complete_json(DECIDE_SYSTEM, prompt)
        except Exception as e:  # JSON parse / network failure
            last_err = e
            continue
        if isinstance(raw, dict) and raw.get("action"):
            return _decision_view(raw)
        last_err = AgentError(f"决策输出缺少 action 字段: {str(raw)[:120]}")
    raise AgentError(f"决策失败: {llm.safe_error(last_err or RuntimeError('未知错误'))}")


async def _execute(tool, raw_args: dict, ctx: ToolContext) -> ToolResult:
    try:
        args = tool.args_model.model_validate(raw_args)
    except Exception as e:
        return ToolResult(ok=False, summary=f"参数校验失败（{tool.name}）：{e}")
    try:
        return await tool.func(args, ctx)
    except Exception as e:
        return ToolResult(ok=False, summary=f"工具执行失败（{tool.name}）：{llm.safe_error(e)}")


def _fail_circuit_open(mem: AgentMemory) -> str | None:
    """Stop when the last 2 steps are failures of the same tool, or last 3 all failed."""
    tail = mem.steps[-3:]
    if len(tail) >= 3 and not any(s.ok for s in tail):
        return "连续工具失败"
    if len(tail) >= 2 and not any(s.ok for s in tail[-2:]) and tail[-1].tool == tail[-2].tool:
        return f"工具 {tail[-1].tool} 连续失败"
    return None


async def _rejection_events(
    decision: dict,
    mem: AgentMemory,
    summary: str,
    persist: AgentStore | None,
    *,
    sse_args: dict | None = None,
) -> AsyncIterator[str]:
    """Record a rejected action (forbidden / unknown / over cap) and emit its events."""
    mem.record_step(decision["action"], decision["args"], decision["reason"], False, summary)
    if persist:
        persist.save(mem)
    yield sse(
        {
            "type": "tool_call",
            "step": mem.step_count,
            "tool": decision["action"],
            "args": sse_args if sse_args is not None else decision["args"],
            "reason": decision["reason"],
        }
    )
    yield sse(
        {
            "type": "tool_result",
            "step": mem.step_count,
            "tool": decision["action"],
            "ok": False,
            "summary": summary,
        }
    )
    yield sse({"type": "state_update", **_state_view(mem)})


def _fallback_summary(mem: AgentMemory) -> str:
    lines = [f"## 复习总结（{mem.goal}）"]
    if mem.findings:
        lines.append("**已覆盖要点**：" + "；".join(mem.findings[:6]))
    if mem.weaknesses:
        lines.append("**薄弱点**：")
        lines += [f"- {w.topic}（{w.error_type}）：{w.reason}" for w in mem.weaknesses[:8]]
    if mem.quiz:
        analyzed = any(s.tool == "analyze_mistakes" and s.ok for s in mem.steps)
        if analyzed:
            lines.append("**测验已判分**，请按薄弱点回查教材对应章节。")
        else:
            lines.append(f"本轮共出 {len(mem.quiz)} 道诊断题。")
    if mem.sources:
        names = {s["name"] for s in mem.sources}
        lines.append("**出处**：" + "、".join(sorted(names)))
    lines.append(f"（停止原因：{mem.stop_reason or 'completed'}）")
    return "\n".join(lines)


async def summarize(mem: AgentMemory) -> str:
    record = {
        "goal": mem.goal,
        "findings": mem.findings[:12],
        "weaknesses": [w.model_dump() for w in mem.weaknesses[:10]],
        "steps": [{"tool": s.tool, "ok": s.ok, "summary": s.result_summary} for s in mem.steps],
        "quiz_questions": len(mem.quiz or []),
        "sources": mem.sources[:10],
    }
    try:
        text = await llm.complete(
            SUMMARY_SYSTEM,
            json.dumps(record, ensure_ascii=False),
            temperature=0.3,
            max_tokens=1024,
        )
        return text.strip() or _fallback_summary(mem)
    except Exception:
        return _fallback_summary(mem)


async def run_session(
    mem: AgentMemory,
    store,
    *,
    persist: AgentStore | None = None,
    max_steps: int | None = None,
) -> AsyncIterator[str]:
    """Run the agent loop, yielding SSE lines until the session pauses or finishes."""
    tools = build_tools(store)
    ctx = ToolContext(store, mem)
    budget = max_steps if max_steps is not None else AGENT_MAX_STEPS
    doc_brief = "、".join(mem.doc_ids) or "（无）"

    if persist:
        persist.save(mem)

    # Pre-flight: an honest stop before any LLM call when nothing is indexed.
    if mem.step_count == 0 and mem.status == "running":
        try:
            known = {d["id"] for d in await store.list_docs()}
        except Exception:
            known = set(mem.doc_ids)
        if not known.intersection(mem.doc_ids):
            mem.status = "finished"
            mem.stop_reason = "no_material"
            mem.final_summary = (
                "所选资料不在书斋中（可能尚未索引或已删除），本次学习任务无法开始。"
                "请先投卷入舱再发起自主学习。"
            )
            if persist:
                persist.save(mem)
            yield sse(
                {"type": "planning", "goal": mem.goal, "doc_ids": mem.doc_ids, "stage": "分析目标"}
            )
            yield sse(
                {
                    "type": "final",
                    "session_id": mem.session_id,
                    "summary": mem.final_summary,
                    "stop_reason": mem.stop_reason,
                    "weaknesses": [],
                    "sources": [],
                    "steps": [],
                }
            )
            yield "data: [DONE]\n\n"
            return
        yield sse(
            {"type": "planning", "goal": mem.goal, "doc_ids": mem.doc_ids, "stage": "分析目标"}
        )

    # Deterministic phase-B entry: score submitted answers before the LLM resumes.
    if mem.status == "running" and mem.pending_answers is not None:
        decision = {
            "stage": "判分",
            "reason": "收到答题结果，先判分并定位薄弱点。",
            "action": "analyze_mistakes",
            "args": {"answers": mem.pending_answers},
        }
        async for event in _run_action(decision, mem, ctx, tools, persist):
            yield event
        # success clears it inside the tool; on failure drop the stale payload
        mem.pending_answers = None
        if mem.status != "running":
            return

    executed_ok: set[tuple[str, str]] = {
        (s.tool, json.dumps(s.args, sort_keys=True, ensure_ascii=False)) for s in mem.steps if s.ok
    }
    stop_reason = "completed"

    while mem.status == "running" and mem.step_count < budget:
        try:
            decision = await decide(mem, tools, doc_brief, budget)
        except AgentError as e:
            mem.status = "finished"
            mem.stop_reason = "error"
            mem.final_summary = f"学习任务中止：{e}"
            if persist:
                persist.save(mem)
            yield sse({"type": "error", "message": str(e)})
            break
        if decision["stage"]:
            mem.stage = decision["stage"]

        if decision["action"] == "finish":
            break

        if decision["action"] not in tools:
            async for ev in _rejection_events(
                decision,
                mem,
                f"调用了不存在的工具「{decision['action']}」，已拒绝。",
                persist,
            ):
                yield ev
            continue

        if decision["action"] == "analyze_mistakes":
            # genuine answers only arrive via /answers; the LLM must not
            # score fabricated ones
            async for ev in _rejection_events(
                decision,
                mem,
                "已拒绝：判分由学习者提交答案自动触发，不能手动调用。",
                persist,
                sse_args={},
            ):
                yield ev
            continue

        if decision["action"] == "create_quiz" and mem.last_quiz_wrong == 0:
            # spiral ends when the latest round was clean: no new weaknesses
            async for ev in _rejection_events(
                decision,
                mem,
                "已拒绝：最近一轮诊断全部正确，未新增薄弱点，无需再测；请转为讲解或总结。",
                persist,
            ):
                yield ev
            continue

        tool = tools[decision["action"]]
        call_key = (tool.name, json.dumps(decision["args"], sort_keys=True, ensure_ascii=False))
        if call_key in executed_ok:
            async for ev in _rejection_events(
                decision,
                mem,
                "重复调用被拦截：同样的工具与参数已成功执行过，请更换查询或换一个工具。",
                persist,
            ):
                yield ev
            continue

        async for event in _run_action(decision, mem, ctx, tools, persist):
            yield event
        if mem.steps and mem.steps[-1].ok:
            executed_ok.add(call_key)

        if mem.status == "awaiting_answers":
            return

        breaker = _fail_circuit_open(mem)
        if breaker:
            stop_reason = f"tool_failures（{breaker}）"
            break

    else:
        if mem.status == "running":
            stop_reason = "max_steps"

    if mem.status != "finished":
        mem.status = "finished"
        mem.stop_reason = stop_reason
        if stop_reason == "completed":
            mem.stage = "总结"
    if stop_reason != "error" or not mem.final_summary:
        mem.final_summary = await summarize(mem)
    if persist:
        persist.save(mem)
    yield sse(
        {
            "type": "final",
            "session_id": mem.session_id,
            "summary": mem.final_summary,
            "stop_reason": mem.stop_reason,
            "weaknesses": [w.model_dump() for w in mem.weaknesses],
            "sources": mem.sources,
            "steps": [
                {"step": s.step, "tool": s.tool, "ok": s.ok, "summary": s.result_summary}
                for s in mem.steps
            ],
        }
    )
    yield "data: [DONE]\n\n"


async def _run_action(
    decision: dict,
    mem: AgentMemory,
    ctx: ToolContext,
    tools: dict,
    persist: AgentStore | None,
) -> AsyncIterator[str]:
    """Execute one whitelisted action, emit tool_call/tool_result/state_update."""
    tool_name = decision["action"]
    tool = tools[tool_name]
    yield sse(
        {
            "type": "tool_call",
            "step": mem.step_count + 1,
            "tool": tool_name,
            "args": decision["args"],
            "reason": decision["reason"],
        }
    )
    result = await _execute(tool, decision["args"], ctx)
    step = mem.record_step(
        tool_name, decision["args"], decision["reason"], result.ok, result.summary
    )
    if persist:
        persist.save(mem)

    if tool_name == "create_quiz" and result.ok:
        mem.status = "awaiting_answers"
        mem.stage = "待作答"
        if persist:
            persist.save(mem)
        yield sse(
            {
                "type": "tool_result",
                "step": step.step,
                "tool": tool_name,
                "ok": True,
                "summary": result.summary,
                "data": _event_data(tool_name, result),
            }
        )
        yield sse(
            {
                "type": "awaiting_input",
                "session_id": mem.session_id,
                "input_type": "quiz_answers",
                "quiz": mem.public_quiz(),
            }
        )
        yield sse({"type": "state_update", **_state_view(mem)})
        return

    yield sse(
        {
            "type": "tool_result",
            "step": step.step,
            "tool": tool_name,
            "ok": result.ok,
            "summary": result.summary,
            "data": _event_data(tool_name, result),
        }
    )
    yield sse({"type": "state_update", **_state_view(mem)})
