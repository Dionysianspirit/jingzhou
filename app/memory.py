"""Minimal agent session state + JSON persistence (data/agent/<session_id>.json).

Scope is deliberately tiny: one review session, its goal, executed steps,
quiz, weaknesses and sources. No cross-session user profiling.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

from pydantic import BaseModel, Field

from app.config import AGENT_DIR


class AgentStep(BaseModel):
    step: int
    tool: str
    args: dict = Field(default_factory=dict)
    reason: str = ""
    ok: bool = True
    result_summary: str = ""


class WeakPoint(BaseModel):
    topic: str
    error_type: str = ""
    reason: str = ""
    question: str = ""
    source: dict | None = None


class AgentMemory(BaseModel):
    session_id: str = Field(default_factory=lambda: "a" + uuid.uuid4().hex[:10])
    goal: str
    doc_ids: list[str] = Field(default_factory=list)
    status: str = "running"  # running | awaiting_answers | finished
    stage: str = "分析目标"
    stop_reason: str = ""  # completed | max_steps | no_material | tool_failures | error
    step_count: int = 0
    steps: list[AgentStep] = Field(default_factory=list)
    plan: list[str] = Field(default_factory=list)
    findings: list[str] = Field(default_factory=list)
    quiz: list[dict] | None = None
    pending_answers: list[int] | None = None
    weaknesses: list[WeakPoint] = Field(default_factory=list)
    sources: list[dict] = Field(default_factory=list)
    final_summary: str = ""
    created_at: int = Field(default_factory=lambda: int(time.time()))
    updated_at: int = Field(default_factory=lambda: int(time.time()))

    def touch(self) -> None:
        self.updated_at = int(time.time())

    def record_step(self, tool: str, args: dict, reason: str, ok: bool, summary: str) -> AgentStep:
        self.step_count += 1
        item = AgentStep(
            step=self.step_count,
            tool=tool,
            args=args,
            reason=reason,
            ok=ok,
            result_summary=summary,
        )
        self.steps.append(item)
        self.touch()
        return item

    def add_sources(self, hits: list[dict]) -> None:
        seen = {(s["doc_id"], s["chunk"]) for s in self.sources}
        for h in hits:
            key = (h["doc_id"], h.get("chunk_idx"))
            if key in seen:
                continue
            seen.add(key)
            self.sources.append(
                {
                    "doc_id": h["doc_id"],
                    "name": h.get("doc_name") or h.get("name") or h["doc_id"],
                    "chunk": h.get("chunk_idx", h.get("chunk")),
                    "score": round(float(h.get("score", 0.0)), 3),
                }
            )

    def public_quiz(self) -> list[dict]:
        """Quiz without answers/wrong_analysis — safe to hand to the learner."""
        if not self.quiz:
            return []
        return [{"q": q.get("q", ""), "options": q.get("options", [])} for q in self.quiz]


class AgentStore:
    """File-per-session persistence under data/agent/."""

    def __init__(self, root: Path | None = None):
        self.root = Path(root) if root else AGENT_DIR
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, session_id: str) -> Path:
        return self.root / f"{session_id}.json"

    def save(self, mem: AgentMemory) -> None:
        mem.touch()
        self._path(mem.session_id).write_text(
            json.dumps(mem.model_dump(), ensure_ascii=False, indent=1),
            encoding="utf-8",
        )

    def load(self, session_id: str) -> AgentMemory | None:
        path = self._path(session_id)
        if not path.exists():
            return None
        try:
            return AgentMemory.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception:
            return None
