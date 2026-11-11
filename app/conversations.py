"""Multi-turn chat: conversation persistence and history-aware retrieval.

A follow-up like 「那它和齐次方程有什么关系？」 cannot be retrieved on as it
stands: the vector for 「它」 points nowhere. So each follow-up is first
rewritten into a standalone question (shown to the learner, like every other
retrieval step), retrieval runs on that, and only then does the model answer
with the recent turns as conversational context. Facts still have to come
from the chunks retrieved for the current turn.

Storage mirrors the agent sessions: one JSON file per conversation under
data/chat/, written only after a turn has actually been answered.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from pathlib import Path

from pydantic import BaseModel, Field

from app import llm
from app.config import CHAT_DIR
from app.schemas import SESSION_ID_PATTERN

_SAFE_ID = re.compile(SESSION_ID_PATTERN)

# How much of the past travels with each request. Older turns stay on disk
# and in the UI; they just stop being sent to the model.
HISTORY_TURNS = 6
HISTORY_USER_CHARS = 500
HISTORY_ASSISTANT_CHARS = 1200
TITLE_CHARS = 30
MAX_STANDALONE_CHARS = 300

CONDENSE_SYSTEM = (
    "你负责把学习者的追问改写成可以独立用于检索教材的问句。"
    "结合对话历史补全指代与省略（如「它」「这个定理」「再举个例子」），保留原意；"
    "不要回答问题，不要加入历史里没有的信息。"
    "如果追问本身已经完整，原样输出。只输出改写后的问句本身，不要引号和解释。"
)


class Turn(BaseModel):
    role: str  # user | assistant
    content: str
    ts: int = Field(default_factory=lambda: int(time.time()))
    # assistant turns only: what was retrieved, and the query it was retrieved with
    sources: list[dict] = Field(default_factory=list)
    search_query: str = ""


class Conversation(BaseModel):
    conversation_id: str = Field(default_factory=lambda: "c" + uuid.uuid4().hex[:10])
    title: str = ""
    doc_ids: list[str] = Field(default_factory=list)
    turns: list[Turn] = Field(default_factory=list)
    created_at: int = Field(default_factory=lambda: int(time.time()))
    updated_at: int = Field(default_factory=lambda: int(time.time()))

    def add_exchange(
        self,
        question: str,
        answer: str,
        *,
        sources: list[dict],
        search_query: str,
        doc_ids: list[str],
    ) -> None:
        if not self.title:
            self.title = _clip(" ".join(question.split()), TITLE_CHARS)
        self.turns.append(Turn(role="user", content=question))
        self.turns.append(
            Turn(role="assistant", content=answer, sources=sources, search_query=search_query)
        )
        self.doc_ids = list(doc_ids)
        self.updated_at = int(time.time())

    def history_messages(self) -> list[dict]:
        """Recent turns as chat messages, clipped so history cannot crowd out
        the retrieved context of the current turn."""
        messages = []
        for turn in self.turns[-HISTORY_TURNS:]:
            limit = HISTORY_USER_CHARS if turn.role == "user" else HISTORY_ASSISTANT_CHARS
            messages.append({"role": turn.role, "content": _clip(turn.content, limit)})
        return messages

    def last_question(self) -> str:
        for turn in reversed(self.turns):
            if turn.role == "user":
                return turn.content
        return ""


class ConversationStore:
    """File-per-conversation persistence under data/chat/."""

    def __init__(self, root: Path | None = None):
        self.root = Path(root) if root else CHAT_DIR
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, conversation_id: str) -> Path:
        # the id is a file name: never let one carry a path out of data/chat/
        if not isinstance(conversation_id, str) or _SAFE_ID.fullmatch(conversation_id) is None:
            raise ValueError("非法的对话编号")
        return self.root / f"{conversation_id}.json"

    def save(self, conv: Conversation) -> None:
        path = self._path(conv.conversation_id)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(conv.model_dump(), ensure_ascii=False, indent=1), encoding="utf-8"
        )
        tmp.replace(path)

    def load(self, conversation_id: str) -> Conversation | None:
        try:
            path = self._path(conversation_id)
        except ValueError:
            return None
        if not path.exists():
            return None
        try:
            return Conversation.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def delete(self, conversation_id: str) -> bool:
        try:
            path = self._path(conversation_id)
        except ValueError:
            return False
        if not path.exists():
            return False
        path.unlink()
        return True

    def list_conversations(self) -> list[dict]:
        """Summaries, newest first; corrupt files are skipped."""
        items: list[dict] = []
        for path in self.root.glob("*.json"):
            try:
                conv = Conversation.model_validate_json(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not conv.turns:
                continue
            items.append(
                {
                    "conversation_id": conv.conversation_id,
                    "title": conv.title,
                    "doc_ids": conv.doc_ids,
                    "exchanges": sum(1 for t in conv.turns if t.role == "user"),
                    "updated_at": conv.updated_at,
                }
            )
        items.sort(key=lambda x: (x["updated_at"], x["conversation_id"]), reverse=True)
        return items


async def standalone_query(conv: Conversation | None, question: str) -> tuple[str, bool]:
    """The question retrieval should run on, and whether it was rewritten.

    First turns go through untouched. A follow-up is rewritten by the model;
    if that fails, the previous question is prepended instead, which is crude
    but keeps the topic words in the query rather than searching on 「它」.
    """
    question = question.strip()
    if conv is None or not conv.turns:
        return question, False
    lines = []
    for turn in conv.turns[-HISTORY_TURNS:]:
        speaker = "学习者" if turn.role == "user" else "助手"
        lines.append(f"{speaker}：{_clip(turn.content, 300)}")
    prompt = "【对话历史】\n" + "\n".join(lines) + f"\n\n【追问】{question}"
    try:
        raw = await llm.complete(CONDENSE_SYSTEM, prompt, temperature=0.0, max_tokens=160)
    except Exception:
        raw = ""
    rewritten = _clean_standalone(raw)
    if rewritten:
        return rewritten, rewritten != question
    fallback = _clip(f"{conv.last_question()} {question}".strip(), MAX_STANDALONE_CHARS)
    return fallback, fallback != question


def _clean_standalone(raw: str) -> str:
    text = " ".join((raw or "").split()).strip("\"'「」“”『』")
    if not text or len(text) > MAX_STANDALONE_CHARS:
        return ""
    return text


def _clip(text: str, limit: int) -> str:
    text = text or ""
    return text if len(text) <= limit else text[:limit].rstrip() + "…"
