from typing import Annotated, Literal

from pydantic import BaseModel, Field

# Ids that become directory or file names on disk. Anything outside this
# alphabet (path separators, dots, absolute paths) is rejected before it can
# reach the filesystem.
DOC_ID_PATTERN = r"^[a-zA-Z0-9_-]+$"
DocId = Annotated[str, Field(min_length=1, max_length=64, pattern=DOC_ID_PATTERN)]
SESSION_ID_PATTERN = r"^[a-zA-Z0-9]{1,32}$"


class IndexRequest(BaseModel):
    doc_id: DocId
    name: str = Field(min_length=1, max_length=256)
    text: str = Field(min_length=1, max_length=10_000_000)


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4096)
    doc_ids: list[str] = Field(min_length=1, max_length=50)
    system: str = Field(default="", max_length=4096)
    # Multi-turn: pass the id of a stored conversation to continue it, or
    # remember=true to start one. With neither, the call is a stateless
    # single question exactly as before.
    conversation_id: str | None = Field(default=None, pattern=SESSION_ID_PATTERN)
    remember: bool = False


class RemoveRequest(BaseModel):
    doc_id: DocId


class FeatureRequest(BaseModel):
    doc_ids: list[str] = Field(min_length=1, max_length=50)
    query: str = Field(default="", max_length=4096)
    count: int = Field(default=10, ge=1, le=30)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4096)
    doc_ids: list[str] = Field(min_length=1, max_length=50)
    k: int = Field(default=8, ge=1, le=20)
    # None = server default (RETRIEVAL_MODE); lets one query be compared across modes
    mode: Literal["hybrid", "dense", "bm25"] | None = None


class AgentStartRequest(BaseModel):
    goal: str = Field(min_length=4, max_length=500)
    doc_ids: list[str] = Field(min_length=1, max_length=50)


class AgentAnswersRequest(BaseModel):
    answers: list[int] = Field(min_length=1, max_length=30)
