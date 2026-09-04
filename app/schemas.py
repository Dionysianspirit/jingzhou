from pydantic import BaseModel, Field


class IndexRequest(BaseModel):
    doc_id: str = Field(min_length=1, max_length=64, pattern=r"^[a-zA-Z0-9_-]+$")
    name: str = Field(min_length=1, max_length=256)
    text: str = Field(min_length=1, max_length=10_000_000)


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4096)
    doc_ids: list[str] = Field(min_length=1, max_length=50)
    system: str = Field(default="", max_length=4096)


class RemoveRequest(BaseModel):
    doc_id: str


class FeatureRequest(BaseModel):
    doc_ids: list[str] = Field(min_length=1, max_length=50)
    query: str = Field(default="", max_length=4096)
    count: int = Field(default=10, ge=1, le=30)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4096)
    doc_ids: list[str] = Field(min_length=1, max_length=50)
    k: int = Field(default=8, ge=1, le=20)
