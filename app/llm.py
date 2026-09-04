from __future__ import annotations

import json

from openai import AsyncOpenAI, AuthenticationError as OpenAIAuthError

from app.config import LLM_KEY, LLM_MODEL, LLM_URL

llm: AsyncOpenAI | None = None


def init_client() -> AsyncOpenAI | None:
    global llm
    if not LLM_KEY:
        llm = None
        return None
    llm = AsyncOpenAI(api_key=LLM_KEY, base_url=LLM_URL)
    return llm


def available() -> bool:
    return llm is not None


def model_name() -> str:
    return LLM_MODEL


def safe_error(e: Exception) -> str:
    msg = str(e)
    if isinstance(e, OpenAIAuthError):
        return "API Key 无效，请检查 .env 中的 LLM_API_KEY"
    if "sk-" in msg:
        return "API 调用失败，请检查 API Key 和网络连接"
    return msg


def parse_json(content: str) -> dict:
    text = (content or "").strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        if text.endswith("```"):
            text = text[:-3]
    return json.loads(text)


def build_ctx(results: list[dict]) -> str:
    parts = [f"[{r['doc_name']} 片段{r['chunk_idx']}]\n{r['text']}" for r in results]
    return "\n\n---\n\n".join(parts)
