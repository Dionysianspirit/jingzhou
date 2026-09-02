import os
import json
import asyncio
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from dotenv import load_dotenv

# HuggingFace 国内镜像
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import numpy as np
from openai import AsyncOpenAI, AuthenticationError as OpenAIAuthError
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

load_dotenv()

# ── Config ──────────────────────────────────────────────
LLM_KEY = os.environ.get("LLM_API_KEY")
LLM_URL = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o")
EMBED_MODEL_NAME = os.environ.get("EMBED_MODEL", "BAAI/bge-small-zh-v1.5")

CHUNK_CHARS = 2048
OVERLAP_CHARS = 512
TOP_K = 8


# ── Lifespan ────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Validate config
    if not LLM_KEY:
        raise RuntimeError(
            "LLM_API_KEY 未设置。请编辑 .env 文件，填入你的 API Key。"
        )

    # Lazy-load embedding model
    from sentence_transformers import SentenceTransformer

    global embed_model, embed_pool
    try:
        embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    except Exception as e:
        raise RuntimeError(
            f"嵌入模型加载失败: {e}\n"
            "请检查网络连接，或设置环境变量 HF_TOKEN 以访问 HuggingFace Hub。"
        )
    embed_pool = ThreadPoolExecutor(max_workers=2)

    # LLM client
    global llm
    llm = AsyncOpenAI(api_key=LLM_KEY, base_url=LLM_URL)

    yield

    # Graceful shutdown
    embed_pool.shutdown(wait=True)
    await llm.close()


app = FastAPI(title="径舟", lifespan=lifespan)

# CORS：允许所有来源访问 API（开发环境安全，生产环境可收窄）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── DocStore ────────────────────────────────────────────
class DocStore:
    def __init__(self):
        self.docs: dict[str, dict] = {}
        self.matrices: dict[str, np.ndarray] = {}
        self.lock = asyncio.Lock()

    def chunk_text(self, text: str) -> list[str]:
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if not paragraphs:
            paragraphs = [text.strip()]

        chunks = []
        current = ""
        for para in paragraphs:
            if len(current) + len(para) <= CHUNK_CHARS:
                current += ("\n\n" + para) if current else para
            else:
                if current:
                    chunks.append(current)
                current = para
                while len(current) > CHUNK_CHARS:
                    split_at = current.rfind("。", 0, CHUNK_CHARS)  # 。
                    if split_at == -1:
                        split_at = current.rfind(".", 0, CHUNK_CHARS)
                    if split_at == -1:
                        split_at = CHUNK_CHARS
                    else:
                        split_at += 1
                    chunks.append(current[:split_at])
                    current = current[max(split_at - OVERLAP_CHARS, 0) :]

        if current:
            chunks.append(current)

        merged = []
        for c in chunks:
            if merged and len(c) < 200:
                merged[-1] += "\n\n" + c
            else:
                merged.append(c)
        return merged

    def _embed(self, texts: list[str]) -> np.ndarray:
        # Returns L2-normalized embeddings → cosine = dot product
        return embed_model.encode(texts, normalize_embeddings=True)

    async def add(self, doc_id: str, name: str, text: str):
        chunks = self.chunk_text(text)
        if not chunks:
            raise ValueError("No extractable text")

        loop = asyncio.get_event_loop()
        matrix = await loop.run_in_executor(embed_pool, self._embed, chunks)

        async with self.lock:
            self.docs[doc_id] = {"name": name, "chunks": chunks, "raw": text}
            self.matrices[doc_id] = matrix

    async def remove(self, doc_id: str):
        async with self.lock:
            self.docs.pop(doc_id, None)
            self.matrices.pop(doc_id, None)

    async def search(self, query: str, doc_ids: list[str], k: int = TOP_K) -> list[dict]:
        loop = asyncio.get_event_loop()
        q_vec = await loop.run_in_executor(embed_pool, self._embed, [query])
        q_vec = q_vec[0]

        results = []
        async with self.lock:
            for did in doc_ids:
                if did not in self.matrices or did not in self.docs:
                    continue
                matrix = self.matrices[did]
                doc = self.docs[did]
                scores = np.dot(matrix, q_vec)
                for i, score in enumerate(scores):
                    results.append({
                        "doc_id": did,
                        "doc_name": doc["name"],
                        "chunk_idx": i,
                        "text": doc["chunks"][i],
                        "score": float(score),
                    })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:k]

    async def list_docs(self) -> list[dict]:
        async with self.lock:
            return [
                {"id": did, "name": d["name"], "chunks": len(d["chunks"])}
                for did, d in self.docs.items()
            ]


store = DocStore()


# ── Models ──────────────────────────────────────────────
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


# ── Helpers ─────────────────────────────────────────────
def _build_ctx(results: list[dict]) -> str:
    parts = []
    for r in results:
        parts.append(f"[{r['doc_name']} 片段{r['chunk_idx']}]\n{r['text']}")
    return "\n\n---\n\n".join(parts)


def _parse_json(content: str) -> dict:
    text = content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        if text.endswith("```"):
            text = text[:-3]
    return json.loads(text)
def _safe_error(e: Exception) -> str:
    msg = str(e)
    if isinstance(e, OpenAIAuthError):
        return "API Key 无效，请检查 .env 中的 LLM_API_KEY"
    if "sk-" in msg:
        return "API 调用失败，请检查 API Key 和网络连接"
    return msg


# ── Routes ──────────────────────────────────────────────
@app.post("/api/index")
async def index_doc(req: IndexRequest):
    if not req.text.strip():
        raise HTTPException(400, "Empty text")
    try:
        await store.add(req.doc_id, req.name, req.text)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"索引失败: {_safe_error(e)}")
    return {"status": "ok", "doc_id": req.doc_id}


@app.post("/api/remove")
async def remove_doc(req: RemoveRequest):
    try:
        await store.remove(req.doc_id)
    except Exception as e:
        raise HTTPException(500, f"删除失败: {_safe_error(e)}")
    return {"status": "ok"}


@app.get("/api/docs")
async def list_docs():
    return await store.list_docs()


@app.post("/api/chat")
async def chat(req: ChatRequest):
    if not req.query.strip():
        raise HTTPException(400, "Empty query")

    try:
        results = await store.search(req.query, req.doc_ids)
    except Exception as e:
        raise HTTPException(500, f"检索失败: {_safe_error(e)}")

    async def generate():
        ctx_parts = []
        for r in results:
            ctx_parts.append(
                f"[来源: {r['doc_name']} 片段{r['chunk_idx']}]\n{r['text']}"
            )
        context = "\n\n---\n\n".join(ctx_parts)

        # 角色设定：优先用前端传来的 system prompt，否则用默认
        persona = req.system.strip() if req.system.strip() else (
            "你是径舟书斋的伴读书童「小舟」，性情温润，言辞典雅，善用中国古典文风。"
            "你自称「小舟」，对用户以「公子/姑娘」相称。"
        )

        system = (
            f"{persona}\n\n"
            "【任务规则】\n"
            "1. 仅基于提供的文档片段回答，不要使用外部知识。\n"
            '2. 如果文档中没有相关信息，诚实地说"遍览全卷，未有所获"。\n'
            "3. 回答时引用来源，格式为 [来源: 文档名 片段N]。\n"
            "4. 在回答后，主动提供「相关知识点回顾」和「下一步研读建议」。\n"
            "5. 优先提取文档中的核心概念、公式和推导过程。"
        )

        try:
            stream = await llm.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": f"文档内容：\n\n{context}\n\n用户问题：{req.query}"},
                ],
                stream=True,
                temperature=0.3,
                max_tokens=2048,
            )

            async for chunk in stream:
                choices = chunk.choices
                if not choices:
                    continue
                delta = choices[0].delta
                if delta.content:
                    yield f"data: {json.dumps({'text': delta.content})}\n\n"

            sources = [
                {"name": r["doc_name"], "chunk": r["chunk_idx"],
                 "score": round(r["score"], 3), "preview": r["text"][:200]}
                for r in results
            ]
            yield f"data: {json.dumps({'sources': sources})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'error': _safe_error(e)})}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


# ── Flashcards ──────────────────────────────────────────
@app.post("/api/flashcards")
async def flashcards(req: FeatureRequest):
    results = await store.search(req.query or "关键概念 核心知识点 定义 公式", req.doc_ids, k=15)
    ctx = _build_ctx(results)

    prompt = (
        "你是一个学习助手。基于以下文档内容，生成学习闪卡。\n"
        "每张闪卡包含一个问题和答案。问题应覆盖文档中的关键概念、定义、公式和重要事实。\n"
        "答案应简洁准确，适合背诵。\n\n"
        f"文档内容：\n\n{ctx}\n\n"
        f"生成 {req.count} 张闪卡。返回JSON："
        '{"cards":[{"q":"问题","a":"答案"}]}'
    )

    try:
        resp = await llm.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=2048,
        )
        return _parse_json(resp.choices[0].message.content)
    except json.JSONDecodeError:
        raise HTTPException(500, "闪卡生成失败：模型返回格式异常，请重试")
    except Exception as e:
        raise HTTPException(500, f"闪卡生成失败: {_safe_error(e)}")


# ── Quiz ───────────────────────────────────────────────
@app.post("/api/quiz")
async def quiz(req: FeatureRequest):
    results = await store.search(req.query or "重要概念 原理 知识点", req.doc_ids, k=12)
    ctx = _build_ctx(results)

    prompt = (
        "你是一个教育评估专家。基于以下文档内容，生成选择题测验。\n"
        "每题包含：题目、4个选项(A/B/C/D)、正确答案索引(0-3)、解释、错误选项分析。\n"
        "题目应有区分度，覆盖不同难度和知识点。\n\n"
        "错误选项分析(wrong_analysis)要求：为每个错误选项分析学生选错的原因，\n"
        "归为四类之一：概念混淆（将不同概念张冠李戴）、公式遗忘（忘记或记错公式）、\n"
        "审题偏移（忽略了题目关键条件或问法）、推导断裂（推理链某步出错）。\n"
        "每个错误选项提供一个type（四选一）和一段简洁reason。\n\n"
        f"文档内容：\n\n{ctx}\n\n"
        f"生成 {req.count} 题。返回JSON：\n"
        '{"questions":[{"q":"题目","options":["A","B","C","D"],"answer":0,'
        '"explanation":"正确解释",'
        '"wrong_analysis":{"1":{"type":"概念混淆","reason":"选B者可能将X与Y混淆"},'
        '"2":{"type":"公式遗忘","reason":"选C者可能遗漏了Z公式"},'
        '"3":{"type":"推导断裂","reason":"选D者在第三步推导出错"}}]}\n'
        "wrong_analysis的key是错误选项的索引(0-3中非正确答案的索引)。"
    )

    try:
        resp = await llm.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,
            max_tokens=3072,
        )
        return _parse_json(resp.choices[0].message.content)
    except json.JSONDecodeError:
        raise HTTPException(500, "测验生成失败：模型返回格式异常，请重试")
    except Exception as e:
        raise HTTPException(500, f"测验生成失败: {_safe_error(e)}")


# ── Mindmap ────────────────────────────────────────────
@app.post("/api/mindmap")
async def mindmap(req: FeatureRequest):
    results = await store.search(req.query or "核心概念 知识体系 结构", req.doc_ids, k=10)
    ctx = _build_ctx(results)

    prompt = (
        "你是一个知识管理专家。基于以下文档内容，生成一个思维导图。\n"
        "使用 Markdown 标题层级表示知识结构：# 主题, ## 子主题, ### 细节。\n"
        "遵循以下规则：\n"
        "1. 根节点（# ）概括文档主题\n"
        "2. 二级标题（##）列出主要知识模块\n"
        "3. 三级标题（###）列出具体概念和细节\n"
        "4. 控制在15-25个节点，简洁清晰\n"
        "5. 只输出 Markdown，不要其他文字\n\n"
        f"文档内容：\n\n{ctx}"
    )

    try:
        resp = await llm.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1024,
        )
        return {"markdown": resp.choices[0].message.content.strip()}
    except Exception as e:
        raise HTTPException(500, f"导图生成失败: {_safe_error(e)}")


# ── Report ─────────────────────────────────────────────
@app.post("/api/report")
async def report(req: FeatureRequest):
    results = await store.search(req.query or "概述 总结 结论 要点", req.doc_ids, k=12)
    ctx = _build_ctx(results)

    system = (
        "你是一个学术写作助手。基于提供的文档内容，生成一份结构清晰的学习报告。\n"
        "报告应包含：摘要、核心概念、关键发现、总结。\n"
        "使用 Markdown 格式，适当使用标题、列表和引用标注。"
    )

    async def generate():
        try:
            stream = await llm.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": f"文档内容：\n\n{ctx}\n\n请基于以上文档生成学习报告。"},
                ],
                stream=True,
                temperature=0.3,
                max_tokens=2048,
            )
            async for chunk in stream:
                choices = chunk.choices
                if not choices:
                    continue
                delta = choices[0].delta
                if delta.content:
                    yield f"data: {json.dumps({'text': delta.content})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': _safe_error(e)})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


# ── Infographic ────────────────────────────────────────
@app.post("/api/infographic")
async def infographic(req: FeatureRequest):
    results = await store.search(req.query or "关键数据 对比 趋势 结论 要点", req.doc_ids, k=12)
    ctx = _build_ctx(results)

    prompt = (
        "你是一个信息可视化专家。基于以下文档内容，提取可用于信息图展示的结构化数据。\n"
        "返回JSON，包含：\n"
        '1. title: 信息图标题\n'
        '2. stats: 3-5个关键数据点，每项包含 {label, value, desc}\n'
        '3. comparisons: 1-3组对比，每组 {left, right, aspect}\n'
        '4. conceptFlow: 3-5个步骤的概念流程，每项 {step, description}\n'
        '5. keyInsight: 一句话核心洞察\n'
        "数据应精确、可追溯至原文。只返回JSON，不要其他文字。\n\n"
        f"文档内容：\n\n{ctx}"
    )

    try:
        resp = await llm.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1536,
        )
        return _parse_json(resp.choices[0].message.content)
    except json.JSONDecodeError:
        raise HTTPException(500, "信息图生成失败：模型返回格式异常，请重试")
    except Exception as e:
        raise HTTPException(500, f"信息图生成失败: {_safe_error(e)}")


# ── Table ──────────────────────────────────────────────
@app.post("/api/table")
async def table(req: FeatureRequest):
    results = await store.search(req.query or "数据 对比 指标 参数 统计", req.doc_ids, k=12)
    ctx = _build_ctx(results)

    prompt = (
        "你是一个数据分析助手。基于以下文档内容，提取结构化数据生成 Markdown 表格。\n"
        "优先提取：对比数据、性能指标、参数列表、分类信息。\n"
        "表格应有清晰的列标题和数据行。如果没有适合表格的数据，返回空表格。\n"
        "只输出 Markdown 表格，不要其他文字。\n\n"
        f"文档内容：\n\n{ctx}\n\n"
        f"主题: {req.query or '关键数据'}"
    )

    try:
        resp = await llm.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1024,
        )
        return {"markdown": resp.choices[0].message.content.strip()}
    except Exception as e:
        raise HTTPException(500, f"表格生成失败: {_safe_error(e)}")


# ── Static ──────────────────────────────────────────────
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8777)
