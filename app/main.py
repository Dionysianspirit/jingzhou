from __future__ import annotations

import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from app import embeddings, llm
from app.config import EMBED_MODEL_NAME, LLM_KEY, LLM_MODEL, ROOT, SAMPLE_DIR, TOP_K
from app.schemas import (
    ChatRequest,
    FeatureRequest,
    IndexRequest,
    RemoveRequest,
    SearchRequest,
)
from app.store import DocStore

store = DocStore()


@asynccontextmanager
async def lifespan(app: FastAPI):
    embeddings.load_model(EMBED_MODEL_NAME)
    llm.init_client()
    store.load_disk()
    yield
    embeddings.shutdown()
    if llm.llm is not None:
        await llm.llm.close()


app = FastAPI(title="径舟", version="0.2.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _require_llm():
    if not llm.available():
        raise HTTPException(503, "未配置 LLM_API_KEY。复制 .env.example 为 .env 后填入密钥。")


@app.get("/api/health")
async def health():
    return {
        "ok": True,
        "embed_ready": embeddings.ready(),
        "llm_ready": llm.available(),
        "docs": len(store.docs),
        "model": LLM_MODEL if LLM_KEY else None,
    }


@app.post("/api/index")
async def index_doc(req: IndexRequest):
    if not req.text.strip():
        raise HTTPException(400, "Empty text")
    try:
        n = await store.add(req.doc_id, req.name, req.text)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    except Exception as e:
        raise HTTPException(500, f"索引失败: {llm.safe_error(e)}") from e
    return {"status": "ok", "doc_id": req.doc_id, "chunks": n}


@app.post("/api/remove")
async def remove_doc(req: RemoveRequest):
    await store.remove(req.doc_id)
    return {"status": "ok"}


@app.get("/api/docs")
async def list_docs():
    return await store.list_docs()


@app.get("/api/docs/{doc_id}/chunks/{chunk_idx}")
async def get_chunk(doc_id: str, chunk_idx: int):
    item = await store.get_chunk(doc_id, chunk_idx)
    if item is None:
        raise HTTPException(404, "片段不存在")
    return item


@app.post("/api/search")
async def search(req: SearchRequest):
    try:
        hits = await store.search(req.query, req.doc_ids, k=req.k)
    except Exception as e:
        raise HTTPException(500, f"检索失败: {llm.safe_error(e)}") from e
    return {
        "query": req.query,
        "k": req.k,
        "hits": hits,
        "explain": {
            "embed_model": EMBED_MODEL_NAME,
            "similarity": "cosine = dot(L2-normalized BGE vectors)",
            "top_k": req.k,
        },
    }


@app.post("/api/chat")
async def chat(req: ChatRequest):
    _require_llm()
    if not req.query.strip():
        raise HTTPException(400, "Empty query")
    try:
        results = await store.search(req.query, req.doc_ids)
    except Exception as e:
        raise HTTPException(500, f"检索失败: {llm.safe_error(e)}") from e

    async def generate():
        context = llm.build_ctx(results)
        persona = req.system.strip() or (
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
            stream = await llm.llm.chat.completions.create(
                model=llm.model_name(),
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": f"文档内容：\n\n{context}\n\n用户问题：{req.query}"},
                ],
                stream=True,
                temperature=0.3,
                max_tokens=2048,
            )
            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta.content:
                    yield f"data: {json.dumps({'text': delta.content}, ensure_ascii=False)}\n\n"
            sources = [
                {
                    "doc_id": r["doc_id"],
                    "name": r["doc_name"],
                    "chunk": r["chunk_idx"],
                    "score": round(r["score"], 3),
                    "preview": r["text"][:200],
                    "why": r.get("why", ""),
                    "overlap": r.get("overlap", []),
                }
                for r in results
            ]
            yield f"data: {json.dumps({'sources': sources}, ensure_ascii=False)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': llm.safe_error(e)}, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


async def _feature_json(prompt: str, temperature: float, max_tokens: int) -> dict:
    _require_llm()
    resp = await llm.llm.chat.completions.create(
        model=llm.model_name(),
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return llm.parse_json(resp.choices[0].message.content)


@app.post("/api/flashcards")
async def flashcards(req: FeatureRequest):
    results = await store.search(req.query or "关键概念 核心知识点 定义 公式", req.doc_ids, k=15)
    ctx = llm.build_ctx(results)
    prompt = (
        "你是一个学习助手。基于以下文档内容，生成学习闪卡。\n"
        "每张闪卡包含一个问题和答案。\n\n"
        f"文档内容：\n\n{ctx}\n\n"
        f"生成 {req.count} 张闪卡。返回JSON："
        '{"cards":[{"q":"问题","a":"答案"}]}'
    )
    try:
        return await _feature_json(prompt, 0.5, 2048)
    except Exception as e:
        raise HTTPException(500, f"闪卡生成失败: {llm.safe_error(e)}") from e


@app.post("/api/quiz")
async def quiz(req: FeatureRequest):
    results = await store.search(req.query or "重要概念 原理 知识点", req.doc_ids, k=12)
    ctx = llm.build_ctx(results)
    prompt = (
        "基于文档生成选择题。错因归四类。\n"
        f"文档内容：\n\n{ctx}\n\n"
        f"生成 {req.count} 题。返回JSON questions。"
    )
    try:
        return await _feature_json(prompt, 0.6, 3072)
    except Exception as e:
        raise HTTPException(500, f"测验生成失败: {llm.safe_error(e)}") from e


@app.post("/api/mindmap")
async def mindmap(req: FeatureRequest):
    _require_llm()
    results = await store.search(req.query or "核心概念 知识体系 结构", req.doc_ids, k=10)
    ctx = llm.build_ctx(results)
    prompt = f"生成 Markdown 思维导图。\n\n{ctx}"
    try:
        resp = await llm.llm.chat.completions.create(
            model=llm.model_name(),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1024,
        )
        return {"markdown": resp.choices[0].message.content.strip()}
    except Exception as e:
        raise HTTPException(500, f"导图生成失败: {llm.safe_error(e)}") from e


@app.post("/api/report")
async def report(req: FeatureRequest):
    _require_llm()
    results = await store.search(req.query or "概述 总结 结论 要点", req.doc_ids, k=12)
    ctx = llm.build_ctx(results)

    async def generate():
        try:
            stream = await llm.llm.chat.completions.create(
                model=llm.model_name(),
                messages=[
                    {"role": "system", "content": "生成 Markdown 学习报告。"},
                    {"role": "user", "content": ctx},
                ],
                stream=True,
                temperature=0.3,
                max_tokens=2048,
            )
            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta.content:
                    yield f"data: {json.dumps({'text': delta.content}, ensure_ascii=False)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': llm.safe_error(e)}, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/api/study-guide")
async def study_guide(req: FeatureRequest):
    _require_llm()
    results = await store.search(
        req.query or "导读 核心概念 知识结构 易错点 思考题",
        req.doc_ids,
        k=14,
    )
    ctx = llm.build_ctx(results)
    prompt = (
        "生成学习指南 JSON：title,overview,key_points,formulas,pitfalls,questions,next_steps\n\n"
        f"{ctx}"
    )
    try:
        data = await _feature_json(prompt, 0.35, 2048)
        data["sources"] = [
            {
                "doc_id": r["doc_id"],
                "name": r["doc_name"],
                "chunk": r["chunk_idx"],
                "score": round(r["score"], 3),
                "preview": r["text"][:180],
                "why": r.get("why", ""),
            }
            for r in results[:6]
        ]
        return data
    except Exception as e:
        raise HTTPException(500, f"学习指南生成失败: {llm.safe_error(e)}") from e


@app.post("/api/infographic")
async def infographic(req: FeatureRequest):
    results = await store.search(req.query or "关键数据 对比 趋势 结论 要点", req.doc_ids, k=12)
    ctx = llm.build_ctx(results)
    try:
        return await _feature_json("提取 JSON stats/comparisons/conceptFlow/keyInsight\n\n" + ctx, 0.3, 2048)
    except Exception as e:
        raise HTTPException(500, f"信息图生成失败: {llm.safe_error(e)}") from e


@app.post("/api/table")
async def table(req: FeatureRequest):
    results = await store.search(req.query or "对比 分类 条目 一览", req.doc_ids, k=12)
    ctx = llm.build_ctx(results)
    try:
        return await _feature_json("提取对比表 JSON title/headers/rows\n\n" + ctx, 0.3, 2048)
    except Exception as e:
        raise HTTPException(500, f"簿册生成失败: {llm.safe_error(e)}") from e


@app.post("/api/sample")
async def load_sample():
    path = SAMPLE_DIR / "线性代数导引.txt"
    if not path.exists():
        raise HTTPException(404, "示例教材缺失")
    text = path.read_text(encoding="utf-8")
    n = await store.add("sample-linalg", "线性代数导引（示例）", text)
    return {"status": "ok", "doc_id": "sample-linalg", "chunks": n}


static_dir = ROOT / "static"
if static_dir.exists():
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")
