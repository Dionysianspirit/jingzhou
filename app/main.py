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
    results = await store.search(req.query or "重要概念 原理 知识点 定理 公式", req.doc_ids, k=12)
    ctx = llm.build_ctx(results)
    n = min(req.count, 8)
    prompt = (
        "基于文档生成中文选择题，用于诊断学习偏误。\n"
        "错因只能是这四类之一：概念混淆、公式遗忘、审题偏移、推导断裂。\n"
        "wrong_analysis 必须覆盖每一个错误选项（不是正确答案的选项）。\n\n"
        f"文档内容：\n\n{ctx}\n\n"
        f"生成 {n} 题。严格返回 JSON：\n"
        '{"questions":[{"q":"题干","options":["甲","乙","丙","丁"],'
        '"answer":0,"explanation":"正解说明",'
        '"wrong_analysis":{"1":{"type":"概念混淆","reason":"为何会选这个错项"},'
        '"2":{"type":"公式遗忘","reason":"..."},'
        '"3":{"type":"审题偏移","reason":"..."}}}]}'
    )
    try:
        data = await _feature_json(prompt, 0.6, 3072)
        questions = data.get("questions") or data.get("quiz") or []
        return {"questions": _normalize_questions(questions)}
    except HTTPException as e:
        if e.status_code == 503:
            return {**SAMPLE_QUIZ, "fallback": True, "reason": "未配置 LLM，已载入示例考核"}
        raise
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


def _normalize_questions(raw: list) -> list:
    out = []
    types = {"概念混淆", "公式遗忘", "审题偏移", "推导断裂"}
    for item in raw:
        if not isinstance(item, dict):
            continue
        options = item.get("options") or item.get("choices") or []
        if not isinstance(options, list) or len(options) < 2:
            continue
        try:
            answer = int(item.get("answer", 0))
        except (TypeError, ValueError):
            answer = 0
        answer = max(0, min(answer, len(options) - 1))
        analysis = item.get("wrong_analysis") or item.get("analysis") or {}
        if not isinstance(analysis, dict):
            analysis = {}
        cleaned = {}
        for key, val in analysis.items():
            if not isinstance(val, dict):
                continue
            kind = str(val.get("type") or "概念混淆")
            if kind not in types:
                kind = "概念混淆"
            cleaned[str(key)] = {"type": kind, "reason": str(val.get("reason") or "")}
        out.append(
            {
                "q": str(item.get("q") or item.get("question") or ""),
                "options": [str(x) for x in options[:4]],
                "answer": answer,
                "explanation": str(item.get("explanation") or item.get("explain") or ""),
                "wrong_analysis": cleaned,
            }
        )
    return out


SAMPLE_QUIZ = {
    "questions": [
        {
            "q": "秩-零化度定理在说什么？",
            "options": [
                "rank(T) + nullity(T) = dim(V)",
                "rank(T) = nullity(T)",
                "核空间与像空间维数一定相等",
                "任何线性映射都是单射",
            ],
            "answer": 0,
            "explanation": "秩-零化度定理：线性映射的秩与核空间维数之和等于定义域维数。",
            "wrong_analysis": {
                "1": {"type": "公式遗忘", "reason": "把加法关系记成了相等。"},
                "2": {"type": "概念混淆", "reason": "核与像一般不正交互补到同维。"},
                "3": {"type": "审题偏移", "reason": "定理讨论维数关系，不是单射判定。"},
            },
        },
        {
            "q": "若线性映射把一组基送到线性相关的像，能推出什么？",
            "options": [
                "映射一定是满射",
                "映射一定是零映射",
                "映射不是单射，核空间维数至少为 1",
                "定义域维数必须为 0",
            ],
            "answer": 2,
            "explanation": "基的像线性相关，说明存在非零线性组合被映成零，故核非平凡，映射不是单射。",
            "wrong_analysis": {
                "0": {"type": "审题偏移", "reason": "相关只约束核，推不出满射。"},
                "1": {"type": "概念混淆", "reason": "把「像线性相关」误读成「映射整体为零」。"},
                "3": {"type": "推导断裂", "reason": "从相关跳到维数为零，中间缺了核的判断。"},
            },
        },
        {
            "q": "计算 2×2 矩阵 [[1,2],[2,4]] 的秩时，关键步骤是？",
            "options": [
                "两行相加得到秩为 2",
                "行列式非零所以秩为 2",
                "矩阵有四个元素所以秩为 4",
                "第二行是第一行的 2 倍，行阶梯后只剩一行，秩为 1",
            ],
            "answer": 3,
            "explanation": "两行成比例，行空间一维，秩为 1。",
            "wrong_analysis": {
                "0": {"type": "推导断裂", "reason": "停在「看起来有两行」而没有做行变换。"},
                "1": {"type": "公式遗忘", "reason": "该矩阵行列式为 0，不能用「非零⇒满秩」。"},
                "2": {"type": "概念混淆", "reason": "把元素个数当成了秩。"},
            },
        },
        {
            "q": "下列哪一句更接近「线性相关」的含义？",
            "options": [
                "其中至少有一个向量可以写成其余向量的线性组合",
                "所有向量长度都相等",
                "向量两两垂直",
                "向量都落在不同的坐标轴上",
            ],
            "answer": 0,
            "explanation": "一组向量线性相关，当且仅当存在不全为零的系数使线性组合为零。",
            "wrong_analysis": {
                "1": {"type": "概念混淆", "reason": "把几何外观当成了代数关系。"},
                "2": {"type": "审题偏移", "reason": "正交是内积概念，不是相关。"},
                "3": {"type": "公式遗忘", "reason": "标准基彼此无关，方向不同并不等于相关。"},
            },
        },
    ]
}


@app.get("/api/sample-quiz")
async def sample_quiz():
    return SAMPLE_QUIZ


static_dir = ROOT / "static"
if static_dir.exists():
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")
