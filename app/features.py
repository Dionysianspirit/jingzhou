"""Shared generation cores for feature endpoints and agent tools.

Extracted verbatim from main.py so both the REST routes and the agent
wrap the same quiz / study-guide logic instead of duplicating prompts.
"""

from __future__ import annotations

from fastapi import HTTPException

from app import llm

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

ERROR_TYPES = ("概念混淆", "公式遗忘", "审题偏移", "推导断裂")


async def feature_json(prompt: str, temperature: float, max_tokens: int) -> dict:
    if not llm.available():
        raise HTTPException(503, "未配置 LLM_API_KEY。复制 .env.example 为 .env 后填入密钥。")
    resp = await llm.llm.chat.completions.create(
        model=llm.model_name(),
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return llm.parse_json(resp.choices[0].message.content)


def normalize_questions(raw: list) -> list:
    out = []
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
            if kind not in ERROR_TYPES:
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


async def gen_quiz(store, doc_ids: list[str], query: str, count: int) -> list[dict]:
    """Diagnostic quiz core shared by /api/quiz and the agent's create_quiz tool."""
    results = await store.search(query or "重要概念 原理 知识点 定理 公式", doc_ids, k=12)
    ctx = llm.build_ctx(results)
    n = min(count, 8)
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
    data = await feature_json(prompt, 0.6, 3072)
    questions = data.get("questions") or data.get("quiz") or []
    return normalize_questions(questions)


async def gen_guide(store, doc_ids: list[str], query: str) -> dict:
    """Study-guide core shared by /api/study-guide and the agent tool."""
    results = await store.search(
        query or "导读 核心概念 知识结构 易错点 思考题",
        doc_ids,
        k=14,
    )
    ctx = llm.build_ctx(results)
    prompt = (
        "生成学习指南 JSON：title,overview,key_points,formulas,pitfalls,questions,next_steps\n\n"
        f"{ctx}"
    )
    data = await feature_json(prompt, 0.35, 2048)
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
