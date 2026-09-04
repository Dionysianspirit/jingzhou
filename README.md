# 径舟

基于语义检索的多文档 AI 学习助手。上传讲义或教材，按语义找到相关片段，再让模型只根据这些片段回答、出题、画脉络、写学习指南。

产品形态参考腾讯 IMA 与 Google NotebookLM：私有资料入库、问答必须溯源、学习材料一键生成。实现刻意做成「可以把检索过程讲清楚」的工程，而不是黑盒包装。

## 克隆后 5 分钟跑起来

需要：Python 3.10+，以及任意 OpenAI 兼容接口的 Key（官方 / 中转 / 国产均可）。

```bash
git clone https://github.com/Dionysianspirit/jingzhou.git
cd jingzhou
cp .env.example .env               # 填入 LLM_API_KEY
chmod +x start.sh && ./start.sh    # Windows 用 start.bat
```

浏览器打开 `http://127.0.0.1:8777`，点「载入示例教材」，选中卷册后提问，例如：

> 秩-零化度定理在说什么？和齐次方程有没有非零解有什么关系？

首次启动会从 HuggingFace 镜像下载 `BAAI/bge-small-zh-v1.5`（约百兆），之后本地缓存。关进程再开，文档向量会从 `data/store/` 读回，不必重传。

```bash
python -m unittest discover -s tests -v
```

## 能做什么

| 能力 | 说明 |
|---|---|
| 多文档入库 | PDF / TXT / MD，按段切块后做中文向量 |
| 可解释检索 | 返回 top-k、余弦分数、字词重叠、选中原因 |
| 可点击溯源 | 回答里的笺片跳到原文片段 |
| 伴读问答 | 只根据命中块生成，引用格式 `[来源: 文档 片段N]` |
| 学习指南 | 导读、要点、易错、思考题（NotebookLM Study Guide） |
| 笺卡 / 考核 / 脉络 / 析报 | 原有学习工具，考核含四种错因分类 |

## 架构

```
上传文本
  → 段落下切块（句号硬切 + 重叠 + 短尾合并）
  → BGE-small-zh 编码，L2 归一化
  → 写入内存矩阵，并落盘 data/store/<id>/{meta,chunks,embeddings}
提问
  → 问句同样编码
  → score = dot(chunk, query)   # 因已归一化，即余弦
  → 取 top-k，附 overlap / why
  → 仅把这些块塞进 LLM 上下文
  → 流式回答 + 可点击来源
```

关键模块：

- [`app/chunking.py`](app/chunking.py) 切块，无模型依赖，有单测
- [`app/store.py`](app/store.py) 检索、持久化、解释字段
- [`app/embeddings.py`](app/embeddings.py) BGE 线程池
- [`app/main.py`](app/main.py) FastAPI 路由
- [`static/index.html`](static/index.html) 书斋界面

## 面试时建议讲的三点

1. **为什么这样切块**：段先行，超长段按句号切，重叠防止定义落在边界上，短尾合并避免噪声向量。
2. **为什么余弦等于点积**：`normalize_embeddings=True` 之后向量在单位球上，点积就是余弦；分数可直接比较。
3. **生成如何被检索约束**：系统提示禁止外部知识，引用必须带片段号，前端用 `doc_id + chunk_idx` 回查原文，避免「有引用看起来却点不开」。

更完整的口述稿见 [INTERVIEW.md](INTERVIEW.md)。

## 回退

改造前的完整旧树在分支 [`backup/pre-hardening`](https://github.com/Dionysianspirit/jingzhou/tree/backup/pre-hardening)（含单文件 `server.py` 和当时的 README）。若当前 `main` 不可用：

```bash
git fetch origin
git checkout backup/pre-hardening
```

## 环境变量

见 `.env.example`。`LLM_BASE_URL` 可指向任何 OpenAI 兼容网关。
