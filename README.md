# 径舟 · JingZhou

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.3.0-009688.svg)](https://fastapi.tiangolo.com)

> 书山有径，学海泛舟。

**径舟** 是一个面向中文教材的多文档 RAG 学习助手：资料投进书斋，系统用 BGE 中文向量做检索，大模型**只根据命中片段**回答，并一键生成闪卡、测验、思维导图、学习指南等学习材料。v0.3 起在之上加了一层**单 Agent 调度**——用户不再逐个点功能，而是给出学习目标（如「帮我复习前三章，重点找出我薄弱的知识」），由 Agent 自主决定检索、出题、判分、讲解与总结的顺序，完成一个多步骤学习闭环。产品形态对标 NotebookLM / 腾讯 IMA——「资料在先，回答必须带出处」。

与黑盒包装的 RAG 应用不同，径舟把检索过程做成透明可验证的：切块、编码、相似度、落盘、引用回查各自成模块，每次命中都带分数与入选原因，回答里的来源笺可以逐条点开原文核对。

## 克隆后 5 分钟跑起来

需要 Python 3.10+，以及任意 OpenAI 兼容接口的 Key（官方 / 中转 / 国产均可）。

```bash
git clone https://github.com/Dionysianspirit/jingzhou.git
cd jingzhou
./start.sh          # Windows 双击 start.bat
```

启动脚本会自动建虚拟环境、装依赖，并在首次运行时引导你填入 API Key（也可手动 `cp .env.example .env` 填写）。服务就绪后打开 <http://127.0.0.1:8777>：

1. 左侧点「载入示例教材《线性代数导引》」，或直接拖入自己的 PDF / TXT / MD
2. 选中卷册，提问，例如「秩-零化度定理在说什么？」
3. 点回答下方的来源笺，核对命中的原文片段与检索分数
4. 顶栏「指南」一键生成导读、要点与思考题；关掉进程再开，卷册仍在（向量已落盘）
5. 顶栏「自主」进入 Agent 模式：输入学习目标（或在输入框写好再点），看着它 检索 → 出诊断题 → 等你作答 → 判分找薄弱点 → 针对性讲解 → 总结

另有一个最小化的 API 走查页在 `/demo.html`，可用来逐接口验证检索与生成链路。

运行测试：

```bash
python -m unittest discover -s tests -v
```

## 功能特性

| 能力 | 说明 |
| --- | --- |
| 🧭 自主学习 Agent | 给一句学习目标，Agent 自主调度检索/出题/判分/讲解/总结，SSE 流式回报每步动作与理由，测验答案在服务端判分 |
| 💬 伴读问答 | 多文档语义检索 + SSE 流式回答，引用格式 `[来源: 文档名 片段N]`，书童「小舟」人设可自定义 |
| 📜 拖拽投卷 | PDF 由浏览器端 pdf.js 解析取字，TXT / MD 直传，全程文本不出本机（仅命中片段送 LLM） |
| 🔍 可解释检索 | 独立 `/api/search` 接口，每个命中块返回余弦分数、排名、字词重叠与一句话入选原因 |
| 📌 点击溯源 | 来源笺点开即原文（`/api/docs/{id}/chunks/{i}`），引用可验证而非模型编造 |
| 💾 向量持久化 | 索引落盘 `data/store/`，重启自动读回，文档不必重传 |
| 📖 学习工具组 | 指南（导读/要点/易错/思考题）、笺卡翻转、考核（识海偏误图 + 研思足迹 + 四类错因）、脉络、析报、簿册、览图 |
| 🛟 无 Key 可跑 | 不配 LLM_API_KEY 时检索类接口照常工作，生成类接口优雅返回 503 |

## 检索管线

```mermaid
flowchart LR
    subgraph 入库
        A[上传文本] --> B[chunking.py<br>段落优先 · 句号硬切<br>2048 字 / 512 重叠 / 短尾合并]
        B --> C[embeddings.py<br>BGE-small-zh · L2 归一化]
        C --> D[store.py<br>内存矩阵 + 落盘<br>meta.json / chunks.json / embeddings.npy]
    end
    subgraph 问答
        E[提问] --> F[问句编码]
        F --> G[score = chunk · query<br>归一化后即余弦]
        G --> H[Top-K + rank / overlap / why]
        H --> I[仅命中块进 LLM 上下文]
        I --> J[SSE 流式回答 + 可回查来源]
    end
```

三个关键设计：

1. **切块不一刀切**：先按空行拼段，超长段在 2048 字内找最近句号切开并回带 512 字重叠，短于 200 字的尾巴并回上一块，避免无语义碎片向量。
2. **余弦即点积**：`normalize_embeddings=True` 使向量落在单位球上，`np.dot` 结果就是余弦相似度，分数跨查询可直接比较。
3. **生成被检索约束**：提示词禁止外部知识，未命中要求坦承「遍览全卷，未有所获」；引用带 `doc_id + chunk_idx`，前端可回查原文。

更多设计细节见 [INTERVIEW.md](INTERVIEW.md)。

## 自主学习 Agent（v0.3）

单 Agent、无新框架：LLM 只做「下一步动作决策」，所有实际能力都是对既有模块的包装。

```mermaid
flowchart TD
    A[学习目标 + 选定卷册] --> B[planning 分析目标]
    B --> C{Agent 决策<br>observe → decide → act}
    C -->|search_knowledge / create_study_guide| D[检索与提炼]
    C -->|create_quiz| E[交付诊断题<br>会话暂停 awaiting_answers]
    E --> F[学习者交卷<br>POST /answers]
    F --> G[analyze_mistakes 判分<br>确定性，不走 LLM]
    G --> C
    C -->|answer_question / read_source| H[针对薄弱点回查与讲解]
    C -->|finish| I[final 复习总结<br>含薄弱点与出处]
```

- **工具白名单**（`app/tools.py`，每个都有 Pydantic 参数校验）：`search_knowledge`（语义检索）、`read_source`（片段回查）、`create_study_guide`（要点提炼）、`create_quiz`（诊断出题）、`analyze_mistakes`（判分与薄弱点归类）、`answer_question`（带出处的针对性讲解）
- **受控保证**：硬性步数预算（默认 8，`AGENT_MAX_STEPS` 可调 6–10）；工具白名单外的调用直接拒绝；相同工具+参数的重复调用拦截；同一工具连续失败 2 次熔断结束；检索无命中/资料不存在时诚实终止而非编造；决策 JSON 解析失败重试一次后报错收场
- **人在环上**：出题后暂停等学习者作答，答案只经 `POST /api/agent/{sid}/answers` 进入，Agent 无法自行编造答题结果（LLM 主动调用 `analyze_mistakes` 会被拒绝）
- **多轮诊断**：判分后若仍有薄弱点且步数充裕，Agent 可只针对薄弱主题再出一轮小测、再判分，如此多轮直到步数预算用尽或最近一轮全对（此时再出题会被确定性拒绝），薄弱点跨轮累计
- **可解释**：SSE 事件流 `planning / tool_call / tool_result / state_update / awaiting_input / final`，每步带一句面向用户的理由，不展示内部 chain-of-thought；测验下发给前端时会剥离正确答案，判分在服务端完成；`answer_question` 讲解中的 `[来源: 文档名 片段N]` 引用会逐条自动经 `read_source` 回读原文核验，并区分三态——有据可查（可点开原文）、可读但不在本次依据中、未能在原文核对——杜绝空口讲解
- **会话持久化**：`data/agent/<session_id>.json` 记录目标、步骤、发现、测验、薄弱点与出处，`GET /api/agent` 列出全部会话，「自主」页可一键恢复待作答会话（重建时间线后继续交卷）或回看已了结的总结
- **限制**：一次会话内的复习闭环（不含跨会话长期记忆与用户画像）；测验再测轮数受步数预算约束（每轮消耗出题与判分两步）

## API 一览

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `GET` | `/api/health` | 健康检查：嵌入模型 / LLM 就绪状态、文档数 |
| `POST` | `/api/index` | 索引文档（doc_id + 纯文本，上限 10MB） |
| `POST` | `/api/remove` | 删除文档（内存与磁盘同步清理） |
| `GET` | `/api/docs` | 已索引文档列表 |
| `GET` | `/api/docs/{doc_id}/chunks/{chunk_idx}` | 取回指定片段原文（溯源回查） |
| `POST` | `/api/search` | 可解释检索，返回 score / rank / overlap / why |
| `POST` | `/api/chat` | RAG 问答（SSE 流式，含来源与解释字段） |
| `POST` | `/api/study-guide` | 生成学习指南（JSON，附来源） |
| `POST` | `/api/flashcards` · `/api/quiz` | 笺卡、考核（考核含错因分类） |
| `POST` | `/api/mindmap` · `/api/report` | 脉络（Markdown）、析报（SSE） |
| `POST` | `/api/infographic` · `/api/table` | 览图数据、簿册（JSON） |
| `GET` | `/api/agent` | 会话摘要列表（待作答可恢复、已了结可回看） |
| `POST` | `/api/agent` | 发起自主学习（SSE：planning/tool_call/tool_result/state_update/awaiting_input/final） |
| `POST` | `/api/agent/{session_id}/answers` | 提交诊断题答案，Agent 续跑判分→讲解→总结（SSE） |
| `GET` | `/api/agent/{session_id}` | 回看会话状态（待作答时不含答案） |
| `POST` | `/api/sample` | 一键载入示例教材 |

## 配置

全部经 `.env` 注入（首次运行启动脚本会引导填写），见 [.env.example](.env.example)：

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `LLM_API_KEY` | 空 | 留空则仅检索可用，生成接口返回 503 |
| `LLM_BASE_URL` | `https://api.openai.com/v1` | 任意 OpenAI 兼容端点 |
| `LLM_MODEL` | `gpt-4o` | 对话模型 |
| `EMBED_MODEL` | `BAAI/bge-small-zh-v1.5` | 本地嵌入模型（默认走 hf-mirror 镜像，首启约百兆下载） |
| `CHUNK_CHARS` / `OVERLAP_CHARS` / `TOP_K` | `2048` / `512` / `8` | 切块与检索参数 |
| `AGENT_MAX_STEPS` | `8` | 自主学习 Agent 每会话最大执行步数（限制在 6–10） |
| `JINGZHOU_DATA` | `./data` | 持久化与示例数据目录（含 Agent 会话 `data/agent/`） |
| `PORT` | `8777` | 服务端口 |

## 项目结构

```
jingzhou/
├── app/                    # 后端包
│   ├── main.py             # FastAPI 路由与生命周期（含 Agent API）
│   ├── agent.py            # 单 Agent 循环：决策/执行/守卫/SSE 事件
│   ├── tools.py            # Agent 工具白名单（包装既有能力 + 参数校验）
│   ├── memory.py           # Agent 会话状态与 JSON 落盘（data/agent/）
│   ├── features.py         # quiz / study-guide 生成内核（路由与工具共用）
│   ├── store.py            # DocStore：检索 / 持久化 / 解释字段
│   ├── chunking.py         # 段落感知切块（纯函数，可单测）
│   ├── embeddings.py       # BGE 模型加载与编码线程池
│   ├── llm.py              # OpenAI 兼容客户端、补全/JSON 帮手、错误脱敏
│   ├── schemas.py          # Pydantic 请求模型
│   └── config.py           # 环境变量与路径
├── server.py               # 兼容入口，等价于 uvicorn app.main:app
├── static/                 # 前端（零构建）
│   ├── index.html          # 书斋主界面
│   ├── studio.css / .js    # 样式与交互逻辑
│   └── demo.html           # API 走查演示页
├── data/
│   ├── sample/             # 内置示例教材《线性代数导引》
│   ├── store/              # 向量落盘目录（已 gitignore）
│   └── agent/              # Agent 会话落盘目录（已 gitignore）
├── tests/                  # unittest：切块 / 检索契约 / Agent 循环
├── start.sh / start.bat    # 一键启动（建 venv、装依赖、引导配置）
└── INTERVIEW.md            # 设计与讲述稿
```

## 回退

v0.1 单文件版本（含当时的方案说明书与演示素材）保留在分支 [`backup/pre-hardening`](https://github.com/Dionysianspirit/jingzhou/tree/backup/pre-hardening)：

```bash
git fetch origin && git checkout backup/pre-hardening
```

## 路线展望

- [x] 向量持久化
- [x] 可解释检索与来源回查
- [x] 自主学习 Agent（单 Agent 多步闭环）
- [x] Agent 会话恢复（待作答续跑 + 总结回看）
- [ ] BM25 混合检索
- [ ] 会话历史与多轮对话
- [ ] 更多文档格式（DOCX / EPUB）
- [ ] 闪卡导出 Anki

## 许可

[MIT License](LICENSE)

---

径舟书斋，伴读不倦。📖
