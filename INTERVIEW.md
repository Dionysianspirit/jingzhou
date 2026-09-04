# 径舟 · 面试口述稿

把项目讲成「我解决过的检索问题」，不要讲成「我接了个模型」。

## 30 秒定位

径舟是面向中文教材的 RAG 学习助手。BGE 检索后大模型只根据命中片段回答。对标 IMA / NotebookLM。

## 3 分钟主线

### 1. 切块
先按空行分段；超窗口按句号切并回带 512 字重叠；短于 200 字的尾巴并回上一块。`python -m unittest tests.test_chunking -v`

### 2. 可解释检索
BGE L2 归一化后 `score = chunk · query` 即余弦。返回 score / rank / overlap / why。点笺请求 `/api/docs/{id}/chunks/{i}` 看原文。

### 3. 生成被检索管住
没命中则「遍览全卷，未有所获」。引用 `[来源: 文档名 片段N]`。指南是同一条 RAG 的另一种读出。

## 演示脚本（与 UI 文案一致）

1. 左侧点「载入示例教材《线性代数导引》」
2. 问「秩-零化度定理在说什么？」
3. 点来源笺，看「检索解释：BGE 向量余弦 Top-k」与原文
4. 顶栏点「指南」
5. 重启后示例卷仍在

对应 `static/index.html` 的 `sampleBtn`、来源笺、`data-feature="guide"`。
