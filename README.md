# 径舟

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)

基于语义检索的多文档 AI 学习助手。上传讲义或教材，按语义找到相关片段，再让模型只根据这些片段回答、出题、画脉络、写学习指南。

产品形态参考腾讯 IMA 与 Google NotebookLM：私有资料入库、问答必须溯源、学习材料一键生成。

## 克隆后 5 分钟跑起来

需要：Python 3.10+，以及任意 OpenAI 兼容接口的 Key。

```bash
git clone https://github.com/Dionysianspirit/jingzhou.git
cd jingzhou
cp .env.example .env
chmod +x start.sh && ./start.sh
```

打开 `http://127.0.0.1:8777`。左侧点「载入示例教材《线性代数导引》」，选中该卷后提问「秩-零化度定理在说什么？」答完点来源笺；顶栏「指南」生成导读。

```bash
python -m unittest discover -s tests -v
```

## 能做什么

| 能力 | 说明 |
|---|---|
| 可解释检索 | top-k、余弦、字词重叠、why |
| 可点击溯源 | 笺片跳原文片段 |
| 学习指南 | 导读、要点、易错、思考题 |
| 笺卡 / 考核 / 脉络 / 析报 | 考核含四种错因 |

详见 [INTERVIEW.md](INTERVIEW.md)。旧版在 [`backup/pre-hardening`](https://github.com/Dionysianspirit/jingzhou/tree/backup/pre-hardening)。

## 许可

[MIT License](LICENSE)。
