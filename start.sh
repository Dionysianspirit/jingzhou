#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

say() { printf '[径舟] %s\n' "$*"; }

if ! command -v python3 >/dev/null 2>&1; then
  say "未找到 python3。请先安装 Python 3.10+"
  exit 1
fi

PY_VER="$(python3 -c 'import sys; print("%d.%d"%sys.version_info[:2])')"
python3 - <<'PY'
import sys
if sys.version_info < (3, 10):
    raise SystemExit(1)
PY
say "Python ${PY_VER}"

if [[ ! -d .venv ]]; then
  say "创建虚拟环境 .venv"
  python3 -m venv .venv
fi
source .venv/bin/activate

say "安装依赖"
python -m pip install -U pip -q
python -m pip install -r requirements.txt -q

if [[ ! -f .env ]]; then
  cp .env.example .env
  say "已生成 .env"
  if [[ -t 0 ]]; then
    printf '[径舟] 粘贴 LLM_API_KEY（必填，输入时不可见）: '
    read -r -s KEY
    printf '\n'
    if [[ -z "${KEY}" ]]; then
      say "未填写 Key。已留下 .env，填好后再运行 ./start.sh"
      exit 1
    fi
    printf '[径舟] API 地址 [默认 https://api.openai.com/v1]: '
    read -r URL
    URL="${URL:-https://api.openai.com/v1}"
    printf '[径舟] 模型名 [默认 gpt-4o]: '
    read -r MODEL
    MODEL="${MODEL:-gpt-4o}"
    python - "$KEY" "$URL" "$MODEL" <<'PY'
import pathlib, sys
key, url, model = sys.argv[1], sys.argv[2], sys.argv[3]
text = pathlib.Path(".env").read_text(encoding="utf-8")
text = text.replace("LLM_API_KEY=sk-your-key-here", f"LLM_API_KEY={key}")
text = text.replace("LLM_BASE_URL=https://api.openai.com/v1", f"LLM_BASE_URL={url}")
text = text.replace("LLM_MODEL=gpt-4o", f"LLM_MODEL={model}")
pathlib.Path(".env").write_text(text, encoding="utf-8")
PY
    say "环境已写入 .env"
  else
    say "非交互环境：请自行填入 LLM_API_KEY"
    exit 1
  fi
fi

set +e
python - <<'PY'
from pathlib import Path
raw = Path(".env").read_text(encoding="utf-8")
key = ""
for line in raw.splitlines():
    if line.startswith("LLM_API_KEY="):
        key = line.split("=", 1)[1].strip().strip('"')
if not key or key in {"sk-your-key-here", "changeme"}:
    raise SystemExit(2)
PY
KEY_STATUS=$?
set -e
if [[ "$KEY_STATUS" -eq 2 ]]; then
  say ".env 里的 LLM_API_KEY 还是占位符"
  exit 1
fi

PORT="${PORT:-8777}"
say "启动 http://127.0.0.1:${PORT}"
exec python -m uvicorn app.main:app --host 0.0.0.0 --port "$PORT"
