#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [[ ! -f .env ]]; then
  echo "[径舟] 未找到 .env。请先：cp .env.example .env 并填入 LLM_API_KEY"
  exit 1
fi

PORT="${PORT:-8777}"
python3 -m pip install -r requirements.txt -q
echo "[径舟] 启动 http://127.0.0.1:${PORT}"
exec python3 -m uvicorn app.main:app --host 0.0.0.0 --port "$PORT"
