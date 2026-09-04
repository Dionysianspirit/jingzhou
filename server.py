"""Backward-compatible entrypoint. Prefer: uvicorn app.main:app --host 0.0.0.0 --port 8777"""

import os

import uvicorn

from app.main import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8777"))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=False)
