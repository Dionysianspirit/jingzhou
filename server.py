"""Backward-compatible entrypoint. Prefer: uvicorn app.main:app --port 8777

Listens on 127.0.0.1 by default: the service has no login, so exposing it to
the whole network would let anyone nearby read the indexed documents and spend
the configured LLM key. Set HOST=0.0.0.0 to opt in to LAN access.
"""

import uvicorn

from app.config import HOST, PORT

if __name__ == "__main__":
    uvicorn.run("app.main:app", host=HOST, port=PORT, reload=False)
