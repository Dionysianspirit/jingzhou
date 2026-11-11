import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("JINGZHOU_DATA", ROOT / "data"))
STORE_DIR = DATA_DIR / "store"
SAMPLE_DIR = DATA_DIR / "sample"

LLM_KEY = os.environ.get("LLM_API_KEY", "")
LLM_URL = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o")
EMBED_MODEL_NAME = os.environ.get("EMBED_MODEL", "BAAI/bge-small-zh-v1.5")

CHUNK_CHARS = int(os.environ.get("CHUNK_CHARS", "2048"))
OVERLAP_CHARS = int(os.environ.get("OVERLAP_CHARS", "512"))
TOP_K = int(os.environ.get("TOP_K", "8"))
MIN_MERGE_CHARS = 200

# Agent: hard step budget clamped to 6..10 per spec.
AGENT_MAX_STEPS = max(6, min(10, int(os.environ.get("AGENT_MAX_STEPS", "8"))))
AGENT_DIR = DATA_DIR / "agent"

# Network exposure. The service has no login, so it listens on loopback only
# and accepts no cross-origin callers unless these are set explicitly.
HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "8777"))
CORS_ORIGINS = [o.strip() for o in os.environ.get("CORS_ORIGINS", "").split(",") if o.strip()]

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
