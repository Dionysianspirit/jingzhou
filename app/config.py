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

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
