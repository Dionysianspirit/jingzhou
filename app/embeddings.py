from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np

_model = None
_pool: ThreadPoolExecutor | None = None


def load_model(name: str):
    global _model, _pool
    from sentence_transformers import SentenceTransformer

    _model = SentenceTransformer(name)
    _pool = ThreadPoolExecutor(max_workers=2)
    return _model


def ready() -> bool:
    return _model is not None


def encode(texts: list[str]) -> np.ndarray:
    if _model is None:
        raise RuntimeError("嵌入模型尚未加载")
    return _model.encode(texts, normalize_embeddings=True)


def encode_async(loop, texts: list[str]):
    if _pool is None:
        raise RuntimeError("嵌入线程池尚未启动")
    return loop.run_in_executor(_pool, encode, texts)


def shutdown() -> None:
    global _pool
    if _pool is not None:
        _pool.shutdown(wait=True)
        _pool = None
