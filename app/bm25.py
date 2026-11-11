"""Lexical retrieval: BM25 over a tokenization that needs no word segmenter.

Dense vectors are good at "what is this about" and weak at "contains exactly
this term": a formula name, a theorem, an abbreviation. BM25 covers that side.
Everything here is pure Python and deterministic, so it is unit-testable
without loading the embedding model.

Tokens:
- CJK runs become overlapping character bigrams (a lone CJK char stays a
  unigram). Bigrams approximate Chinese words without a dictionary.
- Latin / digit runs become one lowercased token each (``rank``, ``0.83``).
- Any other letter (Greek ``λ``, etc.) is a single token.
"""

from __future__ import annotations

import math
import re
from collections import Counter

K1 = 1.5
B = 0.75
# Reciprocal-rank-fusion constant from the original RRF paper; large enough
# that one list's top hit cannot drown out agreement between the two lists.
RRF_K = 60

_LATIN = re.compile(r"[A-Za-z][A-Za-z0-9]*|\d+(?:\.\d+)?")


def _is_cjk(ch: str) -> bool:
    code = ord(ch)
    return (
        0x4E00 <= code <= 0x9FFF  # CJK Unified Ideographs
        or 0x3400 <= code <= 0x4DBF  # Extension A
        or 0xF900 <= code <= 0xFAFF  # Compatibility Ideographs
        or 0x3040 <= code <= 0x30FF  # Hiragana / Katakana
    )


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    run: list[str] = []

    def flush_cjk() -> None:
        if len(run) == 1:
            tokens.append(run[0])
        else:
            tokens.extend(run[i] + run[i + 1] for i in range(len(run) - 1))
        run.clear()

    i = 0
    n = len(text or "")
    while i < n:
        ch = text[i]
        if _is_cjk(ch):
            run.append(ch)
            i += 1
            continue
        if run:
            flush_cjk()
        match = _LATIN.match(text, i)
        if match:
            tokens.append(match.group(0).lower())
            i = match.end()
            continue
        if ch.isalpha():
            tokens.append(ch.lower())
        i += 1
    if run:
        flush_cjk()
    return tokens


class LexicalIndex:
    """Per-document postings. Corpus statistics are combined at query time so
    one index per document can serve any selection of documents."""

    def __init__(self, chunks: list[str]):
        self.n_chunks = len(chunks)
        self.lengths: list[int] = []
        self.postings: dict[str, list[tuple[int, int]]] = {}
        for idx, chunk in enumerate(chunks):
            counts = Counter(tokenize(chunk))
            self.lengths.append(sum(counts.values()))
            for token, tf in counts.items():
                self.postings.setdefault(token, []).append((idx, tf))
        self.total_len = sum(self.lengths)

    def df(self, token: str) -> int:
        return len(self.postings.get(token, ()))


def bm25_scores(query: str, indexes: dict[str, LexicalIndex]) -> dict[tuple[str, int], float]:
    """BM25 score per (doc_id, chunk_idx); chunks sharing no query term are absent."""
    terms = Counter(tokenize(query))
    total_chunks = sum(ix.n_chunks for ix in indexes.values())
    if not terms or not total_chunks:
        return {}
    avg_len = sum(ix.total_len for ix in indexes.values()) / total_chunks or 1.0
    scores: dict[tuple[str, int], float] = {}
    for token in terms:
        df = sum(ix.df(token) for ix in indexes.values())
        if not df:
            continue
        # the "+1" form keeps idf positive even for terms in most chunks
        idf = math.log(1.0 + (total_chunks - df + 0.5) / (df + 0.5))
        for doc_id, ix in indexes.items():
            for chunk_idx, tf in ix.postings.get(token, ()):
                norm = 1.0 - B + B * ix.lengths[chunk_idx] / avg_len
                key = (doc_id, chunk_idx)
                scores[key] = scores.get(key, 0.0) + idf * tf * (K1 + 1.0) / (tf + K1 * norm)
    return scores


def rank_map(scores: dict, *, descending: bool = True) -> dict:
    """1-based rank per key; ties broken by key so the order is reproducible."""
    ordered = sorted(scores, key=lambda key: (-scores[key] if descending else scores[key], key))
    return {key: rank for rank, key in enumerate(ordered, start=1)}


def rrf(rank_lists: list[dict], k: int = RRF_K) -> dict:
    """Reciprocal rank fusion: sum of 1/(k + rank) over the lists a key is in."""
    fused: dict = {}
    for ranks in rank_lists:
        for key, rank in ranks.items():
            fused[key] = fused.get(key, 0.0) + 1.0 / (k + rank)
    return fused
