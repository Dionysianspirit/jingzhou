"""Paragraph-aware Chinese/English splitter with overlap."""

from __future__ import annotations


def chunk_text(
    text: str,
    chunk_chars: int = 2048,
    overlap_chars: int = 512,
    min_merge_chars: int = 200,
) -> list[str]:
    raw = (text or "").strip()
    if not raw:
        return []

    chunk_chars = max(1, chunk_chars)
    # The overlap is carried back from the cut, so it has to leave room to move
    # forward: capped at half a window, a hard-split paragraph costs at most 2x.
    overlap_chars = max(0, min(overlap_chars, chunk_chars // 2))

    paragraphs = [p.strip() for p in raw.split("\n\n") if p.strip()]
    if not paragraphs:
        paragraphs = [raw]

    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) <= chunk_chars:
            current = f"{current}\n\n{para}" if current else para
            continue
        if current:
            chunks.append(current)
        current = para
        while len(current) > chunk_chars:
            split_at = current.rfind("。", 0, chunk_chars)
            if split_at == -1:
                split_at = current.rfind(".", 0, chunk_chars)
            split_at = chunk_chars if split_at == -1 else split_at + 1
            if split_at <= overlap_chars:
                # The only sentence end in this window sits inside the overlap,
                # so cutting there would restart the next window at the same
                # place and never finish (a long table, formula or English run
                # after one early "。" does this). Cut at the window edge.
                split_at = chunk_chars
            chunks.append(current[:split_at])
            current = current[split_at - overlap_chars :]

    if current:
        chunks.append(current)

    merged: list[str] = []
    for c in chunks:
        if (
            merged
            and len(c) < min_merge_chars
            and len(merged[-1]) + len(c) + 2 <= chunk_chars
        ):
            merged[-1] += "\n\n" + c
        else:
            merged.append(c)
    return merged
