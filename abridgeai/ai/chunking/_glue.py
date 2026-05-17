"""Stage B — embedding-based semantic glue.

Walks the rule-based ``RawChunk`` list once, merging adjacent chunks whose
embeddings exceed ``threshold`` cosine similarity AND share a content
role AND fit under ``max_window_tokens``. The result is a smaller list of
"semantic windows": each window is itself a ``RawChunk`` whose metadata
carries ``glue_group_id``, ``member_indices``, ``page_range``, and the
aggregated ``token_count``.

When no embedder is available, ``glue_by_similarity`` returns the input
unchanged with single-member glue groups so downstream stages stay
oblivious to whether glue ran.

Inspired by Greg Kamradt's "Level 4" splitting and Jina's late-chunking
research (arXiv:2409.04701).
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Awaitable
from typing import Any, Protocol

from abridgeai.ai.chunking._window import page_range_of
from abridgeai.ai.chunking.base import RawChunk


class Embedder(Protocol):
    async def embed(self, texts: list[str]) -> list[list[float]]: ...


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


async def _to_list(value: Awaitable[list[list[float]]] | list[list[float]]) -> list[list[float]]:
    if asyncio.iscoroutine(value):
        result: list[list[float]] = await value
        return result
    return value  # type: ignore[return-value]


async def glue_by_similarity(
    chunks: list[RawChunk],
    embedder: Embedder | None,
    *,
    threshold: float = 0.72,
    max_window_tokens: int = 2000,
    min_window_tokens: int = 0,
) -> list[RawChunk]:
    """Merge adjacent chunks whose embeddings exceed ``threshold``.

    Greedy linear pass preserves reading order (jumping around would lose
    continuity). Force-breaks on role change so Stage C never has to
    classify a mixed window.
    """
    if not chunks:
        return []
    if embedder is None:
        return [_single_member_window(c, group_id=i) for i, c in enumerate(chunks)]

    embeddings = await embedder.embed([c.content for c in chunks])
    if len(embeddings) != len(chunks):
        raise ValueError(f"embedder returned {len(embeddings)} vectors for {len(chunks)} chunks")

    windows: list[RawChunk] = []
    cur_members: list[RawChunk] = [chunks[0]]
    cur_tokens = _tokens_of(chunks[0])
    cur_role = _role_of(chunks[0])

    for i in range(1, len(chunks)):
        prev_emb = embeddings[i - 1]
        curr_emb = embeddings[i]
        next_role = _role_of(chunks[i])
        next_tokens = _tokens_of(chunks[i])

        sim = _cosine(prev_emb, curr_emb)
        size_ok = (cur_tokens + next_tokens) <= max_window_tokens
        role_ok = next_role == cur_role

        if sim >= threshold and size_ok and role_ok:
            cur_members.append(chunks[i])
            cur_tokens += next_tokens
        else:
            windows.append(_finalize_window(cur_members, group_id=len(windows)))
            cur_members = [chunks[i]]
            cur_tokens = next_tokens
            cur_role = next_role

    windows.append(_finalize_window(cur_members, group_id=len(windows)))

    if min_window_tokens > 0:
        windows = _absorb_tiny_windows(
            windows, min_tokens=min_window_tokens, max_tokens=max_window_tokens
        )
    return windows


def _absorb_tiny_windows(
    windows: list[RawChunk],
    *,
    min_tokens: int,
    max_tokens: int,
) -> list[RawChunk]:
    if not windows:
        return windows

    result: list[RawChunk] = list(windows)
    i = 0
    while i < len(result):
        w = result[i]
        if _tokens_of(w) >= min_tokens:
            i += 1
            continue
        left = result[i - 1] if i > 0 else None
        right = result[i + 1] if i + 1 < len(result) else None

        candidates: list[tuple[int, str]] = []
        if right is not None and _eligible(right, w, max_tokens):
            candidates.append((_tokens_of(right), "right"))
        if left is not None and _eligible(left, w, max_tokens):
            candidates.append((_tokens_of(left), "left"))

        if not candidates:
            i += 1
            continue
        candidates.sort()
        target_dir = candidates[0][1]

        if target_dir == "right" and right is not None:
            merged = _merge(w, right, group_id=int(w.metadata["glue_group_id"]))
            result[i] = merged
            del result[i + 1]
        elif left is not None:
            merged = _merge(left, w, group_id=int(left.metadata["glue_group_id"]))
            result[i - 1] = merged
            del result[i]
            i -= 1
    for idx, win in enumerate(result):
        new_md = dict(win.metadata)
        new_md["glue_group_id"] = idx
        result[idx] = RawChunk(content=win.content, chunk_index=idx, metadata=new_md)
    return result


def _eligible(neighbour: RawChunk, current: RawChunk, max_tokens: int) -> bool:
    if _role_of(neighbour) != _role_of(current):
        return False
    return _tokens_of(neighbour) + _tokens_of(current) <= max_tokens


def _merge(a: RawChunk, b: RawChunk, *, group_id: int) -> RawChunk:
    members_a = list(a.metadata.get("member_indices") or [a.chunk_index])
    members_b = list(b.metadata.get("member_indices") or [b.chunk_index])
    merged_members = members_a + members_b
    merged_text = a.content + "\n\n" + b.content
    merged_tokens = _tokens_of(a) + _tokens_of(b)
    page_lo_a, page_hi_a = a.metadata.get("page_range") or (None, None)
    page_lo_b, page_hi_b = b.metadata.get("page_range") or (None, None)
    pages = [p for p in (page_lo_a, page_hi_a, page_lo_b, page_hi_b) if isinstance(p, int)]
    page_range = (min(pages), max(pages)) if pages else (None, None)

    md: dict[str, Any] = dict(a.metadata)
    md["member_indices"] = merged_members
    md["token_count"] = merged_tokens
    md["page_range"] = page_range
    md["content_role"] = _role_of(a)
    md["glue_group_id"] = group_id
    return RawChunk(content=merged_text, chunk_index=group_id, metadata=md)


def _finalize_window(members: list[RawChunk], *, group_id: int) -> RawChunk:
    text = "\n\n".join(m.content for m in members)
    tokens = sum(_tokens_of(m) for m in members)
    md: dict[str, Any] = {
        "member_indices": [m.chunk_index for m in members],
        "token_count": tokens,
        "page_range": page_range_of(members),
        "content_role": _role_of(members[0]),
        "glue_group_id": group_id,
        "source_type": members[0].metadata.get("source_type"),
    }
    section = members[0].metadata.get("section")
    if section:
        md["section"] = section
    return RawChunk(content=text, chunk_index=group_id, metadata=md)


def _single_member_window(chunk: RawChunk, *, group_id: int) -> RawChunk:
    md = dict(chunk.metadata)
    md["member_indices"] = [chunk.chunk_index]
    md.setdefault("token_count", 0)
    md["page_range"] = page_range_of([chunk])
    md.setdefault("content_role", "body")
    md["glue_group_id"] = group_id
    return RawChunk(content=chunk.content, chunk_index=group_id, metadata=md)


def _tokens_of(chunk: RawChunk) -> int:
    val = chunk.metadata.get("token_count")
    return int(val) if isinstance(val, int) else 0


def _role_of(chunk: RawChunk) -> str:
    return str(chunk.metadata.get("content_role") or "body")


__all__ = ["Embedder", "glue_by_similarity"]
