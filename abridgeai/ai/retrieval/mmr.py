"""Maximal Marginal Relevance (MMR) diversification.

Greedy MMR algorithm ported from
``backend/app/ai/haystack/pipelines/quiz_generation.py:1082-1106``. The
legacy version diversified by Jaccard 5-gram on raw ``content``; this
version diversifies by **cosine similarity over the chunk embeddings**
which is the textbook MMR formulation and matches what the cosine-search
already gave us — no extra tokenization step.

Trade-off: pure-Python cosine sim is O(top_k * remaining * dim). For the
typical case (top_k=12, remaining≤30, dim=1536) that's ~600k float ops,
under 10 ms on CPU — negligible next to the LLM call this feeds. If
Phase 5+ retrieval ever pushes through 100s of chunks per generation,
swap the inner loop for ``numpy.dot``.
"""

from __future__ import annotations

import math

from abridgeai.ai.retrieval.pgvector import ChunkWithDistance


def _cosine_sim(a: list[float] | None, b: list[float] | None) -> float:
    if not a or not b:
        return 0.0
    if len(a) != len(b):
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b, strict=True):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


def mmr_diversify(
    chunks: list[ChunkWithDistance],
    top_k: int,
    *,
    lambda_diversity: float = 0.5,
) -> list[ChunkWithDistance]:
    """Greedy MMR: balance relevance against diversity among picks.

    The score for a candidate ``c`` (given currently-selected ``S``) is::

        score(c) = lambda * (1 - distance(c))
                 - (1 - lambda) * max_{s in S} cosine_sim(c.embedding, s.embedding)

    where ``1 - distance`` converts pgvector cosine distance into a
    similarity. ``lambda_diversity`` is the trade-off knob:

      * ``0.0`` → ignore relevance, pick whatever is most different from
        the current selection
      * ``1.0`` → ignore diversity, pick by relevance only (i.e. the
        original distance ordering)
      * ``0.5`` (default) → balanced

    The first pick is always the most-relevant chunk (lowest
    ``distance``); successive picks are made greedily.

    Parameters
    ----------
    chunks
        Candidate pool, **already sorted by ascending distance**
        (vector_search returns rows in this order). Each entry must
        carry ``embedding`` (call ``vector_search(...,
        include_embeddings=True)``).
    top_k
        Maximum number of chunks to return.
    lambda_diversity
        Trade-off in ``[0.0, 1.0]``.

    Returns
    -------
    list[ChunkWithDistance]
        Up to ``top_k`` chunks, in selection order.
    """

    if not chunks or top_k <= 0:
        return []
    if top_k >= len(chunks):
        return list(chunks)

    lam = max(0.0, min(1.0, lambda_diversity))

    selected: list[ChunkWithDistance] = [chunks[0]]
    remaining: list[ChunkWithDistance] = list(chunks[1:])

    while remaining and len(selected) < top_k:
        best_idx = 0
        best_score = -math.inf
        for i, candidate in enumerate(remaining):
            relevance = 1.0 - candidate.distance
            max_sim = max(
                (_cosine_sim(candidate.embedding, s.embedding) for s in selected),
                default=0.0,
            )
            score = lam * relevance - (1.0 - lam) * max_sim
            if score > best_score:
                best_score = score
                best_idx = i
        selected.append(remaining.pop(best_idx))

    return selected


__all__ = ["mmr_diversify"]
