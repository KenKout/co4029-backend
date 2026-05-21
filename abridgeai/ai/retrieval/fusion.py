"""Weighted Reciprocal Rank Fusion for hybrid retrieval (Phase 3).

Ports the Anthropic cookbook ``retrieve_advanced`` fusion logic
(``capabilities/contextual-embeddings/guide.ipynb``, cells 26-27) to
this codebase: weighted ``1/(rank+1)`` over two ranked lists
(semantic + BM25), then re-rank the union by descending fused score.

Anthropic's defaults — ``semantic_weight=0.8``, ``bm25_weight=0.2``,
``num_chunks_to_recall=150`` — are codified in
:mod:`abridgeai.core.config` as ``hybrid_*`` settings. The fusion is
rank-based (not score-based) so it's robust to score-distribution
mismatch between cosine distance and ``ts_rank_cd``.

Why a sibling module, not inline in ``logic.py``?
  Phase 4 already pushed ``logic.py`` close to the 250-LOC budget; the
  retrieval-stage god-file test enforces that ceiling. Fusion is a
  pure function of two ranked lists, easy to unit-test in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from uuid import UUID

if TYPE_CHECKING:
    from abridgeai.ai.retrieval import ChunkWithDistance, ChunkWithRank


@dataclass(frozen=True)
class FusedChunk:
    """A chunk surfaced by hybrid retrieval, after RRF fusion.

    ``fused_score`` is the weighted ``1/(rank+1)`` total used for
    ranking. ``sources`` records which retriever(s) found the chunk —
    ``{"vector"}``, ``{"bm25"}``, or both. Useful for downstream audit
    and for callers that want to filter to dual-source matches.
    ``embedding`` carries through the vector when the chunk came from
    the semantic leg, so MMR can run after fusion if desired.
    """

    chunk_id: UUID
    material_version_id: UUID
    course_id: UUID | None
    lesson_id: UUID | None
    content: str
    fused_score: float
    sources: frozenset[str]
    distance: float | None
    embedding: list[float] | None = None
    metadata: dict[str, object] | None = None


def reciprocal_rank_fusion(
    semantic_hits: list[ChunkWithDistance],
    bm25_hits: list[ChunkWithRank],
    *,
    semantic_weight: float = 0.8,
    bm25_weight: float = 0.2,
) -> list[FusedChunk]:
    """Fuse two ranked retrieval lists with weighted RRF.

    Anthropic cookbook formula::

        score(c) = semantic_weight * (1 / (rank_semantic(c) + 1))
                 + bm25_weight     * (1 / (rank_bm25(c)     + 1))

    where ``rank_*(c)`` is the 0-indexed position of chunk ``c`` in
    each list, and chunks not present in a list contribute 0 from that
    leg. Output is ranked by descending fused score; ties broken by
    chunk_id for determinism.

    Parameters
    ----------
    semantic_hits
        Output of :func:`vector_search`, ordered by ascending distance.
    bm25_hits
        Output of :func:`bm25_search`, ordered by descending rank.
    semantic_weight, bm25_weight
        Per-leg weights (Anthropic default 0.8 / 0.2). Caller is
        responsible for keeping these in [0,1] and ideally summing to 1.

    Returns
    -------
    list[FusedChunk]
        Union of both lists, ranked by fused score (descending).
    """
    if not semantic_hits and not bm25_hits:
        return []

    # Build per-chunk metadata + per-leg ranks. We index by chunk_id,
    # which is the join key across both retrievers.
    semantic_rank: dict[UUID, int] = {hit.chunk_id: i for i, hit in enumerate(semantic_hits)}
    bm25_rank: dict[UUID, int] = {hit.chunk_id: i for i, hit in enumerate(bm25_hits)}

    # Materialize chunk metadata once per id, preferring the semantic
    # row (carries embedding + distance) when both present.
    metadata: dict[UUID, FusedChunk] = {}
    for sem_hit in semantic_hits:
        metadata[sem_hit.chunk_id] = FusedChunk(
            chunk_id=sem_hit.chunk_id,
            material_version_id=sem_hit.material_version_id,
            course_id=sem_hit.course_id,
            lesson_id=sem_hit.lesson_id,
            content=sem_hit.content,
            fused_score=0.0,
            sources=frozenset({"vector"}),
            distance=sem_hit.distance,
            embedding=sem_hit.embedding,
            metadata=sem_hit.metadata,
        )
    for bm25_hit in bm25_hits:
        existing = metadata.get(bm25_hit.chunk_id)
        if existing is None:
            metadata[bm25_hit.chunk_id] = FusedChunk(
                chunk_id=bm25_hit.chunk_id,
                material_version_id=bm25_hit.material_version_id,
                course_id=bm25_hit.course_id,
                lesson_id=bm25_hit.lesson_id,
                content=bm25_hit.content,
                fused_score=0.0,
                sources=frozenset({"bm25"}),
                distance=None,
                embedding=None,
                metadata=bm25_hit.metadata,
            )
        else:
            metadata[bm25_hit.chunk_id] = FusedChunk(
                chunk_id=existing.chunk_id,
                material_version_id=existing.material_version_id,
                course_id=existing.course_id,
                lesson_id=existing.lesson_id,
                content=existing.content,
                fused_score=existing.fused_score,
                sources=existing.sources | frozenset({"bm25"}),
                distance=existing.distance,
                embedding=existing.embedding,
                metadata=existing.metadata if existing.metadata is not None else bm25_hit.metadata,
            )

    # Compute fused scores.
    fused: list[FusedChunk] = []
    for chunk_id, base in metadata.items():
        score = 0.0
        if chunk_id in semantic_rank:
            score += semantic_weight * (1.0 / (semantic_rank[chunk_id] + 1))
        if chunk_id in bm25_rank:
            score += bm25_weight * (1.0 / (bm25_rank[chunk_id] + 1))
        fused.append(
            FusedChunk(
                chunk_id=base.chunk_id,
                material_version_id=base.material_version_id,
                course_id=base.course_id,
                lesson_id=base.lesson_id,
                content=base.content,
                fused_score=score,
                sources=base.sources,
                distance=base.distance,
                embedding=base.embedding,
                metadata=base.metadata,
            )
        )

    # Sort by fused_score desc; chunk_id asc as a stable tiebreaker.
    fused.sort(key=lambda c: (-c.fused_score, str(c.chunk_id)))
    return fused


__all__ = ["FusedChunk", "reciprocal_rank_fusion"]
