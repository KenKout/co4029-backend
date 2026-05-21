"""Hybrid (vector + BM25) retrieval per anchor (Phase 3 contextual RAG).

Split out from ``logic.py`` to keep the orchestrator under the 250-LOC
budget enforced by the retrieval-stage god-file test.

Per-anchor flow when ``settings.hybrid_bm25_enabled`` is True:

  1. ``vector_search`` with ``recall_k`` (default 150) — same embedding
     the orchestrator already computed.
  2. ``bm25_search`` with the same ``recall_k`` over the contextualized
     ``content_tsv`` column (Phase 3 migration 0015).
  3. ``reciprocal_rank_fusion`` with the cookbook defaults (semantic
     0.8, BM25 0.2). Each leg's ``1/(rank+1)`` contribution is weighted
     and summed.
  4. The fused list is mapped back to :class:`ChunkWithDistance` so
     downstream MMR + Voyage rerank stages stay unchanged. Distance is
     synthesized from ``1 - fused_score`` for chunks that came only
     from BM25 (so MMR's similarity ordering still makes sense); the
     embedding column is None for BM25-only hits, which the MMR
     primitive tolerates (falls back to relevance ordering).
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

from abridgeai.ai.retrieval import (
    ChunkWithDistance,
    bm25_search,
    vector_search,
)
from abridgeai.ai.retrieval.fusion import reciprocal_rank_fusion

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from abridgeai.core.config import Settings


async def hybrid_search_for_anchor(
    db: AsyncSession,
    *,
    anchor_text: str,
    embedding: list[float],
    course_id: UUID | None,
    lesson_ids: list[UUID] | None,
    settings: Settings,
) -> list[ChunkWithDistance]:
    """Run vector + BM25 + RRF fusion for a single anchor.

    Returns chunks shaped as :class:`ChunkWithDistance` so the caller's
    pool / MMR / rerank pipeline stays oblivious to the hybrid path.
    """
    semantic_hits = await vector_search(
        db,
        embedding,
        course_id=course_id,
        lesson_ids=lesson_ids,
        top_k=settings.hybrid_recall_k,
        include_embeddings=True,
    )
    bm25_hits = await bm25_search(
        db,
        anchor_text,
        course_id=course_id,
        lesson_ids=lesson_ids,
        top_k=settings.hybrid_recall_k,
    )
    fused = reciprocal_rank_fusion(
        semantic_hits,
        bm25_hits,
        semantic_weight=settings.hybrid_semantic_weight,
        bm25_weight=settings.hybrid_bm25_weight,
    )
    return [
        ChunkWithDistance(
            chunk_id=row.chunk_id,
            material_version_id=row.material_version_id,
            course_id=row.course_id,
            lesson_id=row.lesson_id,
            content=row.content,
            # When the chunk came from the semantic leg we keep the real
            # cosine distance (smaller == better); for BM25-only hits we
            # synthesize ``1 - fused_score`` so a higher fused_score
            # still surfaces with a smaller distance for downstream
            # ordering. distance is a sortable proxy here, not a true
            # cosine measure — that's an acceptable contract because
            # downstream consumers (MMR, rerank) only use distance for
            # initial ordering before re-scoring anyway.
            distance=row.distance if row.distance is not None else 1.0 - row.fused_score,
            embedding=row.embedding,
            metadata=row.metadata,
        )
        for row in fused
    ]


__all__ = ["hybrid_search_for_anchor"]
