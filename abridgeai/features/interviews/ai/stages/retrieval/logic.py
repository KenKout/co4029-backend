"""Interview retrieval orchestrator (T6.4).

Ports the retrieval prelude from
``backend/app/ai/haystack/pipelines/interview_generation.py:42-54`` and
extends it with the multi-anchor + KG-aware composition the plan calls
out for interviews:

* Builds anchors via :func:`build_interview_anchors` (focus_topics →
  KG concepts → lesson titles → config.title).
* Runs :func:`vector_search` per anchor (capped at :data:`MAX_ANCHORS`)
  and merges the per-anchor pools by chunk id keeping the smallest
  cosine distance.
* Diversifies with :func:`mmr_diversify` (λ=0.5, top_k=12 by default).

The public entry point is :func:`retrieve_interview_context`. It
returns an :class:`InterviewRetrievalContext` that the generation
stage (T6.5) consumes — primary chunks, KG concepts, primary embedding
and audit metadata are kept separate so the prompt builder can decide
how to weight each pool.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol
from uuid import UUID

from abridgeai.ai.knowledge_graph.schemas import Concept
from abridgeai.ai.llm.embeddings import EmbeddingClient
from abridgeai.ai.retrieval import ChunkWithDistance, mmr_diversify, vector_search
from abridgeai.ai.retrieval.role_filter import split_by_role
from abridgeai.features.interviews.ai.stages.retrieval.anchors import (
    MAX_ANCHORS,
    build_interview_anchors,
)
from abridgeai.features.interviews.ai.stages.retrieval.metadata import (
    retrieval_metadata,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from abridgeai.features.interviews.models import InterviewConfig


class _GenerationRunLike(Protocol):
    """Structural type for the generation-run row.

    Avoids a cross-feature import of ``features.quizzes.models.GenerationRun``
    (currently the stop-gap home per T5.1 notepad). The retrieval stage
    only reads ``id`` / ``config_json`` / ``course_id`` so a Protocol is
    sufficient and keeps the import-linter ``Features are independent``
    contract green.
    """

    id: UUID
    config_json: dict[str, Any]
    course_id: UUID | None


logger = logging.getLogger(__name__)

DEFAULT_PER_ANCHOR_TOP_K = 20
DEFAULT_FINAL_TOP_K = 12
DEFAULT_MMR_LAMBDA = 0.5


@dataclass(frozen=True)
class InterviewRetrievalContext:
    """Aggregated retrieval payload consumed by the generation stage."""

    chunks: list[ChunkWithDistance]
    kg_concepts: list[Concept]
    query_embedding: list[float]
    anchors: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


async def retrieve_interview_context(
    db: AsyncSession,
    *,
    run: _GenerationRunLike,
    config: InterviewConfig,
    kg_context_enabled: bool = True,
    pipeline_run_id: UUID | None = None,
    per_anchor_top_k: int = DEFAULT_PER_ANCHOR_TOP_K,
    final_top_k: int = DEFAULT_FINAL_TOP_K,
    embedding_client: EmbeddingClient | None = None,
) -> InterviewRetrievalContext:
    """Multi-anchor retrieval with KG + student-weakness augmentation.

    The function is the single public entry point for T6.4. T6.5
    (generation) calls it once per run before composing the LLM
    prompt; the returned context is kept verbatim in
    ``run.config_json["retrieval"]`` for audit replay.
    """

    run_config: dict[str, Any] = dict(run.config_json or {})

    anchors, kg_concepts = await build_interview_anchors(
        db,
        config,
        run_config,
        kg_context_enabled=kg_context_enabled,
    )

    if not anchors:
        meta = retrieval_metadata(
            [],
            anchors=[],
            kg_concepts=kg_concepts,
            primary_embedding=None,
            kg_context_enabled=kg_context_enabled,
        )
        return InterviewRetrievalContext(
            chunks=[],
            kg_concepts=kg_concepts,
            query_embedding=[],
            anchors=[],
            metadata=meta,
        )

    capped_anchors = anchors[:MAX_ANCHORS]
    course_id = run.course_id
    lesson_ids = _maybe_uuid_list(
        run_config.get("lesson_ids") or run_config.get("source_lesson_ids")
    )

    client = embedding_client or EmbeddingClient()
    pool: dict[UUID, ChunkWithDistance] = {}
    primary_embedding: list[float] = []

    for index, anchor in enumerate(capped_anchors):
        embedding = await client.embed_query(
            anchor,
            db=db,
            pipeline_run_id=pipeline_run_id,
            parent_run_id=run.id,
        )
        if index == 0:
            primary_embedding = embedding

        hits = await vector_search(
            db,
            embedding,
            course_id=course_id,
            lesson_ids=lesson_ids,
            top_k=per_anchor_top_k,
            include_embeddings=True,
        )
        for hit in hits:
            existing = pool.get(hit.chunk_id)
            if existing is None or hit.distance < existing.distance:
                pool[hit.chunk_id] = hit

    if pool:
        merged = sorted(pool.values(), key=lambda c: c.distance)
        # Cap summary/review/front_matter chunks before MMR, matching the quiz
        # MMR-only path. Without it a cover slide, table of contents or recap
        # page can win on cosine distance and become the source a question is
        # grounded in — the interview then probes the syllabus rather than the
        # material. Chunks whose metadata lacks ``content_role`` count as body,
        # so pre-classification corpora are unaffected.
        body_priority = split_by_role(merged, final_top_k)
        diversified = mmr_diversify(
            body_priority,
            top_k=final_top_k,
            lambda_diversity=DEFAULT_MMR_LAMBDA,
        )
    else:
        diversified = []

    meta = retrieval_metadata(
        diversified,
        anchors=capped_anchors,
        kg_concepts=kg_concepts,
        primary_embedding=primary_embedding,
        kg_context_enabled=kg_context_enabled,
    )

    return InterviewRetrievalContext(
        chunks=diversified,
        kg_concepts=kg_concepts,
        query_embedding=primary_embedding,
        anchors=capped_anchors,
        metadata=meta,
    )


def _maybe_uuid_list(raw: object) -> list[UUID] | None:
    if not raw or not isinstance(raw, list):
        return None
    parsed: list[UUID] = []
    for item in raw:
        try:
            parsed.append(UUID(str(item)))
        except (ValueError, TypeError):
            continue
    return parsed or None


__all__ = [
    "InterviewRetrievalContext",
    "retrieve_interview_context",
]
