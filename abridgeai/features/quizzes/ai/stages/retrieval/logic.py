"""Quiz retrieval orchestrator (T5.4).

Ports ``_retrieve_chunks`` from
``backend/app/ai/haystack/pipelines/quiz_generation.py:961-1016``.

Composes the T2.9 primitives — embedding, vector search, MMR — and the
quiz-specific anchor builder. Stages do not reimplement vector logic;
they only orchestrate it. Audit threading goes through
``EmbeddingClient.embed_query`` (T2.4): every embedding call carries
``pipeline_run_id`` so ``ai_model_calls`` rows roll up to the parent
run.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import UUID

from abridgeai.ai.llm.embeddings import EmbeddingClient
from abridgeai.ai.retrieval import ChunkWithDistance, mmr_diversify, vector_search
from abridgeai.features.quizzes.ai.stages.retrieval.anchors import (
    MAX_ANCHORS,
    build_query_anchors,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from abridgeai.features.quizzes.models import Quiz

DEFAULT_PER_ANCHOR_TOP_K = 20
DEFAULT_FINAL_TOP_K = 12
DEFAULT_MMR_LAMBDA = 0.5


async def retrieve_chunks(
    db: AsyncSession,
    run_id: UUID,
    quiz: Quiz,
    config: dict[str, Any],
    *,
    question_anchor: str | None = None,
    kg_context_enabled: bool = True,
    pipeline_run_id: UUID | None = None,
    per_anchor_top_k: int = DEFAULT_PER_ANCHOR_TOP_K,
    final_top_k: int = DEFAULT_FINAL_TOP_K,
    embedding_client: EmbeddingClient | None = None,
) -> tuple[list[ChunkWithDistance], list[float], list[str]]:
    """Multi-anchor retrieval composing vector_search + MMR.

    Steps:
        1. Build anchor list via :func:`build_query_anchors`
           (precedence: explicit ``question_anchor`` > ``focus_topics``
           > KG concepts > lesson titles > quiz title).
        2. For each anchor (capped at :data:`MAX_ANCHORS`), embed and
           run :func:`vector_search` with ``include_embeddings=True``
           so MMR can run on the merged pool.
        3. Merge per-anchor results by ``chunk_id`` keeping the smallest
           distance seen for each chunk.
        4. Run :func:`mmr_diversify` over the merged pool to drop
           cross-anchor near-duplicates.

    Parameters
    ----------
    db
        Async session — passed through to ``vector_search`` (raw SQL
        for now, ORM in Phase 4).
    run_id
        Generation-run row id; threaded into the embedding-client
        ``parent_run_id`` so ``ai_model_calls`` rows roll up.
    quiz
        Quiz draft; used for module / lesson scoping plus the title
        fallback in the anchor builder.
    config
        Run config dict (``focus_topics``, ``source_lesson_ids``,
        ``course_id``, ``lesson_ids``).
    question_anchor
        Optional override anchor (regenerate-one path); skips the
        precedence chain.
    kg_context_enabled
        When ``False``, the anchor builder skips the Neo4j lookup.
    pipeline_run_id
        Pipeline-run id for ``ai_model_calls`` audit rollups (T2.4).
    per_anchor_top_k
        Vector search ``top_k`` per anchor before MMR. The legacy
        pipeline used 6 — Phase 5 widens to 20 so MMR has more pool.
    final_top_k
        Cap on returned chunks after MMR.
    embedding_client
        Inject a custom client (test seam). Defaults to ``EmbeddingClient()``.

    Returns
    -------
    tuple[list[ChunkWithDistance], list[float], list[str]]
        ``(chunks, primary_embedding, anchors)`` — the primary embedding
        and full anchor list flow into :func:`retrieval_metadata` for
        audit.
    """

    anchors = await build_query_anchors(
        db,
        quiz,
        config,
        question_hint=question_anchor,
        kg_context_enabled=kg_context_enabled,
    )
    if not anchors:
        return [], [], []

    capped_anchors = anchors[:MAX_ANCHORS]
    course_id = _maybe_uuid(config.get("course_id"))
    lesson_ids = _maybe_uuid_list(config.get("lesson_ids") or config.get("source_lesson_ids"))

    client = embedding_client or EmbeddingClient()
    pool: dict[UUID, ChunkWithDistance] = {}
    primary_embedding: list[float] = []

    for index, anchor in enumerate(capped_anchors):
        embedding = await client.embed_query(
            anchor,
            db=db,
            pipeline_run_id=pipeline_run_id,
            parent_run_id=run_id,
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

    if not pool:
        return [], primary_embedding, capped_anchors

    merged = sorted(pool.values(), key=lambda c: c.distance)
    diversified = mmr_diversify(
        merged,
        top_k=final_top_k,
        lambda_diversity=DEFAULT_MMR_LAMBDA,
    )
    return diversified, primary_embedding, capped_anchors


def _maybe_uuid(raw: object) -> UUID | None:
    if raw is None:
        return None
    try:
        return UUID(str(raw))
    except (ValueError, TypeError):
        return None


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


__all__ = ["retrieve_chunks"]
