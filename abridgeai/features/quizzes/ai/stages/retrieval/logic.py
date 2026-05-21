"""Quiz retrieval orchestrator (T5.4).

Ports ``_retrieve_chunks`` from
``backend/app/ai/haystack/pipelines/quiz_generation.py:961-1016``.

Composes the T2.9 primitives — embedding, vector search, MMR — and the
quiz-specific anchor builder. Stages do not reimplement vector logic;
they only orchestrate it. Audit threading goes through
``EmbeddingClient.embed_query`` (T2.4): every embedding call carries
``pipeline_run_id`` so ``ai_model_calls`` rows roll up to the parent
run.

Phase 4 — contextual rerank
---------------------------
When ``settings.voyage_api_key`` is set, the post-MMR pool is reranked
by Voyage rerank-2.5 (cross-encoder) before being capped at
``final_top_k``. Cross-encoder logic lives in the sibling
``rerank.py`` module; this orchestrator only decides whether to call
it. The reranker is a graceful enhancement: when the key is unset OR
the call fails, retrieval falls back to the MMR-only path so quiz
generation never blocks on the rerank provider.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any
from uuid import UUID

from abridgeai.ai.llm.embeddings import EmbeddingClient
from abridgeai.ai.llm.voyage_rerank import VoyageRerankClient
from abridgeai.ai.retrieval import ChunkWithDistance, mmr_diversify, vector_search
from abridgeai.ai.retrieval.role_filter import split_by_role
from abridgeai.core.config import Settings, get_settings
from abridgeai.features.quizzes.ai.stages.retrieval.anchors import (
    MAX_ANCHORS,
    build_query_anchors,
)
from abridgeai.features.quizzes.ai.stages.retrieval.hybrid import (
    hybrid_search_for_anchor,
)
from abridgeai.features.quizzes.ai.stages.retrieval.rerank import rerank_pool

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from abridgeai.features.quizzes.models import Quiz

logger = logging.getLogger(__name__)

DEFAULT_PER_ANCHOR_TOP_K = 20
DEFAULT_FINAL_TOP_K = 12
DEFAULT_MMR_LAMBDA = 0.5

# When rerank is enabled we widen MMR output so the cross-encoder has a
# meaningful pool to score, then collapse back to ``final_top_k``.
RERANK_POOL_MULTIPLIER = 3
RERANK_POOL_CAP = 30


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
    rerank_client: VoyageRerankClient | None = None,
    settings: Settings | None = None,
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
    active_settings = settings or get_settings()
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

        if active_settings.hybrid_bm25_enabled:
            hits = await hybrid_search_for_anchor(
                db,
                anchor_text=anchor,
                embedding=embedding,
                course_id=course_id,
                lesson_ids=lesson_ids,
                settings=active_settings,
            )
        else:
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

    # Resolve rerank knob: caller may inject a stub client (test seam) OR
    # let us read settings. When no key is present, skip — MMR-only path.
    voyage_key = (
        active_settings.voyage_api_key.get_secret_value()
        if active_settings.voyage_api_key is not None
        else None
    )
    rerank_enabled = rerank_client is not None or bool(voyage_key)

    if rerank_enabled:
        # Widen MMR output so the cross-encoder has a meaningful pool to
        # score, then collapse back to ``final_top_k`` after rerank.
        mmr_top_k = min(
            RERANK_POOL_CAP,
            max(final_top_k, final_top_k * RERANK_POOL_MULTIPLIER),
        )
        # Cap summary/review/front_matter chunks before MMR (legacy
        # ``_split_by_role`` parity) so cover slides + ToC + recap
        # pages cannot soak up the rerank pool.
        body_priority = split_by_role(merged, mmr_top_k)
        diversified = mmr_diversify(
            body_priority,
            top_k=mmr_top_k,
            lambda_diversity=DEFAULT_MMR_LAMBDA,
        )
        reranked = await rerank_pool(
            diversified,
            anchor=capped_anchors[0],
            final_top_k=final_top_k,
            client=rerank_client,
            settings=active_settings,
            voyage_key=voyage_key,
        )
        return reranked, primary_embedding, capped_anchors

    # Same body-priority cap on the MMR-only path.
    body_priority = split_by_role(merged, final_top_k)
    diversified = mmr_diversify(
        body_priority,
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
