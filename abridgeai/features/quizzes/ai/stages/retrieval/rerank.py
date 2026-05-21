"""Voyage rerank-2.5 helper for the quiz retrieval stage (Phase 4).

Split out from ``logic.py`` to keep the orchestrator under the 250-LOC
budget (see ``test_no_god_file_in_retrieval_stage``). This module owns
the cross-encoder reorder + fallback policy; the orchestrator just
decides whether to call it.

Failure modes are non-fatal — any provider error is logged and the
MMR pool (capped at ``final_top_k``) is returned unchanged. Quiz
generation must not block on a third-party reranker outage.
"""

from __future__ import annotations

import logging

from abridgeai.ai.llm.errors import ProviderError, ResponseFormatError
from abridgeai.ai.llm.voyage_rerank import VoyageRerankClient
from abridgeai.ai.retrieval import ChunkWithDistance
from abridgeai.core.config import Settings

logger = logging.getLogger(__name__)


async def rerank_pool(
    pool: list[ChunkWithDistance],
    *,
    anchor: str,
    final_top_k: int,
    client: VoyageRerankClient | None,
    settings: Settings,
    voyage_key: str | None,
) -> list[ChunkWithDistance]:
    """Reorder ``pool`` using Voyage rerank-2.5 and return top-K.

    Parameters
    ----------
    pool
        Post-MMR candidate chunks (already diversified).
    anchor
        Primary query anchor — the cross-encoder scores ``(anchor, doc)``.
    final_top_k
        Cap on returned chunks after rerank.
    client
        Optional injected rerank client (test seam). When ``None`` and
        ``voyage_key`` is set, a fresh client is constructed from
        ``settings``.
    settings
        Live :class:`Settings` for base URL / model / timeout.
    voyage_key
        Resolved API key (``settings.voyage_api_key.get_secret_value()``),
        passed in so the orchestrator can decide rerank-on/off without
        re-reading SecretStr.

    Returns
    -------
    list[ChunkWithDistance]
        Reordered top-``final_top_k`` chunks. On any provider failure,
        returns ``pool[:final_top_k]`` unchanged.
    """
    if not pool:
        return []
    fallback = pool[:final_top_k]

    active_client = client
    if active_client is None:
        if not voyage_key:
            return fallback
        active_client = VoyageRerankClient(
            api_key=voyage_key,
            model=settings.voyage_rerank_model,
            base_url=settings.voyage_base_url,
            timeout_s=settings.voyage_rerank_timeout_seconds,
        )

    documents = [chunk.content for chunk in pool]
    try:
        results, latency_ms = await active_client.rerank(
            anchor,
            documents,
            top_k=final_top_k,
        )
    except (ProviderError, ResponseFormatError) as exc:
        logger.warning(
            "voyage rerank failed (%s); falling back to MMR-only top-%d",
            type(exc).__name__,
            final_top_k,
        )
        return fallback

    if not results:
        return fallback

    logger.info(
        "voyage rerank: scored %d docs in %dms, kept top %d",
        len(documents),
        latency_ms,
        min(len(results), final_top_k),
    )

    reordered: list[ChunkWithDistance] = []
    seen: set[int] = set()
    for row in results[:final_top_k]:
        if row.index in seen or row.index < 0 or row.index >= len(pool):
            continue
        seen.add(row.index)
        reordered.append(pool[row.index])

    # Defensive: if Voyage drops some indices we still want to fill up to
    # ``final_top_k`` with the MMR ordering as a backstop.
    if len(reordered) < final_top_k:
        for idx, chunk in enumerate(pool):
            if idx in seen:
                continue
            reordered.append(chunk)
            if len(reordered) >= final_top_k:
                break

    return reordered


__all__ = ["rerank_pool"]
