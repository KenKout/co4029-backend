"""Three-stage semantic chunker orchestrator.

Composes the private ``_window`` / ``_glue`` / ``_enrich`` helpers into a
single public chunker. Replaces the legacy 810-LOC ``chunk_enrichment.py``
god file (split during port per plan §3673, §3685, §3706).

Pipeline:

  Stage A  ``_window.window_chunks``   — token-budget rule-based split
  Stage B  ``_glue.glue_by_similarity`` — embedding-driven merge (skipped
                                          when ``embedder=None``)
  Stage C  ``_enrich.enrich_with_llm`` — LLM enrichment with cache
                                          (skipped when
                                          ``llm_enrichment=False`` or
                                          ``llm_gateway=None``)

When Stage C is skipped the orchestrator still returns ``EnrichedChunk``
rows (rule-based fallback per plan §3686) so downstream callers see one
return type regardless of pipeline configuration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import UUID

from abridgeai.ai.chunking._enrich import (
    LLMGatewayProto,
    enrich_with_llm,
    promote_rule_based,
)
from abridgeai.ai.chunking._glue import Embedder, glue_by_similarity
from abridgeai.ai.chunking._window import window_chunks
from abridgeai.ai.chunking.base import EnrichedChunk

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from abridgeai.ai.chunking.cache import ChunkingCache
    from abridgeai.ai.extraction import ExtractedContent


class SemanticChunker:
    """3-stage chunker: rule-based windowing → glue → LLM enrichment."""

    def __init__(
        self,
        *,
        max_tokens: int = 800,
        overlap_tokens: int = 80,
        glue_threshold: float = 0.72,
        max_window_tokens: int = 2000,
        min_window_tokens: int = 30,
        parallelism: int = 4,
    ) -> None:
        self._max_tokens = max_tokens
        self._overlap_tokens = overlap_tokens
        self._glue_threshold = glue_threshold
        self._max_window_tokens = max_window_tokens
        self._min_window_tokens = min_window_tokens
        self._parallelism = parallelism

    async def chunk(
        self,
        content: ExtractedContent,
        *,
        embedder: Embedder | None = None,
        llm_gateway: LLMGatewayProto | None = None,
        db: AsyncSession | None = None,
        cache: ChunkingCache | None = None,
        llm_enrichment: bool = True,
        document_title: str | None = None,
        pipeline_run_id: UUID | None = None,
        parent_job_id: UUID | None = None,
        section_context: str = "",
        session_factory: Any = None,  # noqa: ANN401 -- optional sessionmaker for parallel enrichment
        **opts: Any,  # noqa: ANN401 -- forwarded chunker kwargs
    ) -> list[EnrichedChunk]:
        rule_chunks = window_chunks(
            content,
            max_tokens=int(opts.get("max_tokens", self._max_tokens)),
            overlap_tokens=int(opts.get("overlap_tokens", self._overlap_tokens)),
            section_context=section_context,
        )
        if not rule_chunks:
            return []

        windows = await glue_by_similarity(
            rule_chunks,
            embedder,
            threshold=float(opts.get("glue_threshold", self._glue_threshold)),
            max_window_tokens=int(opts.get("max_window_tokens", self._max_window_tokens)),
            min_window_tokens=int(opts.get("min_window_tokens", self._min_window_tokens)),
        )

        if llm_enrichment and llm_gateway is not None and db is not None:
            return await enrich_with_llm(
                windows,
                llm_gateway=llm_gateway,
                db=db,
                cache=cache,
                document_title=document_title,
                pipeline_run_id=pipeline_run_id,
                parent_job_id=parent_job_id,
                parallelism=int(opts.get("parallelism", self._parallelism)),
                session_factory=session_factory,
            )

        return [promote_rule_based(w) for w in windows]


__all__ = ["SemanticChunker"]
