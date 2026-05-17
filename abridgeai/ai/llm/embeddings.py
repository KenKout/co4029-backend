"""Embedding client.

Mirrors ``LLMGateway`` for ``POST /embeddings``:
  * one HTTP round-trip per call,
  * one ``ai_model_calls`` row per call (metadata-only ``request_payload``),
  * ``ConfigError`` at startup if the configured ``EMBEDDING_DIMENSIONS``
    does not match the actual ``document_chunks.embedding`` pgvector column
    width — see FR-13.

No ``mock`` branch and no silent zero-vector fallback (that was the bug in
the legacy ``embeddings.py`` module which this replaces).
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.ai.llm.audit import write_ai_model_call
from abridgeai.ai.llm.client import OpenAICompatibleClient
from abridgeai.ai.llm.errors import ConfigError, ProviderError, ResponseFormatError
from abridgeai.ai.llm.pricing import compute_cost
from abridgeai.ai.llm.roles import LLMRole, binding_for
from abridgeai.core.config import Settings, get_settings
from abridgeai.core.exceptions import AppError


class EmbeddingClient:
    """One client, one model, one HTTP path."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    async def validate_dimensions(self, db: AsyncSession) -> None:
        """Assert ``EMBEDDING_DIMENSIONS`` matches the pgvector column width.

        Run once at startup (e.g. from a FastAPI lifespan hook). Reads
        ``atttypmod`` from ``pg_attribute`` for the
        ``document_chunks.embedding`` column. pgvector encodes the vector
        dimension in ``atttypmod`` as ``dim + 1`` (PG type modifier
        convention).

        Raises:
            ConfigError: when the configured dimension doesn't match the
                column. The message names migration ``0006_widen_embedding_vector``
                so operators know how to widen the column intentionally.
        """
        result = await db.execute(
            text(
                """
                SELECT atttypmod
                  FROM pg_attribute
                 WHERE attrelid = 'document_chunks'::regclass
                   AND attname  = 'embedding'
                """
            )
        )
        atttypmod_row = result.first()
        if atttypmod_row is None:
            raise ConfigError(
                "document_chunks.embedding column not found; "
                "run db/schema.sql or alembic upgrade head"
            )
        # pgvector stores the declared dimension directly in atttypmod
        # (verified empirically: vector(1536) yields atttypmod=1536).
        column_dim = atttypmod_row[0]
        configured_dim = self._settings.embedding_dimensions

        if configured_dim != column_dim:
            raise ConfigError(
                f"EMBEDDING_DIMENSIONS={configured_dim} but "
                f"document_chunks.embedding is vector({column_dim}). "
                "Run migration 0006_widen_embedding_vector first, or revert "
                f"EMBEDDING_DIMENSIONS to {column_dim}."
            )

    async def embed(
        self,
        texts: list[str],
        *,
        db: AsyncSession,
        pipeline_run_id: UUID | None = None,
        parent_job_id: UUID | None = None,
        parent_run_id: UUID | None = None,
    ) -> list[list[float]]:
        """Embed ``texts``; return one float vector per input, in order.

        Writes one ``ai_model_calls`` row per invocation (metadata-only
        ``request_payload``). Raises ``AppError`` on any upstream failure
        after writing a ``status='failed'`` audit row.
        """
        binding = binding_for(LLMRole.EMBEDDING, self._settings)
        client = OpenAICompatibleClient(binding)

        request_meta: dict[str, Any] = {
            "model": binding.model,
            "input_count": len(texts),
            "total_input_chars": sum(len(t) for t in texts),
            "dimensions": self._settings.embedding_dimensions,
        }

        try:
            response_body, latency_ms = await client.embeddings(
                texts, dimensions=self._settings.embedding_dimensions
            )
        except (ProviderError, ResponseFormatError) as exc:
            await write_ai_model_call(
                db,
                role=LLMRole.EMBEDDING,
                tier=None,
                operation="embedding",
                model_name=binding.model,
                base_url=binding.base_url,
                stage_name="embedding",
                pipeline_run_id=pipeline_run_id,
                parent_run_id=parent_run_id,
                parent_job_id=parent_job_id,
                request_payload=request_meta,
                response_payload=None,
                input_tokens=None,
                output_tokens=None,
                cached_input_tokens=None,
                latency_ms=0,
                status="failed",
                error_message=str(exc),
                estimated_cost_usd=None,
            )
            raise AppError(str(exc)) from exc

        usage = response_body.get("usage") or {}
        input_tokens: int | None = usage.get("prompt_tokens")
        cost = compute_cost(binding.model, input_tokens, 0)

        try:
            data = response_body["data"]
        except KeyError as exc:
            await write_ai_model_call(
                db,
                role=LLMRole.EMBEDDING,
                tier=None,
                operation="embedding",
                model_name=binding.model,
                base_url=binding.base_url,
                stage_name="embedding",
                pipeline_run_id=pipeline_run_id,
                parent_run_id=parent_run_id,
                parent_job_id=parent_job_id,
                request_payload=request_meta,
                response_payload=None,
                input_tokens=input_tokens,
                output_tokens=0,
                cached_input_tokens=None,
                latency_ms=latency_ms,
                status="failed",
                error_message="embedding response missing 'data' key",
                estimated_cost_usd=cost,
            )
            raise AppError("embedding response missing 'data' key") from exc

        await write_ai_model_call(
            db,
            role=LLMRole.EMBEDDING,
            tier=None,
            operation="embedding",
            model_name=binding.model,
            base_url=binding.base_url,
            stage_name="embedding",
            pipeline_run_id=pipeline_run_id,
            parent_run_id=parent_run_id,
            parent_job_id=parent_job_id,
            request_payload=request_meta,
            response_payload=None,
            input_tokens=input_tokens,
            output_tokens=0,
            cached_input_tokens=None,
            latency_ms=latency_ms,
            status="success",
            error_message=None,
            estimated_cost_usd=cost,
        )

        items = sorted(data, key=lambda x: x["index"])
        return [item["embedding"] for item in items]

    async def embed_query(
        self,
        text_query: str,
        *,
        db: AsyncSession,
        pipeline_run_id: UUID | None = None,
        parent_job_id: UUID | None = None,
        parent_run_id: UUID | None = None,
    ) -> list[float]:
        """Convenience wrapper for single-text retrieval queries."""
        results = await self.embed(
            [text_query],
            db=db,
            pipeline_run_id=pipeline_run_id,
            parent_job_id=parent_job_id,
            parent_run_id=parent_run_id,
        )
        return results[0]
