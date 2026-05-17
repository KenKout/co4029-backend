"""Persistent cache for Stage-C LLM enrichment.

Backed by the ``chunking_enrichment_cache`` table from the baseline schema
(see ``migrations/versions/0001_baseline_schema.py``). Keyed by
``(content_hash, prompt_version)`` per Reconciliation §B9 and §C12: bumping
``PROMPT_VERSION`` in ``_enrich`` invalidates old rows without touching
the table.

Inserts use ``ON CONFLICT (content_hash, prompt_version) DO NOTHING`` so
two parallel pipelines computing the same content do not race on the
unique constraint.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sqlalchemy import text

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

_GET_SQL = text(
    """
    SELECT output_json, model_name, input_tokens, output_tokens
    FROM chunking_enrichment_cache
    WHERE content_hash = :content_hash AND prompt_version = :prompt_version
    LIMIT 1
    """
)

_PUT_SQL = text(
    """
    INSERT INTO chunking_enrichment_cache
        (content_hash, prompt_version, model_name, output_json, input_tokens, output_tokens)
    VALUES (:content_hash, :prompt_version, :model_name, CAST(:output_json AS JSONB),
            :input_tokens, :output_tokens)
    ON CONFLICT (content_hash, prompt_version) DO NOTHING
    """
)


class ChunkingCache:
    """Async wrapper around ``chunking_enrichment_cache``."""

    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def get(
        self,
        content_hash: str,
        prompt_version: str,
    ) -> dict[str, Any] | None:
        result = await self._db.execute(
            _GET_SQL,
            {"content_hash": content_hash, "prompt_version": prompt_version},
        )
        row = result.first()
        if row is None:
            return None
        return {
            "output_json": row.output_json,
            "model_name": row.model_name,
            "input_tokens": row.input_tokens,
            "output_tokens": row.output_tokens,
        }

    async def put(
        self,
        content_hash: str,
        prompt_version: str,
        *,
        output_json: dict[str, Any],
        model_name: str | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
    ) -> None:
        import json as _json

        await self._db.execute(
            _PUT_SQL,
            {
                "content_hash": content_hash,
                "prompt_version": prompt_version,
                "model_name": model_name,
                "output_json": _json.dumps(output_json),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            },
        )


__all__ = ["ChunkingCache"]
