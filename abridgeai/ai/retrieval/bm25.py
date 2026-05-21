"""Postgres tsvector lexical search over ``document_chunks`` (Phase 3).

Companion to :mod:`abridgeai.ai.retrieval.pgvector`. Where pgvector
covers semantic similarity, this module covers exact-term matching —
the BM25-flavoured half of Anthropic's Contextual Retrieval pattern.

We don't run ParadeDB / pg_search yet, so the score here is
``ts_rank_cd`` (TF-IDF-flavoured cover-density ranking) rather than
true BM25. RRF fusion downstream rank-normalizes both retrievers, so
the exact score formula matters less than candidate-set recall — and
``ts_rank_cd`` over a GIN index is plenty for that.

The ``content_tsv`` column (migration 0015) is GENERATED ALWAYS over
``content || section_title || context_sentence``, matching the
Anthropic cookbook's multi-field BM25 (``content`` +
``contextualized_content``).

Query parsing
-------------
We use ``plainto_tsquery('simple', q)`` rather than ``to_tsquery``: the
former tolerates arbitrary user input (no ``&``/``|`` escaping needed),
the latter requires a syntax-clean expression. The ``simple``
dictionary mirrors the column's generation expression so query terms
match index tokens consistently.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import bindparam, text
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.dialects.postgresql import UUID as PG_UUID

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


@dataclass(frozen=True)
class ChunkWithRank:
    """Single result row from :func:`bm25_search`.

    ``rank`` is the raw ``ts_rank_cd`` score (larger == more relevant).
    Callers using RRF fusion only need the *order* of results, not the
    absolute score values.
    """

    chunk_id: UUID
    material_version_id: UUID
    course_id: UUID | None
    lesson_id: UUID | None
    content: str
    rank: float


_SELECT_CLAUSE = (
    "SELECT id, material_version_id, course_id, lesson_id, content, "
    "       ts_rank_cd(content_tsv, query) AS rank "
    "FROM document_chunks, "
    "     plainto_tsquery('simple', :query_text) AS query "
)
_FILTERS = (
    "WHERE content_tsv @@ query "
    "  AND (CAST(:course_id AS uuid) IS NULL "
    "       OR course_id = CAST(:course_id AS uuid)) "
    "  AND (CAST(:lesson_ids AS uuid[]) IS NULL "
    "       OR lesson_id = ANY(CAST(:lesson_ids AS uuid[]))) "
)
_ORDER = "ORDER BY rank DESC LIMIT :top_k"
_QUERY = _SELECT_CLAUSE + _FILTERS + _ORDER  # noqa: S608  -- bind params, no concat


async def bm25_search(
    db: AsyncSession,
    query_text: str,
    *,
    course_id: UUID | None = None,
    lesson_ids: list[UUID] | None = None,
    top_k: int = 150,
) -> list[ChunkWithRank]:
    """Lexical search over ``document_chunks.content_tsv``.

    Parameters
    ----------
    db
        Active :class:`sqlalchemy.ext.asyncio.AsyncSession`.
    query_text
        Raw user query — passed through ``plainto_tsquery('simple', …)``
        so callers don't need to escape ``&``/``|``/``!``.
    course_id, lesson_ids
        Optional scoping filters (mirror :func:`vector_search`).
    top_k
        Recall cap. The Anthropic cookbook uses 150 for the BM25 leg of
        hybrid search; we default to the same.

    Returns
    -------
    list[ChunkWithRank]
        Rows ordered by descending ``ts_rank_cd``. Returns ``[]`` when
        ``query_text`` is empty/whitespace (Postgres errors on an empty
        ``plainto_tsquery`` otherwise — guard at the boundary).
    """
    if top_k <= 0:
        return []
    if not query_text or not query_text.strip():
        return []

    stmt = text(_QUERY).bindparams(
        bindparam("course_id", type_=PG_UUID(as_uuid=True)),
        bindparam("lesson_ids", type_=ARRAY(PG_UUID(as_uuid=True))),
    )

    params: dict[str, object] = {
        "query_text": query_text,
        "course_id": course_id,
        "lesson_ids": list(lesson_ids) if lesson_ids else None,
        "top_k": int(top_k),
    }

    result = await db.execute(stmt, params)
    rows = result.mappings().all()
    return [
        ChunkWithRank(
            chunk_id=row["id"],
            material_version_id=row["material_version_id"],
            course_id=row["course_id"],
            lesson_id=row["lesson_id"],
            content=row["content"],
            rank=float(row["rank"]),
        )
        for row in rows
    ]


__all__ = ["ChunkWithRank", "bm25_search"]
