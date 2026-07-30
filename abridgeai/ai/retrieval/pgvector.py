"""pgvector cosine-distance vector search over ``document_chunks``.

This is the consumer-side primitive used by quiz / interview retrieval
stages (Phase 5/6). Storage of chunks lives in the materials feature
(Phase 4); we only read here.

Implementation note (temporary): the ``DocumentChunk`` ORM model is owned
by ``abridgeai.features.materials`` and does not exist yet — Phase 4
lands it. Until then this helper issues raw SQL via ``text()``.

TODO Phase 4: replace the raw SQL below with ::

    distance = DocumentChunk.embedding.cosine_distance(embedding)
    stmt = (
        select(DocumentChunk, distance.label("distance"))
        .order_by(distance)
        .limit(top_k)
    )

once ``features.materials.models.DocumentChunk`` is importable.

Cosine-distance semantics
-------------------------
pgvector's ``<=>`` operator is **cosine distance** in the range
``[0.0, 2.0]``. Smaller is more similar (``0.0`` == identical direction;
``1.0`` == orthogonal; ``2.0`` == opposite). Callers wanting a similarity
score should compute ``similarity = 1.0 - distance``.

Soft-delete behaviour
---------------------
``document_chunks`` does **not** have a ``deleted_at`` column in the
baseline schema (T0.9 §608-623; the table is omitted from
``SOFT_DELETE_TABLES``). Chunks are deleted via ``ON DELETE CASCADE``
when their parent ``courses`` / ``modules`` / ``lessons`` /
``learning_material_versions`` rows are hard-deleted. Therefore the raw
SQL in this module does **not** include a ``WHERE deleted_at IS NULL``
filter. If Phase 4 adds soft-deletion to ``document_chunks`` directly,
this filter must be added back here AND in the eventual ORM port.
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
class ChunkWithDistance:
    """Single result row from :func:`vector_search`.

    ``distance`` is the raw pgvector cosine distance (``0.0`` == identical
    direction, larger == less similar). Callers running MMR
    (:func:`abridgeai.ai.retrieval.mmr.mmr_diversify`) need ``embedding``
    to be populated; pass ``include_embeddings=True`` to
    :func:`vector_search` to opt in (default skips the column to save
    bandwidth). ``metadata`` carries the JSONB ``document_chunks.metadata``
    column verbatim — downstream role-aware filters
    (:mod:`abridgeai.ai.retrieval.role_filter`) read
    ``metadata['semantic']['content_role']``, falling back to the
    rule-based top-level ``metadata['content_role']``, to deprioritize
    summary / review / front_matter chunks during quiz generation. The
    whole JSONB blob is selected rather than the one key precisely so
    that precedence can be resolved here.
    """

    chunk_id: UUID
    material_version_id: UUID
    course_id: UUID | None
    lesson_id: UUID | None
    content: str
    distance: float
    embedding: list[float] | None = None
    metadata: dict[str, object] | None = None


_SELECT_COLS = (
    "id, material_version_id, course_id, lesson_id, content, metadata, "
    "(embedding <=> CAST(:query_embedding AS halfvec)) AS distance"
)
_SELECT_COLS_WITH_EMB = _SELECT_COLS + ", embedding"


def _build_query(*, include_embeddings: bool) -> str:
    cols = _SELECT_COLS_WITH_EMB if include_embeddings else _SELECT_COLS
    select_clause = f"SELECT {cols} FROM document_chunks "  # noqa: S608
    course_filter = (
        "  AND (CAST(:course_id AS uuid) IS NULL        OR course_id = CAST(:course_id AS uuid)) "
    )
    lesson_filter = (
        "  AND (CAST(:lesson_ids AS uuid[]) IS NULL "
        "       OR lesson_id = ANY(CAST(:lesson_ids AS uuid[]))) "
    )
    # ``ai/preprocessing`` marks pages that can never carry an assessable fact
    # (bare title pages, "Thank you / Questions?" slides, bibliographies).
    # Filtering here rather than deleting at ingest keeps the decision
    # reversible: flipping a threshold is a config change, not a re-index.
    # COALESCE so the ~all rows written before preprocessing existed, which
    # have no such key, keep matching.
    excluded_filter = (
        "  AND COALESCE((metadata->>'retrieval_excluded')::boolean, false) = false "
    )
    return (
        select_clause
        + "WHERE embedding IS NOT NULL "
        + excluded_filter
        + course_filter
        + lesson_filter
        + "ORDER BY embedding <=> CAST(:query_embedding AS halfvec) "
        "LIMIT :top_k"
    )


def _format_vector(embedding: list[float]) -> str:
    """Serialize a Python list to pgvector's text input format ``[a,b,c]``."""

    return "[" + ",".join(repr(float(v)) for v in embedding) + "]"


def _parse_vector(raw: object) -> list[float] | None:
    """Parse a pgvector value coming back from psycopg.

    The pgvector adapter (when registered) returns ``list[float]``
    directly; without the adapter (the case for raw ``text()`` queries)
    it returns the textual ``"[1.0,2.0,...]"`` representation. Handle
    both shapes defensively.
    """

    if raw is None:
        return None
    if isinstance(raw, list):
        return [float(v) for v in raw]
    if isinstance(raw, str):
        stripped = raw.strip()
        if not stripped or stripped in ("[]",):
            return []
        if stripped.startswith("[") and stripped.endswith("]"):
            stripped = stripped[1:-1]
        return [float(v) for v in stripped.split(",") if v.strip()]
    return None


async def vector_search(
    db: AsyncSession,
    embedding: list[float],
    *,
    course_id: UUID | None = None,
    lesson_ids: list[UUID] | None = None,
    top_k: int = 10,
    include_embeddings: bool = False,
) -> list[ChunkWithDistance]:
    """Cosine-distance vector search over ``document_chunks``.

    Parameters
    ----------
    db
        Active :class:`sqlalchemy.ext.asyncio.AsyncSession`.
    embedding
        Query vector; must match the column dimension (3072 in the
        baseline schema, see ``embedding halfvec(3072)``).
    course_id
        Optional course scope. ``None`` means no course filter.
    lesson_ids
        Optional lesson scope; chunks not in this list are excluded.
        ``None`` means no lesson filter.
    top_k
        Number of rows to return. Use a generous number (e.g.
        ``limit * 3``) when the caller plans to run MMR after.
    include_embeddings
        When ``True``, populate :attr:`ChunkWithDistance.embedding`.
        Required for downstream MMR; off by default to save bandwidth.

    Returns
    -------
    list[ChunkWithDistance]
        Rows ordered by ascending distance (most-similar first).
    """

    if top_k <= 0:
        return []

    stmt = text(_build_query(include_embeddings=include_embeddings)).bindparams(
        bindparam("course_id", type_=PG_UUID(as_uuid=True)),
        bindparam("lesson_ids", type_=ARRAY(PG_UUID(as_uuid=True))),
    )

    params: dict[str, object] = {
        "query_embedding": _format_vector(embedding),
        "course_id": course_id,
        "lesson_ids": list(lesson_ids) if lesson_ids else None,
        "top_k": int(top_k),
    }

    result = await db.execute(stmt, params)
    rows = result.mappings().all()
    return [
        ChunkWithDistance(
            chunk_id=row["id"],
            material_version_id=row["material_version_id"],
            course_id=row["course_id"],
            lesson_id=row["lesson_id"],
            content=row["content"],
            distance=float(row["distance"]),
            embedding=_parse_vector(row["embedding"]) if include_embeddings else None,
            metadata=row.get("metadata") if isinstance(row.get("metadata"), dict) else None,
        )
        for row in rows
    ]


__all__ = ["ChunkWithDistance", "vector_search"]
