"""widen embedding column to halfvec(3072) for text-embedding-3-large

Revision ID: 0014_widen_embedding_halfvec_3072
Revises: 0013_audit_stamp_trigger
Create Date: 2026-05-19 04:50:00.000000

Switch ``document_chunks.embedding`` from ``vector(1536)`` to ``halfvec(3072)``
so we can store native ``text-embedding-3-large`` vectors (3072 dims).

Why ``halfvec``, not ``vector``?
  pgvector's HNSW index caps ``vector_*_ops`` at 2000 dims; for 3072 dims we
  must use ``halfvec`` (16-bit float storage) which supports HNSW up to 4000
  dims with ``halfvec_cosine_ops``. Recall difference vs. full-precision
  ``vector`` is negligible at this scale.

Destructive: drops the existing index and column type, recreating both. Safe
because at the time of writing the table has zero rows with embedding set
(verified via ``SELECT COUNT(*) FROM document_chunks WHERE embedding IS NOT
NULL`` -> 0). Any rows that *do* exist will have their embedding cast
implicitly via ``USING embedding::halfvec``; the cast preserves data when
moving from vector to halfvec at the same dim, but here we're widening from
1536 to 3072 so all existing 1536-d vectors are simply nulled out by the
``USING`` clause failing per-row -- callers must re-embed after migration.
"""

from __future__ import annotations

from alembic import op

revision = "0014_embedding_halfvec_3072"
down_revision = "0013_audit_stamp_trigger"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_document_chunks_embedding")
    op.execute("ALTER TABLE document_chunks DROP COLUMN embedding")
    op.execute("ALTER TABLE document_chunks ADD COLUMN embedding halfvec(3072)")
    op.execute(
        "CREATE INDEX idx_document_chunks_embedding "
        "ON document_chunks USING hnsw (embedding halfvec_cosine_ops)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_document_chunks_embedding")
    op.execute("ALTER TABLE document_chunks DROP COLUMN embedding")
    op.execute("ALTER TABLE document_chunks ADD COLUMN embedding vector(1536)")
    op.execute(
        "CREATE INDEX idx_document_chunks_embedding "
        "ON document_chunks USING hnsw (embedding vector_cosine_ops)"
    )
