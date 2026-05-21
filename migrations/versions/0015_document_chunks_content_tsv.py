"""contextual BM25 — content_tsv generated column + GIN index

Revision ID: 0015_document_chunks_content_tsv
Revises: 0014_embedding_halfvec_3072
Create Date: 2026-05-21 13:30:00.000000

Phase 3 of the contextual-RAG upgrade (Anthropic guide). Adds a Postgres
``tsvector`` generated column over ``content`` PLUS the contextual
description Stage C writes to ``metadata.semantic.context_sentence`` /
``metadata.semantic.section_title``, then a GIN index for fast lexical
search. Matches the Anthropic cookbook pattern where BM25 indexes both
the chunk content AND the contextualized description.

Why ``simple`` dictionary?
  Project content mixes English and Vietnamese (UI strings, lecture
  slides). The ``english`` analyzer applies Porter stemming + English
  stopwords which mangles Vietnamese terms. ``simple`` does
  case-folding + tokenization but no stemming, which is the right
  default for a multilingual corpus. Operators can switch this to
  ``english`` later by re-running an ALTER + reindex if all content
  becomes English.

Why generated column over a trigger?
  PostgreSQL 12+ supports ``GENERATED ALWAYS AS (...) STORED`` for
  tsvector columns. The column is recomputed automatically on
  INSERT/UPDATE — no trigger maintenance, no application-side mirror.

Why ``ts_rank_cd``-shaped, not real BM25?
  Postgres' built-in full-text search uses TF-IDF-flavored ranking via
  ``ts_rank_cd`` rather than the BM25 scoring function. For our scale
  (hundreds of chunks per material) the difference is negligible — the
  RRF fusion downstream rank-normalizes both retrievers anyway, so the
  exact ranking-score formula matters less than the candidate-set
  recall, which tsvector + GIN handles fine. If we ever need true BM25
  scoring we can swap to the ``pg_search`` (ParadeDB) extension; the
  retriever interface stays the same.

Backfill behaviour
  GENERATED columns auto-populate on the next read for existing rows
  via Postgres' standard rules — no explicit UPDATE pass needed.
"""

from __future__ import annotations

from alembic import op

revision = "0015_document_chunks_content_tsv"
down_revision = "0014_embedding_halfvec_3072"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Generated tsvector over content + the LLM-enriched contextual
    # description. ``COALESCE`` guards rows that haven't been re-ingested
    # with Phase 2 contextual enrichment yet — they degrade to plain
    # content-only BM25, which is still useful.
    op.execute(
        """
        ALTER TABLE document_chunks
            ADD COLUMN content_tsv tsvector GENERATED ALWAYS AS (
                to_tsvector(
                    'simple',
                    coalesce(content, '') || ' ' ||
                    coalesce(metadata->'semantic'->>'section_title', '') || ' ' ||
                    coalesce(metadata->'semantic'->>'context_sentence', '')
                )
            ) STORED
        """
    )
    op.execute(
        "CREATE INDEX idx_document_chunks_content_tsv "
        "ON document_chunks USING GIN (content_tsv)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_document_chunks_content_tsv")
    op.execute("ALTER TABLE document_chunks DROP COLUMN IF EXISTS content_tsv")
