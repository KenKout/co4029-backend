"""Add interview_questions.embedding for question-bank duplicate detection

Revision ID: 0063_iq_embedding
Revises: 0062_lesson_kg
Create Date: 2026-07-26

Adds a single nullable ``halfvec(3072)`` column so an authored interview question
can be compared against the existing bank by meaning rather than by string
equality. Feeds the two-stage duplicate check (vector shortlist -> LLM verdict);
the vector half only narrows candidates, it never decides on its own.

Why halfvec(3072)
-----------------
Mirrors ``document_chunks.embedding`` exactly. The same ``EmbeddingClient`` and
``EMBEDDING_DIMENSIONS`` setting produce both, and ``EmbeddingClient`` asserts at
startup that the configured dimension matches the ``document_chunks`` column — so
a divergent width here would be a silent trap the existing guard cannot see.

Nullable on purpose
-------------------
Existing rows keep ``NULL``. A question with no embedding is simply skipped by the
shortlist query (``WHERE embedding IS NOT NULL``), which degrades duplicate
detection to "no candidates found" rather than failing an author's save. Backfill
is optional and can happen lazily as questions are edited.

No index
--------
Deliberately none. The shortlist is always scoped to one config
(``interview_config_id = :id``), and a bank is tens of questions — an exact scan
over that slice beats maintaining an ANN index, and an ivfflat/hnsw index on a
tiny set can actually *lose* recall. Add one only if a bank ever grows past a few
thousand questions.

Additive and fully reversible.
"""

from __future__ import annotations

from alembic import op

# NOTE: alembic_version.version_num is varchar(32); keep revision id <= 32 chars.
revision = "0063_iq_embedding"
down_revision = "0062_lesson_kg"
branch_labels = None
depends_on = None

_TABLE = "interview_questions"
_COLUMN = "embedding"


def upgrade() -> None:
    # pgvector is already installed (document_chunks.embedding depends on it);
    # IF NOT EXISTS keeps this migration self-contained for a database built
    # from migrations alone.
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    # Raw DDL, matching 0014: halfvec has no first-class alembic type helper and
    # the existing embedding column was added this way.
    op.execute(f"ALTER TABLE {_TABLE} ADD COLUMN {_COLUMN} halfvec(3072)")


def downgrade() -> None:
    op.execute(f"ALTER TABLE {_TABLE} DROP COLUMN {_COLUMN}")
