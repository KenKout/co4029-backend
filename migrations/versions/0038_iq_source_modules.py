"""Add source_module_ids to interview_questions (module attribution).

Revision ID: 0038_iq_src_modules
Revises: 0037_iq_bank_id_def
Create Date: 2026-07-23 00:00:00.000000

Per-question module attribution (§QBank-group). Each interview question
records which module(s) its generation drew from, so the teacher question
bank can group questions by module and surface a distinct "multiple
modules" section. Stored as a JSONB array of module UUIDs (as text).

Backfill: existing rows are seeded with their interview config's owning
module (config.module_id) — a single-element array — matching the prior
"the interview's own module" semantics. New rows are populated by the
generation pipeline from the run's ``source_module_ids``.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# NOTE: alembic_version.version_num is varchar(32); keep this id <= 32 chars.
revision = "0038_iq_src_modules"
down_revision = "0037_iq_bank_id_def"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "interview_questions",
        sa.Column(
            "source_module_ids",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
    )
    # Backfill: each question inherits its config's owning module as a
    # single-element JSONB array (["<module_id>"]). jsonb_build_array on the
    # text-cast UUID keeps the same shape the pipeline writes going forward.
    op.execute(
        """
        UPDATE interview_questions AS q
        SET source_module_ids = jsonb_build_array(c.module_id::text)
        FROM interview_configs AS c
        WHERE q.interview_config_id = c.id
          AND c.module_id IS NOT NULL
        """
    )


def downgrade() -> None:
    op.drop_column("interview_questions", "source_module_ids")
