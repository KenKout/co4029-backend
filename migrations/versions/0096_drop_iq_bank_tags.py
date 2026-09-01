"""Drop the unused tags column from the interview question bank.

Free-text bank tags were only ever writable by hand from the course-level
Question Bank editor: nothing generated them, no import path carried them, and
no report requirement referenced them. Every live row held an empty array, so
the column was pure surface area on a table that already has a real second
dimension (logical-question variant groups).

Revision ID: 0096_drop_iq_bank_tags
Revises: 0094_flat_faculties
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0096_drop_iq_bank_tags"
down_revision = "0094_flat_faculties"
branch_labels = None
depends_on = None


_TABLE = "interview_question_bank_items"


def upgrade() -> None:
    op.drop_column(_TABLE, "tags_json")


def downgrade() -> None:
    # Recreated NOT NULL with the same server_default the model carried, so a
    # downgrade lands rows in the original "no tags" state rather than NULL.
    op.add_column(
        _TABLE,
        sa.Column(
            "tags_json",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
    )
