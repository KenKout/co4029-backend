"""Freeze effective per-student quiz timing on attempts.

Revision ID: 0137_quiz_attempt_timing_snapshot
Revises: 0136_quiz_expected_time_check
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0137_quiz_timing_snapshot"
down_revision = "0136_quiz_expected_time_check"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "quiz_attempts",
        sa.Column("timing_snapshot", postgresql.JSONB(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("quiz_attempts", "timing_snapshot")
