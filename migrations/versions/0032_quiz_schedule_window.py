"""Add scheduling window (available_from / available_until / due_at) to quizzes.

Revision ID: 0032_quiz_schedule_window
Revises: 0031_interview_pref_name
Create Date: 2026-07-22 00:00:00.000000

All three columns are nullable and default NULL, meaning "no restriction":
existing quizzes are entirely unaffected until a teacher sets a window.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

# NOTE: alembic_version.version_num is varchar(32); keep this id <= 32 chars.
revision = "0032_quiz_schedule_window"
down_revision = "0031_interview_pref_name"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "quizzes",
        sa.Column("available_from", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "quizzes",
        sa.Column("available_until", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "quizzes",
        sa.Column("due_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("quizzes", "due_at")
    op.drop_column("quizzes", "available_until")
    op.drop_column("quizzes", "available_from")
