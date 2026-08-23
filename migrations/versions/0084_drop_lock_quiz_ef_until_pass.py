"""Drop the unused interview_configs.lock_quiz_ef_until_pass column.

The FR-5.3 quiz-progression lock was never reachable: no UI control sets
the flag, so it stayed at its shipped default FALSE and the
``_ensure_interview_pass_lock`` gate in quizzes/services/taking.py never
fired. The feature is being removed (backend + frontend + report).

Revision ID: 0084_drop_lock_quiz_ef_until_pass
Revises: 0083_learning_programs
Create Date: 2026-08-23
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0084_drop_lock_quiz_ef"
down_revision = "0083_learning_programs"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_column("interview_configs", "lock_quiz_ef_until_pass")


def downgrade() -> None:
    op.add_column(
        "interview_configs",
        sa.Column(
            "lock_quiz_ef_until_pass",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("FALSE"),
        ),
    )
