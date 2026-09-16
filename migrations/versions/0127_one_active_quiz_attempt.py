"""Enforce one active quiz attempt per student and quiz.

Revision ID: 0127_one_active_quiz_attempt
Revises: 0126_drop_formula_version
Create Date: 2026-09-16

The partial unique index applies only while an attempt is ``in_progress``;
historical submitted, graded, abandoned, and expired attempts remain allowed.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0127_one_active_quiz_attempt"
down_revision: str | None = "0126_drop_formula_version"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_INDEX_NAME = "uq_quiz_attempts_active"


def upgrade() -> None:
    duplicate_groups = op.get_bind().scalar(
        sa.text(
            "SELECT COUNT(*) FROM ("
            "SELECT quiz_id, student_id FROM quiz_attempts "
            "WHERE status = 'in_progress' "
            "GROUP BY quiz_id, student_id HAVING COUNT(*) > 1"
            ") AS duplicate_groups"
        )
    )
    if duplicate_groups:
        raise RuntimeError(
            f"{duplicate_groups} quiz/student group(s) have multiple in-progress "
            "attempts. Resolve the duplicate attempts before enabling the one-active-"
            "attempt invariant."
        )

    op.create_index(
        _INDEX_NAME,
        "quiz_attempts",
        ["quiz_id", "student_id"],
        unique=True,
        postgresql_where=sa.text("status = 'in_progress'"),
    )


def downgrade() -> None:
    op.drop_index(_INDEX_NAME, table_name="quiz_attempts")
