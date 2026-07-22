"""Add grading_method (headline-score policy) to quizzes.

Revision ID: 0033_quiz_grading_method
Revises: 0032_quiz_schedule_window
Create Date: 2026-07-22 00:00:00.000000

Moodle-style per-quiz grading method deciding which attempt counts as a
student's headline score. Allowed values: 'highest', 'average', 'first',
'last'. Column is NOT NULL with server_default 'highest', so existing
quizzes are entirely unaffected — they keep the highest-attempt behaviour
until a teacher changes it.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

# NOTE: alembic_version.version_num is varchar(32); keep this id <= 32 chars.
revision = "0033_quiz_grading_method"
down_revision = "0032_quiz_schedule_window"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "quizzes",
        sa.Column(
            "grading_method",
            sa.String(16),
            nullable=False,
            server_default="highest",
        ),
    )
    op.create_check_constraint(
        "ck_quizzes_grading_method",
        "quizzes",
        "grading_method IN ('highest', 'average', 'first', 'last')",
    )


def downgrade() -> None:
    op.drop_constraint("ck_quizzes_grading_method", "quizzes", type_="check")
    op.drop_column("quizzes", "grading_method")
