"""Add nullable learning_outcome_id FK to quiz_questions.

Revision ID: 0034_quiz_q_lo
Revises: 0033_quiz_grading_method
Create Date: 2026-07-22 00:00:00.000000

Links a quiz question to the course learning outcome it assesses (§LO-3).
Nullable + ON DELETE SET NULL, meaning "no outcome": existing questions
are unaffected (all NULL), and deleting an outcome doesn't cascade-delete
its questions — they simply revert to unassigned. The teacher-facing
``(L.O.x)`` prefix is derived from the outcome's live position at display
time, so nothing about the code is stored here.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

# NOTE: alembic_version.version_num is varchar(32); keep this id <= 32 chars.
revision = "0034_quiz_q_lo"
down_revision = "0033_quiz_grading_method"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "quiz_questions",
        sa.Column("learning_outcome_id", sa.UUID(), nullable=True),
    )
    op.create_foreign_key(
        "fk_quiz_questions_learning_outcome",
        "quiz_questions",
        "course_learning_outcomes",
        ["learning_outcome_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_index(
        "ix_quiz_questions_learning_outcome_id",
        "quiz_questions",
        ["learning_outcome_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_quiz_questions_learning_outcome_id", table_name="quiz_questions")
    op.drop_constraint(
        "fk_quiz_questions_learning_outcome", "quiz_questions", type_="foreignkey"
    )
    op.drop_column("quiz_questions", "learning_outcome_id")
