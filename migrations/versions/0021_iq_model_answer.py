"""Add model_answer column to interview_questions

Revision ID: 0021_interview_question_model_answer
Revises: 0020_ai_model_pricing
Create Date: 2026-07-13 00:00:00.000000

Teachers reviewing the AI-generated question bank had no reference
answer to check a question against — only the prompt text. This adds
a nullable ``model_answer`` column so the generation pipeline can
populate a teacher-facing reference answer per question. It is
strictly an authoring aid: never exposed to learners (the public /
learner-facing schemas do not project it) and never used to
auto-grade — interview scoring stays rubric-based against
``InterviewOutcome`` criteria, unchanged.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0021_iq_model_answer"
down_revision = "0020_ai_model_pricing"
branch_labels = None
depends_on = None

_TABLE = "interview_questions"
_COLUMN = "model_answer"


def upgrade() -> None:
    op.add_column(_TABLE, sa.Column(_COLUMN, sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column(_TABLE, _COLUMN)
