"""Phase 8: overall grade-band feedback table + per-option feedback columns.

Revision ID: 0052_quiz_feedback_bands
Revises: 0051_quiz_qtype_expansion
Create Date: 2026-07-24

Additive/reversible DDL only (new tables + nullable/server-defaulted columns) so
the running app — whose ORM does not yet map these — keeps working.
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PGUUID

# NOTE: alembic_version.version_num is varchar(32); keep revision id <= 32 chars.
revision = "0052_quiz_feedback_bands"
down_revision = "0051_quiz_qtype_expansion"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "quiz_feedback",
        sa.Column("id", PGUUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("quiz_id", PGUUID(as_uuid=True), nullable=False),
        sa.Column("min_grade", sa.Numeric(5, 2), nullable=False),
        sa.Column("max_grade", sa.Numeric(5, 2), nullable=False),
        sa.Column("feedback_text", sa.Text(), nullable=False),
        sa.Column("feedback_format", sa.String(16), nullable=False, server_default="markdown"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.CheckConstraint("min_grade >= 0 AND min_grade <= 100", name="ck_quiz_feedback_min_range"),
        sa.CheckConstraint("max_grade >= 0 AND max_grade <= 100", name="ck_quiz_feedback_max_range"),
        sa.CheckConstraint("min_grade < max_grade", name="ck_quiz_feedback_min_lt_max"),
        sa.ForeignKeyConstraint(["quiz_id"], ["quizzes.id"], ondelete="CASCADE"),
    )
    op.create_index("ix_quiz_feedback_quiz_id", "quiz_feedback", ["quiz_id"])
    op.add_column("quiz_question_options", sa.Column("feedback_text", sa.Text(), nullable=True))
    op.add_column("quiz_question_options", sa.Column("feedback_format", sa.String(16), nullable=True))


def downgrade() -> None:
    op.drop_column("quiz_question_options", "feedback_format")
    op.drop_column("quiz_question_options", "feedback_text")
    op.drop_index("ix_quiz_feedback_quiz_id", table_name="quiz_feedback")
    op.drop_table("quiz_feedback")
