"""Phase 10: cached per-question/whole-quiz statistics (facility, discrimination).

Revision ID: 0054_quiz_stats_cache
Revises: 0053_quiz_grade_items
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
revision = "0054_quiz_stats_cache"
down_revision = "0053_quiz_grade_items"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "quiz_statistics_cache",
        sa.Column("id", PGUUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("quiz_id", PGUUID(as_uuid=True), nullable=False),
        sa.Column("question_id", PGUUID(as_uuid=True), nullable=True),
        sa.Column("facility", sa.Numeric(6, 4), nullable=True),
        sa.Column("discrimination", sa.Numeric(6, 4), nullable=True),
        sa.Column("sample_size", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("computed_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["quiz_id"], ["quizzes.id"], ondelete="NO ACTION"),
        sa.ForeignKeyConstraint(["question_id"], ["quiz_questions.id"], ondelete="NO ACTION"),
    )
    op.create_index("ix_quiz_statistics_cache_quiz_id", "quiz_statistics_cache", ["quiz_id"])
    op.create_index("uq_quiz_statistics_cache_wholequiz", "quiz_statistics_cache", ["quiz_id"], unique=True, postgresql_where=sa.text("question_id IS NULL"))
    op.create_index("uq_quiz_statistics_cache_question", "quiz_statistics_cache", ["quiz_id", "question_id"], unique=True, postgresql_where=sa.text("question_id IS NOT NULL"))


def downgrade() -> None:
    op.drop_index("uq_quiz_statistics_cache_question", table_name="quiz_statistics_cache")
    op.drop_index("uq_quiz_statistics_cache_wholequiz", table_name="quiz_statistics_cache")
    op.drop_index("ix_quiz_statistics_cache_quiz_id", table_name="quiz_statistics_cache")
    op.drop_table("quiz_statistics_cache")
