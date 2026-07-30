"""Phase 4: manual-grading columns on quiz_attempt_answers (needs-grading queue + mark/feedback).

Revision ID: 0048_quiz_ans_manual_grade
Revises: 0047_quiz_rich_formats
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
revision = "0048_quiz_ans_manual_grade"
down_revision = "0047_quiz_rich_formats"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("quiz_attempt_answers", sa.Column("needs_manual_grade", sa.Boolean(), nullable=False, server_default=sa.text("false")))
    op.add_column("quiz_attempt_answers", sa.Column("manual_score", sa.Numeric(8, 2), nullable=True))
    op.add_column("quiz_attempt_answers", sa.Column("manual_feedback", sa.Text(), nullable=True))
    op.add_column("quiz_attempt_answers", sa.Column("graded_by", PGUUID(as_uuid=True), nullable=True))
    op.add_column("quiz_attempt_answers", sa.Column("graded_at", sa.DateTime(timezone=True), nullable=True))
    op.create_foreign_key("fk_quiz_attempt_answers_graded_by", "quiz_attempt_answers", "users", ["graded_by"], ["id"], ondelete="SET NULL")
    op.create_index("ix_quiz_attempt_answers_needs_manual_grade", "quiz_attempt_answers", ["needs_manual_grade"], postgresql_where=sa.text("needs_manual_grade = true"))


def downgrade() -> None:
    op.drop_index("ix_quiz_attempt_answers_needs_manual_grade", table_name="quiz_attempt_answers")
    op.drop_constraint("fk_quiz_attempt_answers_graded_by", "quiz_attempt_answers", type_="foreignkey")
    op.drop_column("quiz_attempt_answers", "graded_at")
    op.drop_column("quiz_attempt_answers", "graded_by")
    op.drop_column("quiz_attempt_answers", "manual_feedback")
    op.drop_column("quiz_attempt_answers", "manual_score")
    op.drop_column("quiz_attempt_answers", "needs_manual_grade")
