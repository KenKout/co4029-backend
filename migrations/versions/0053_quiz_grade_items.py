"""Phase 9: materialised gradebook — quiz_grade_items + quiz_grades (grade of record).

Revision ID: 0053_quiz_grade_items
Revises: 0052_quiz_feedback_bands
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
revision = "0053_quiz_grade_items"
down_revision = "0052_quiz_feedback_bands"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "quiz_grade_items",
        sa.Column("id", PGUUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("quiz_id", PGUUID(as_uuid=True), sa.ForeignKey("quizzes.id"), nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("sort_order", sa.Integer, nullable=False, server_default=sa.text("0")),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
    )
    op.create_index("ix_quiz_grade_items_quiz_id", "quiz_grade_items", ["quiz_id"])
    op.create_table(
        "quiz_grades",
        sa.Column("id", PGUUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("quiz_id", PGUUID(as_uuid=True), sa.ForeignKey("quizzes.id"), nullable=False),
        sa.Column("student_id", PGUUID(as_uuid=True), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("grade_item_id", PGUUID(as_uuid=True), sa.ForeignKey("quiz_grade_items.id", ondelete="CASCADE"), nullable=True),
        sa.Column("grade_percent", sa.Numeric(5, 2), nullable=False),
        sa.Column("grade_points", sa.Numeric(8, 2), nullable=False, server_default=sa.text("0")),
        sa.Column("passed", sa.Boolean, nullable=False, server_default=sa.text("false")),
        sa.Column("grading_method", sa.String(16), nullable=False),
        sa.Column("based_on_attempt_id", PGUUID(as_uuid=True), sa.ForeignKey("quiz_attempts.id", ondelete="SET NULL"), nullable=True),
        sa.Column("attempts_counted", sa.Integer, nullable=False, server_default=sa.text("0")),
        sa.Column("computed_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
    )
    op.create_index("ix_quiz_grades_quiz_id", "quiz_grades", ["quiz_id"])
    op.create_index("ix_quiz_grades_student_id", "quiz_grades", ["student_id"])
    op.create_index("uq_quiz_grades_wholequiz", "quiz_grades", ["quiz_id", "student_id"], unique=True, postgresql_where=sa.text("grade_item_id IS NULL"))
    op.create_index("uq_quiz_grades_item", "quiz_grades", ["quiz_id", "student_id", "grade_item_id"], unique=True, postgresql_where=sa.text("grade_item_id IS NOT NULL"))


def downgrade() -> None:
    op.drop_index("uq_quiz_grades_item", table_name="quiz_grades")
    op.drop_index("uq_quiz_grades_wholequiz", table_name="quiz_grades")
    op.drop_index("ix_quiz_grades_student_id", table_name="quiz_grades")
    op.drop_index("ix_quiz_grades_quiz_id", table_name="quiz_grades")
    op.drop_table("quiz_grades")
    op.drop_index("ix_quiz_grade_items_quiz_id", table_name="quiz_grade_items")
    op.drop_table("quiz_grade_items")
