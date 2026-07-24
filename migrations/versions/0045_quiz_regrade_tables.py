"""Phase 1 (regrade): dry-run/commit staging + audit tables.

Revision ID: 0045_quiz_regrade_tables
Revises: 0044_quiz_ans_graded_rev
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
revision = "0045_quiz_regrade_tables"
down_revision = "0044_quiz_ans_graded_rev"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "quiz_regrade_runs",
        sa.Column("id", PGUUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("quiz_id", PGUUID(as_uuid=True), nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default=sa.text("'dry_run'")),
        sa.Column("scope_json", JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("attempts_affected", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("answers_changed", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("requested_by", PGUUID(as_uuid=True), nullable=True),
        sa.Column("committed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.CheckConstraint("status IN ('dry_run','committed','cancelled')", name="ck_quiz_regrade_runs_status"),
        sa.ForeignKeyConstraint(["quiz_id"], ["quizzes.id"], ondelete="NO ACTION"),
        sa.ForeignKeyConstraint(["requested_by"], ["users.id"], ondelete="SET NULL"),
    )
    op.create_index("ix_quiz_regrade_runs_quiz_id", "quiz_regrade_runs", ["quiz_id"])
    op.create_table(
        "quiz_regrade_items",
        sa.Column("id", PGUUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("run_id", PGUUID(as_uuid=True), nullable=False),
        sa.Column("attempt_id", PGUUID(as_uuid=True), nullable=False),
        sa.Column("answer_id", PGUUID(as_uuid=True), nullable=False),
        sa.Column("question_id", PGUUID(as_uuid=True), nullable=False),
        sa.Column("old_is_correct", sa.Boolean(), nullable=False),
        sa.Column("new_is_correct", sa.Boolean(), nullable=False),
        sa.Column("old_points", sa.Numeric(8, 2), nullable=False),
        sa.Column("new_points", sa.Numeric(8, 2), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["run_id"], ["quiz_regrade_runs.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["attempt_id"], ["quiz_attempts.id"], ondelete="NO ACTION"),
        sa.ForeignKeyConstraint(["answer_id"], ["quiz_attempt_answers.id"], ondelete="NO ACTION"),
        sa.ForeignKeyConstraint(["question_id"], ["quiz_questions.id"], ondelete="NO ACTION"),
    )
    op.create_index("ix_quiz_regrade_items_run_id", "quiz_regrade_items", ["run_id"])


def downgrade() -> None:
    op.drop_index("ix_quiz_regrade_items_run_id", table_name="quiz_regrade_items")
    op.drop_table("quiz_regrade_items")
    op.drop_index("ix_quiz_regrade_runs_quiz_id", table_name="quiz_regrade_runs")
    op.drop_table("quiz_regrade_runs")
