"""Phase 5: per-user / per-group quiz overrides (group_id intentionally FK-less).

Revision ID: 0049_quiz_overrides
Revises: 0048_quiz_ans_manual_grade
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
revision = "0049_quiz_overrides"
down_revision = "0048_quiz_ans_manual_grade"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "quiz_overrides",
        sa.Column("id", PGUUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("quiz_id", PGUUID(as_uuid=True), nullable=False),
        sa.Column("scope", sa.String(10), nullable=False),
        sa.Column("user_id", PGUUID(as_uuid=True), nullable=True),
        sa.Column("group_id", PGUUID(as_uuid=True), nullable=True),
        sa.Column("available_from", sa.DateTime(timezone=True), nullable=True),
        sa.Column("available_until", sa.DateTime(timezone=True), nullable=True),
        sa.Column("due_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("time_limit_seconds", sa.Integer(), nullable=True),
        sa.Column("max_attempts", sa.Integer(), nullable=True),
        sa.Column("allow_retakes", sa.Boolean(), nullable=True),
        sa.Column("cooldown_hours", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["quiz_id"], ["quizzes.id"], ondelete="NO ACTION"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="NO ACTION"),
        sa.CheckConstraint("scope IN ('user','group')", name="ck_quiz_overrides_scope"),
        sa.CheckConstraint(
            "(scope = 'user' AND user_id IS NOT NULL AND group_id IS NULL) OR "
            "(scope = 'group' AND group_id IS NOT NULL AND user_id IS NULL)",
            name="ck_quiz_overrides_scope_target",
        ),
        sa.UniqueConstraint("quiz_id", "scope", "user_id", name="uq_quiz_override_user"),
        sa.UniqueConstraint("quiz_id", "scope", "group_id", name="uq_quiz_override_group"),
    )
    op.create_index("ix_quiz_overrides_quiz_id", "quiz_overrides", ["quiz_id"])


def downgrade() -> None:
    op.drop_index("ix_quiz_overrides_quiz_id", table_name="quiz_overrides")
    op.drop_table("quiz_overrides")
