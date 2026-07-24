"""Phase 6: overdue handling + grace period on quizzes; per-attempt layout snapshot.

Revision ID: 0050_quiz_timing_enforce
Revises: 0049_quiz_overrides
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
revision = "0050_quiz_timing_enforce"
down_revision = "0049_quiz_overrides"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("quizzes", sa.Column("overdue_handling", sa.String(20), nullable=False, server_default=sa.text("'autosubmit'")))
    op.add_column("quizzes", sa.Column("grace_period_seconds", sa.Integer(), nullable=True))
    op.create_check_constraint("ck_quizzes_overdue_handling", "quizzes", "overdue_handling IN ('autosubmit', 'graceperiod', 'autoabandon')")
    op.add_column("quiz_attempts", sa.Column("layout", JSONB(), nullable=True))


def downgrade() -> None:
    op.drop_column("quiz_attempts", "layout")
    op.drop_constraint("ck_quizzes_overdue_handling", "quizzes", type_="check")
    op.drop_column("quizzes", "grace_period_seconds")
    op.drop_column("quizzes", "overdue_handling")
