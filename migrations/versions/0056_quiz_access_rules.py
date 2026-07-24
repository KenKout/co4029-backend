"""Phase 12: quiz access rules — password, subnet allowlist, browser security, attempt delays.

Revision ID: 0056_quiz_access_rules
Revises: 0055_question_bank_cats
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
revision = "0056_quiz_access_rules"
down_revision = "0055_question_bank_cats"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("quizzes", sa.Column("require_password", sa.String(255), nullable=True))
    op.add_column("quizzes", sa.Column("require_subnet", sa.String(1024), nullable=True))
    op.add_column("quizzes", sa.Column("browser_security", sa.String(20), nullable=False, server_default=sa.text("'none'")))
    op.add_column("quizzes", sa.Column("delay1_seconds", sa.Integer(), nullable=True))
    op.add_column("quizzes", sa.Column("delay2_seconds", sa.Integer(), nullable=True))
    op.create_check_constraint("ck_quizzes_browser_security", "quizzes", "browser_security IN ('none', 'securewindow')")


def downgrade() -> None:
    op.drop_constraint("ck_quizzes_browser_security", "quizzes", type_="check")
    op.drop_column("quizzes", "delay2_seconds")
    op.drop_column("quizzes", "delay1_seconds")
    op.drop_column("quizzes", "browser_security")
    op.drop_column("quizzes", "require_subnet")
    op.drop_column("quizzes", "require_password")
