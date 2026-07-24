"""Phase 3: per-field format discriminators (plain|markdown|html) for rich media.

Revision ID: 0047_quiz_rich_formats
Revises: 0046_quiz_review_options
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
revision = "0047_quiz_rich_formats"
down_revision = "0046_quiz_review_options"
branch_labels = None
depends_on = None


_FMT_CHECK = "IN ('plain', 'markdown', 'html')"


def upgrade() -> None:
    for col in ("prompt_format", "hint_format", "explanation_format"):
        op.add_column("quiz_questions", sa.Column(col, sa.String(10), nullable=False, server_default="plain"))
        op.create_check_constraint(f"ck_quiz_questions_{col}", "quiz_questions", f"{col} {_FMT_CHECK}")
    op.add_column("quiz_question_options", sa.Column("option_format", sa.String(10), nullable=False, server_default="plain"))
    op.create_check_constraint("ck_quiz_question_options_option_format", "quiz_question_options", f"option_format {_FMT_CHECK}")


def downgrade() -> None:
    op.drop_constraint("ck_quiz_question_options_option_format", "quiz_question_options", type_="check")
    op.drop_column("quiz_question_options", "option_format")
    for col in ("explanation_format", "hint_format", "prompt_format"):
        op.drop_constraint(f"ck_quiz_questions_{col}", "quiz_questions", type_="check")
        op.drop_column("quiz_questions", col)
