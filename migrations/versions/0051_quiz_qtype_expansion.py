"""Phase 7: widen question_type CHECK + add multi-select/numerical/matching/ordering columns.

Revision ID: 0051_quiz_qtype_expansion
Revises: 0050_quiz_timing_enforce
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
revision = "0051_quiz_qtype_expansion"
down_revision = "0050_quiz_timing_enforce"
branch_labels = None
depends_on = None


_OLD_TYPES = ("multiple_choice", "true_false", "short_answer", "fill_blank", "code")
_NEW_TYPES = _OLD_TYPES + ("numerical", "matching", "ordering")
_CK_NAME = "quiz_questions_question_type_check"  # confirmed live name


def upgrade() -> None:
    op.drop_constraint(_CK_NAME, "quiz_questions", type_="check")
    allowed = ", ".join(f"'{t}'" for t in _NEW_TYPES)
    op.create_check_constraint(_CK_NAME, "quiz_questions", f"question_type IN ({allowed})")

    op.add_column("quiz_questions", sa.Column("single_answer", sa.Boolean(), nullable=False, server_default=sa.text("true")))
    op.add_column("quiz_questions", sa.Column("answer_numbering", sa.String(16), nullable=False, server_default=sa.text("'abc'")))
    op.add_column("quiz_questions", sa.Column("numeric_answer", sa.Numeric(18, 6), nullable=True))
    op.add_column("quiz_questions", sa.Column("numeric_tolerance", sa.Numeric(18, 6), nullable=True, server_default=sa.text("0")))
    op.add_column("quiz_questions", sa.Column("match_pairs", JSONB, nullable=True))
    op.add_column("quiz_questions", sa.Column("ordering_sequence", JSONB, nullable=True))
    op.add_column("quiz_question_options", sa.Column("grade_fraction", sa.Numeric(5, 4), nullable=True))
    op.add_column("quiz_attempt_answers", sa.Column("selected_option_ids", JSONB, nullable=True))


def downgrade() -> None:
    op.drop_column("quiz_attempt_answers", "selected_option_ids")
    op.drop_column("quiz_question_options", "grade_fraction")
    op.drop_column("quiz_questions", "ordering_sequence")
    op.drop_column("quiz_questions", "match_pairs")
    op.drop_column("quiz_questions", "numeric_tolerance")
    op.drop_column("quiz_questions", "numeric_answer")
    op.drop_column("quiz_questions", "answer_numbering")
    op.drop_column("quiz_questions", "single_answer")
    op.drop_constraint(_CK_NAME, "quiz_questions", type_="check")
    allowed = ", ".join(f"'{t}'" for t in _OLD_TYPES)
    op.create_check_constraint(_CK_NAME, "quiz_questions", f"question_type IN ({allowed})")
