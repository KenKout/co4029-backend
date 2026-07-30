"""Phase 1 (versioning): pin quiz_attempt_answers to the question revision graded against.

Revision ID: 0044_quiz_ans_graded_rev
Revises: 0043_interview_tts_voice
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
revision = "0044_quiz_ans_graded_rev"
down_revision = "0043_interview_tts_voice"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "quiz_attempt_answers",
        sa.Column("graded_revision_id", PGUUID(as_uuid=True), nullable=True),
    )
    op.create_foreign_key(
        "fk_quiz_attempt_answers_graded_revision",
        "quiz_attempt_answers", "quiz_question_revisions",
        ["graded_revision_id"], ["id"], ondelete="SET NULL",
    )
    # Backfill: point each existing answer at the LOWEST revision of its question.
    op.execute("""
        UPDATE quiz_attempt_answers a
        SET graded_revision_id = r.id
        FROM (
            SELECT DISTINCT ON (question_id) id, question_id
            FROM quiz_question_revisions
            ORDER BY question_id, revision_no ASC
        ) r
        WHERE a.question_id = r.question_id AND a.graded_revision_id IS NULL
    """)


def downgrade() -> None:
    op.drop_constraint("fk_quiz_attempt_answers_graded_revision", "quiz_attempt_answers", type_="foreignkey")
    op.drop_column("quiz_attempt_answers", "graded_revision_id")
