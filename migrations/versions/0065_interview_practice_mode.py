"""Practice vs assessment mode for interviews.

Three columns, one feature: a low-stakes rehearsal run that does not consume an
attempt and is never graded.

``interview_questions.practice_only`` partitions the bank. A practice session
draws only from the marked questions; an assessment session draws only from the
rest. Partitioning rather than sharing the bank is deliberate: question
selection avoids repeats only WITHIN a session (``SelectionContext``), so a
practice run over the shared bank would pre-reveal the graded questions.

``interview_sessions.session_mode`` has to exist as its own column rather than
being inferred from a NULL ``pass_verdict``: ``recover_stalled_evaluations``
treats a terminal session with no verdict as a stalled evaluation and re-enqueues
it, so "ungraded" and "stuck" are otherwise indistinguishable.

Every default reproduces today's behaviour for existing rows: all questions are
gradable, no config offers practice, and every session already recorded is an
assessment.

Revision ID: 0065_interview_practice
Revises: 0064_preprocess_quar
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

# NOTE: alembic_version.version_num is varchar(32); keep this id <= 32 chars.
revision = "0065_interview_practice"
down_revision = "0064_preprocess_quar"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "interview_questions",
        sa.Column(
            "practice_only",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("FALSE"),
        ),
    )
    op.add_column(
        "interview_configs",
        sa.Column(
            "practice_mode_enabled",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("FALSE"),
        ),
    )
    op.add_column(
        "interview_sessions",
        sa.Column(
            "session_mode",
            sa.String(length=20),
            nullable=False,
            server_default=sa.text("'assessment'"),
        ),
    )
    op.create_check_constraint(
        "ck_interview_sessions_session_mode",
        "interview_sessions",
        "session_mode IN ('assessment', 'practice')",
    )


def downgrade() -> None:
    op.drop_constraint(
        "ck_interview_sessions_session_mode",
        "interview_sessions",
        type_="check",
    )
    op.drop_column("interview_sessions", "session_mode")
    op.drop_column("interview_configs", "practice_mode_enabled")
    op.drop_column("interview_questions", "practice_only")
