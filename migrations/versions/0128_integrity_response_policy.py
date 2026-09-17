"""Teacher-selectable response to a quiz proctoring threshold crossing.

Revision ID: 0128_integrity_response_policy
Revises: 0127_one_active_quiz_attempt
Create Date: 2026-09-17

Quizzes score proctoring signals (tab switch, focus loss, full-screen exit)
against a threshold, and until now hardcoded the reaction once that threshold
was crossed: warn the learner. A formative weekly quiz and a summative
examination therefore behaved identically, which is the gap this closes.

``warn_and_continue`` is the default, so every existing quiz keeps the
behaviour it has today and the change is inert until a teacher chooses
otherwise.

Quizzes only. ``interview_configs`` carries a similarly named
``security_response_policy``, but that governs the AI guard's reaction to a
prompt-injection attempt in a candidate's utterance — a different signal with
a different subject. Interview proctoring still always warns; bringing the two
into line is a separate decision, not a side effect of this one.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0128_integrity_response_policy"
down_revision = "0127_one_active_quiz_attempt"
branch_labels = None
depends_on = None

_ALLOWED = "('continue_and_log', 'warn_and_continue')"


def upgrade() -> None:
    op.add_column(
        "quizzes",
        sa.Column(
            "integrity_response_policy",
            sa.String(length=32),
            nullable=False,
            server_default=sa.text("'warn_and_continue'"),
        ),
    )
    op.create_check_constraint(
        "ck_quizzes_integrity_response_policy",
        "quizzes",
        f"integrity_response_policy IN {_ALLOWED}",
    )


def downgrade() -> None:
    op.drop_constraint(
        "ck_quizzes_integrity_response_policy", "quizzes", type_="check"
    )
    op.drop_column("quizzes", "integrity_response_policy")
