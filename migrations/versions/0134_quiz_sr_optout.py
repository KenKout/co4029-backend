"""Let a quiz decline to open spaced-repetition cards.

Every answer to a first-time question opens a card, with no regard for what
the quiz is for. A final assessment therefore behaves like practice: each of
its questions becomes a permanent card, a wrong answer under exam conditions
drives EF down into daily repeats, and because ``min_ef_for_unlock`` and
``coverage_threshold`` read SM-2 state, exam performance can decide lesson
unlock eligibility.

``feeds_spaced_repetition`` is that distinction, defaulting to TRUE so every
existing quiz keeps behaving exactly as it does now.

Off means the quiz never CREATES a card. It does NOT mean its answers are
invisible to SM-2: a question the student already holds a card for — met in a
practice quiz, or imported from the same bank item — is still graded and
rescheduled normally. Suppressing that as well would let an exam shield a
card from the evidence of having been failed, which is the opposite of what
an assessment is for.

Deliberately NOT added to ``_PUBLISHED_EDITABLE_FIELDS``: like every other
SM-2 parameter on the quiz it freezes at publication, so it cannot change
under a live or completed attempt.

Revision ID: 0134_quiz_sr_optout
Revises: 0133_retirable_card_due_at
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0134_quiz_sr_optout"
down_revision = "0133_retirable_card_due_at"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "quizzes",
        sa.Column(
            "feeds_spaced_repetition",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("TRUE"),
        ),
    )


def downgrade() -> None:
    # Dropping the column restores the old behaviour for every quiz: all of
    # them feed the scheduler again. A quiz that had been marked as an
    # assessment loses that mark, and questions it asks will start opening
    # cards on the next first answer.
    op.drop_column("quizzes", "feeds_spaced_repetition")
