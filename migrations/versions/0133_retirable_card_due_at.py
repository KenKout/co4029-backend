"""Let a card retire by holding no next review date.

SM-2 grows an interval by multiplying the last one by EF, without limit, so a
card a student keeps answering correctly recedes indefinitely. Two settings
now bound that:

* ``spaced_repetition.max_interval_days`` caps the wait. This is Anki's
  ``Maximum Interval`` and carries its default of 36500 days (100 years), so
  it changes nothing until an operator lowers it. Note it buys retention
  rather than saving effort: a lower cap means the card returns MORE often.
* ``spaced_repetition.retire_beyond_max_interval`` turns that cap into a
  finish line instead. Off by default.

Retirement is stored as ``due_at IS NULL`` rather than as a new status
column. The card keeps its EF, interval, counters and full review history --
it simply has no next date. Every "due" read already filters
``due_at IS NOT NULL`` (the count query in the feature's public API had that
predicate before this migration existed), so a retired card leaves the queue
without any read needing to learn a new state.

Nothing is retired here. This only makes the state representable; the
settings that produce it are off by default, and no existing row changes.

The ``NOW()`` server default is kept: a card is created due, not retired.

Revision ID: 0133_retirable_card_due_at
Revises: 0132_revoke_reconciled
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0133_retirable_card_due_at"
down_revision = "0132_revoke_reconciled"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column(
        "student_card_state",
        "due_at",
        existing_type=sa.DateTime(timezone=True),
        existing_server_default=sa.text("NOW()"),
        nullable=True,
    )


def downgrade() -> None:
    # A retired card has no honest non-null date to fall back to, and the
    # column is NOT NULL again after this. Bringing them back as due now is
    # the conservative direction: the student sees them in the queue again,
    # which is the behaviour that predates retirement, rather than being
    # silently dropped.
    op.execute(
        "UPDATE student_card_state SET due_at = NOW() WHERE due_at IS NULL"
    )
    op.alter_column(
        "student_card_state",
        "due_at",
        existing_type=sa.DateTime(timezone=True),
        existing_server_default=sa.text("NOW()"),
        nullable=False,
    )
