"""Add partial unique index for adaptive-turn idempotency

Revision ID: 0023_interview_turn_key
Revises: 0022_interview_runtime_state
Create Date: 2026-07-14 00:00:00.000000

Slice 4 (adaptive interviewer, Phase 17) needs a DB-level guarantee that a
retried student turn carrying the same ``turn_key`` can NEVER insert a second
answer row (safeguard #1). The application does a fast pre-insert check against
the runtime-state row, but under a race two concurrent identical POSTs could
both pass that check; this index is the hard backstop.

We store the per-turn idempotency key on the student answer message at
``interview_session_messages.metadata_json->>'turn_key'``. A PARTIAL, FUNCTIONAL
UNIQUE index on ``(session_id, (metadata_json->>'turn_key'))`` — filtered to
rows that actually carry a key (``role='user'`` AND the key is present) — makes
a duplicate insert fail with a unique-violation the service maps to a replay.

The partial predicate matters:
* ``role = 'user'`` — only student answers carry a turn_key; AI turns/system
  messages must be free to share (session_id, NULL).
* ``metadata_json ? 'turn_key'`` — legacy rows and any future keyless writes
  are excluded, so this index is inert for the existing sequential path and for
  all historical data (no backfill, no collision risk on deploy).

Fully additive and reversible.
"""

from __future__ import annotations

from alembic import op

revision = "0023_interview_turn_key"
down_revision = "0022_interview_runtime_state"
branch_labels = None
depends_on = None

_INDEX = "uq_interview_msg_turn_key"
_TABLE = "interview_session_messages"


def upgrade() -> None:
    # Functional + partial unique index. Expressed as raw SQL because the key is
    # a JSONB extraction (->>): Alembic's create_index can't express a functional
    # expression index portably.
    op.execute(
        f"""
        CREATE UNIQUE INDEX {_INDEX}
        ON {_TABLE} (session_id, (metadata_json ->> 'turn_key'))
        WHERE role = 'user' AND metadata_json ? 'turn_key'
        """
    )


def downgrade() -> None:
    op.execute(f"DROP INDEX IF EXISTS {_INDEX}")
