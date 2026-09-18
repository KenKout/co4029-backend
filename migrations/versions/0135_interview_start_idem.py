"""Interview start-session idempotency key.

The start-session schema advertised ``idempotency_key`` ("retried POSTs on
flaky networks resolve to the same session — the QuizAttempt.idempotency_key
UNIQUE pattern") but the service accepted-and-ignored it: every retry
allocated a fresh attempt, and two racing starts that both passed the
active-session check collided on ``uq_interview_sessions_number`` with the
loser surfacing as a raw IntegrityError (HTTP 500) instead of the winner's
session.

Adds ``interview_sessions.idempotency_key`` (nullable UUID, UNIQUE) mirroring
the quiz attempt precedent. Nullable: legacy rows and requests without a key
must keep starting sessions.

Backfill: none, and none is possible — existing sessions were created without
a client key.

Revision IDs are stored in ``varchar(32)``: "0135_interview_start_idem" fits.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0135_interview_start_idem"
down_revision = "0134_quiz_sr_optout"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "interview_sessions",
        sa.Column("idempotency_key", sa.UUID(), nullable=True),
    )
    op.create_index(
        "uq_interview_sessions_idempotency",
        "interview_sessions",
        ["idempotency_key"],
        unique=True,
        postgresql_where=sa.text("idempotency_key IS NOT NULL"),
    )


def downgrade() -> None:
    op.drop_index(
        "uq_interview_sessions_idempotency", table_name="interview_sessions"
    )
    op.drop_column("interview_sessions", "idempotency_key")
