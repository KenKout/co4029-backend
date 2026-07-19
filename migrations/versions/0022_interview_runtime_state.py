"""Add interview_runtime_states table (adaptive interviewer Phase 1)

Revision ID: 0022_interview_runtime_state
Revises: 0021_iq_model_answer
Create Date: 2026-07-14 00:00:00.000000

Introduces the persistent runtime state for the stateful Interview
Orchestrator (adaptive interviewer, staged rollout behind the
``adaptive_interviewer_enabled`` flag).

One row per interview session (UNIQUE ``session_id``, ON DELETE CASCADE).
The bulk of the orchestrator state lives in ``state_json`` (JSONB) so the
shape can evolve without a migration per field; only the hot invariants get
real columns:

* ``phase``                     — CHECK-constrained interview phase.
* ``state_version``             — monotonic optimistic-lock counter that
                                  guards against double-advancement on
                                  retried REST / duplicate LiveKit callbacks.
* ``last_turn_idempotency_key`` — idempotency key of the last processed
                                  student turn (duplicate = no-op replay).

Additive + lazily initialised: existing sessions have no row and the legacy
sequential flow never reads this table, so active sessions are unaffected.
Fully reversible.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision = "0022_interview_runtime_state"
down_revision = "0021_iq_model_answer"
branch_labels = None
depends_on = None

_TABLE = "interview_runtime_states"


def upgrade() -> None:
    op.create_table(
        _TABLE,
        sa.Column(
            "id",
            UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("uuid_generate_v4()"),
            nullable=False,
        ),
        sa.Column(
            "session_id",
            UUID(as_uuid=True),
            sa.ForeignKey("interview_sessions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "phase",
            sa.String(length=20),
            server_default=sa.text("'opening'"),
            nullable=False,
        ),
        sa.Column(
            "state_version",
            sa.Integer(),
            server_default=sa.text("0"),
            nullable=False,
        ),
        sa.Column("last_turn_idempotency_key", sa.String(length=255), nullable=True),
        sa.Column(
            "state_json",
            JSONB(),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("NOW()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("NOW()"),
            nullable=False,
        ),
        sa.UniqueConstraint("session_id", name="uq_interview_runtime_states_session"),
        sa.CheckConstraint(
            "phase IN ('opening', 'warmup', 'core', 'deep_probe', 'closing', 'completed')",
            name="ck_interview_runtime_states_phase",
        ),
    )


def downgrade() -> None:
    op.drop_table(_TABLE)
