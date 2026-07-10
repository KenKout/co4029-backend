"""Add cooldown_hours to interview_configs (FR-5.3)

Revision ID: 0019_interview_cooldown_hours
Revises: 0018_ai_calls_runtime_parentless
Create Date: 2026-06-29 13:30:00.000000

FR-5.3 requires interview retakes to honour a cooldown period between
attempts. The ``interview_configs`` table already carried ``max_attempts``
and ``lock_quiz_ef_until_pass`` but had no cooldown knob (the gap-analysis
flagged this). This adds a nullable ``cooldown_hours`` integer mirroring
``quizzes.cooldown_hours`` — NULL / <=0 disables the gate. Enforcement
lives read-side in ``features/interviews/services/taking.start_session``
(no data backfill required; existing configs default to NULL = no
cooldown, preserving current behaviour).
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0019_interview_cooldown_hours"
down_revision = "0018_ai_calls_runtime_parentless"
branch_labels = None
depends_on = None

_TABLE = "interview_configs"
_COLUMN = "cooldown_hours"


def upgrade() -> None:
    op.add_column(_TABLE, sa.Column(_COLUMN, sa.Integer(), nullable=True))


def downgrade() -> None:
    op.drop_column(_TABLE, _COLUMN)
