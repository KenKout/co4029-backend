"""Add interview_configs.persona_profile_json (teacher per-trait persona overrides)

Revision ID: 0060_persona_profile
Revises: 0059_lo_hierarchy
Create Date: 2026-07-25

Phase 3 of the trait-based persona plan. Adds an optional JSONB column holding
per-trait overrides on top of the preset persona:

* ``persona`` VARCHAR(20) + its CHECK constraint stay UNTOUCHED — it remains the
  preset selector (strict / neutral / supportive).
* ``persona_profile_json`` (nullable JSONB) holds optional trait overrides
  (warmth / directness / verbosity / formality / ack_frequency / opening_style)
  merged on top of the preset in ``profile_from``. NULL → preset only, so every
  existing config is unaffected and behaves exactly as before.

No CHECK constraint on the JSON: trait values are clamped to [0, 4] and unknown
keys ignored in application code (``profile_from``), keeping the trait schema
editable without a migration. Persona traits are teacher-only and never exposed
on any learner-facing schema.

Additive, nullable column — fully reversible.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

# NOTE: alembic_version.version_num is varchar(32); keep revision id <= 32 chars.
revision = "0060_persona_profile"
down_revision = "0059_lo_hierarchy"
branch_labels = None
depends_on = None

_TABLE = "interview_configs"
_COLUMN = "persona_profile_json"


def upgrade() -> None:
    op.add_column(_TABLE, sa.Column(_COLUMN, JSONB(), nullable=True))


def downgrade() -> None:
    op.drop_column(_TABLE, _COLUMN)
