"""Add ai_model_pricing table, seeded from the hardcoded PRICE_TABLE

Revision ID: 0020_ai_model_pricing
Revises: 0019_interview_cooldown_hours
Create Date: 2026-07-13 00:00:00.000000

Cost computation previously read a hand-maintained Python dict
(``abridgeai.ai.llm.pricing.PRICE_TABLE``) that required a code change and
a release to update. This introduces ``ai_model_pricing`` as the runtime
source of truth and seeds it with the exact rates the dict carried at
migration time, so per-call cost figures do not change on upgrade. Admins
can add/edit/remove rows going forward via the admin UI; a model with no
row still yields NULL cost (unchanged "refuse to guess" behaviour).
"""

from __future__ import annotations

import uuid

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0020_ai_model_pricing"
down_revision = "0019_interview_cooldown_hours"
branch_labels = None
depends_on = None

_TABLE = "ai_model_pricing"

# Snapshot of PRICE_TABLE as of 2026-05-16 (see abridgeai/ai/llm/pricing.py).
_SEED_ROWS = [
    ("gpt-4o", "0.005", "0.015"),
    ("gpt-4o-mini", "0.00015", "0.0006"),
    ("text-embedding-3-small", "0.00002", "0"),
    ("text-embedding-3-large", "0.00013", "0"),
]


def upgrade() -> None:
    op.create_table(
        _TABLE,
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("uuid_generate_v4()"),
        ),
        sa.Column("model_name", sa.String(length=100), nullable=False, unique=True),
        sa.Column("input_usd_per_1k", sa.Numeric(12, 6), nullable=False),
        sa.Column("output_usd_per_1k", sa.Numeric(12, 6), nullable=False),
        sa.Column("notes", sa.String(length=255), nullable=True),
        sa.Column(
            "updated_by",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.CheckConstraint("input_usd_per_1k >= 0", name="ck_ai_model_pricing_input_nonneg"),
        sa.CheckConstraint("output_usd_per_1k >= 0", name="ck_ai_model_pricing_output_nonneg"),
    )
    op.create_index(
        "ix_ai_model_pricing_model_name", _TABLE, ["model_name"], unique=True
    )

    pricing_table = sa.table(
        _TABLE,
        sa.column("id", postgresql.UUID(as_uuid=True)),
        sa.column("model_name", sa.String),
        sa.column("input_usd_per_1k", sa.Numeric),
        sa.column("output_usd_per_1k", sa.Numeric),
    )
    op.bulk_insert(
        pricing_table,
        [
            {
                "id": uuid.uuid4(),
                "model_name": name,
                "input_usd_per_1k": input_rate,
                "output_usd_per_1k": output_rate,
            }
            for name, input_rate, output_rate in _SEED_ROWS
        ],
    )


def downgrade() -> None:
    op.drop_index("ix_ai_model_pricing_model_name", table_name=_TABLE)
    op.drop_table(_TABLE)
