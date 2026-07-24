"""Rename ai_model_pricing rates from per-1K to per-1M tokens

Revision ID: 0041_pricing_per_million
Revises: 0040_notif_job_categories
Create Date: 2026-07-24 00:00:00.000000

The industry convention (OpenAI, Anthropic, Google) quotes token prices per
1,000,000 tokens, not per 1,000. The original ``ai_model_pricing`` schema
used ``*_usd_per_1k`` columns, which made admins prone to entering a per-1M
figure into a per-1K field — a silent 1000x over-charge (observed on the
gemini rows: a ~1.7K-token call billed at ~$10 instead of ~$0.01).

This migration switches the stored convention to per-1M:
  * renames ``input_usd_per_1k`` -> ``input_usd_per_1m`` and
    ``output_usd_per_1k`` -> ``output_usd_per_1m``;
  * multiplies every existing value by 1000 so the *per-call cost stays
    identical* for rows that were entered correctly (the 4 OpenAI rows).

Rows that were mis-entered (the gemini rows, where a per-1M number sat in a
per-1K field) are corrected separately by a data-fix run at deploy time, NOT
here — this migration only performs the mechanical, value-preserving unit
change so it is safe and reproducible on any environment.

The check-constraint names are kept stable (they don't reference the unit),
only the guarded column names change.
"""

from __future__ import annotations

from alembic import op

revision = "0041_pricing_per_million"
down_revision = "0040_notif_job_cats"
branch_labels = None
depends_on = None

_TABLE = "ai_model_pricing"


def upgrade() -> None:
    # 1) Convert values in place (per-1K -> per-1M is x1000), preserving the
    #    per-call cost for correctly-entered rows.
    op.execute(
        f"UPDATE {_TABLE} SET "
        "input_usd_per_1k = input_usd_per_1k * 1000, "
        "output_usd_per_1k = output_usd_per_1k * 1000"
    )
    # 2) Rename the columns to reflect the new per-1M unit.
    op.alter_column(_TABLE, "input_usd_per_1k", new_column_name="input_usd_per_1m")
    op.alter_column(_TABLE, "output_usd_per_1k", new_column_name="output_usd_per_1m")


def downgrade() -> None:
    op.alter_column(_TABLE, "input_usd_per_1m", new_column_name="input_usd_per_1k")
    op.alter_column(_TABLE, "output_usd_per_1m", new_column_name="output_usd_per_1k")
    op.execute(
        f"UPDATE {_TABLE} SET "
        "input_usd_per_1k = input_usd_per_1k / 1000, "
        "output_usd_per_1k = output_usd_per_1k / 1000"
    )
