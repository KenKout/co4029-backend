"""Persist strict role-only interview generation mode.

Revision ID: 0091_strict_role_only
Revises: 0090_job_correlation
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0091_strict_role_only"
down_revision = "0090_job_correlation"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "interview_configs",
        sa.Column("generation_variant_strategy", sa.String(length=20), nullable=True),
    )
    op.create_check_constraint(
        "ck_interview_configs_generation_variant_strategy",
        "interview_configs",
        "generation_variant_strategy IS NULL OR generation_variant_strategy = 'role_only'",
    )


def downgrade() -> None:
    op.drop_constraint(
        "ck_interview_configs_generation_variant_strategy",
        "interview_configs",
        type_="check",
    )
    op.drop_column("interview_configs", "generation_variant_strategy")
