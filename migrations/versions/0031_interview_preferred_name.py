"""Add session-scoped preferred name for in-interview identity correction.

Revision ID: 0031_interview_pref_name
Revises: 0028_interview_onboarding_steps
Create Date: 2026-07-21 00:00:00.000000
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

# NOTE: alembic_version.version_num is varchar(32); keep this id <= 32 chars.
revision = "0031_interview_pref_name"
down_revision = "0028_interview_onboarding_steps"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "interview_sessions",
        sa.Column("preferred_name", sa.String(length=60), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("interview_sessions", "preferred_name")
