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
# NOTE (deploy-tree reconciliation): on origin/master this chains off 0028, but
# the live DB reached 0030 via the concurrent (not-yet-merged) 0029/0030
# notification migrations. Repointed to 0030 here to keep a single linear head
# so `alembic upgrade head` succeeds. When 0029/0030 merge to master, rebase the
# canonical 0031 onto 0030 to match.
down_revision = "0030_backfill_action_url"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "interview_sessions",
        sa.Column("preferred_name", sa.String(length=60), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("interview_sessions", "preferred_name")
