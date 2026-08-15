"""Raise the per-question follow-up budget ceiling from 10 to 50.

Applied out-of-band via docker exec on the shared deployment (every config
bumped to 20 turns/question); this migration keeps fresh environments and the
test DB at the same constraint.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

# NOTE: alembic_version.version_num is varchar(32); keep revision id <= 32 chars.
revision = "0078_raise_budget_bounds"
down_revision = "0077_drop_supported_modes"
branch_labels = None
depends_on = None

_TABLE = "interview_configs"


def upgrade() -> None:
    op.drop_constraint("ck_interview_configs_max_follow_ups", _TABLE, type_="check")
    op.create_check_constraint(
        "ck_interview_configs_max_follow_ups",
        _TABLE,
        "max_follow_ups_per_question BETWEEN 0 AND 50",
    )


def downgrade() -> None:
    op.drop_constraint("ck_interview_configs_max_follow_ups", _TABLE, type_="check")
    op.create_check_constraint(
        "ck_interview_configs_max_follow_ups",
        _TABLE,
        "max_follow_ups_per_question BETWEEN 0 AND 10",
    )
