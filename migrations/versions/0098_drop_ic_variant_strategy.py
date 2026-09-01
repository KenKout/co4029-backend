"""Drop the write-only generation_variant_strategy column from interview_configs.

Added by 0091 so the RUNTIME could serve only the interviewer role's preferred
question type ("strict role_only"). Commit c923073 then made runtime selection
role-driven instead: complete 4-angle variant groups collapse to one role-aware
member and ungrouped questions are askable by any role, so nothing reads the
column any more — ``ai/pipelines/generation.py`` still set it to 'role_only'
after a successful role-only run and no code path ever read it back. It also
survived a config's regeneration (never cleared on a mixed re-run), which made
it misleading as metadata on top of being unused.

The generation-time strategy still lives where it is actually consumed:
``generation_runs.config_json['variant_strategy']``, written per run and read by
the generation pipeline.

Revision ID: 0098_drop_ic_variant_strat
Revises: 0097_path_change_in_progress
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0098_drop_ic_variant_strat"
down_revision = "0097_path_change_in_progress"
branch_labels = None
depends_on = None


_TABLE = "interview_configs"
_COLUMN = "generation_variant_strategy"
_CHECK = "ck_interview_configs_generation_variant_strategy"


def upgrade() -> None:
    op.drop_constraint(_CHECK, _TABLE, type_="check")
    op.drop_column(_TABLE, _COLUMN)


def downgrade() -> None:
    # Nullable with no server_default, exactly as 0091 created it: NULL was the
    # "legacy / not role-only" state, so a downgrade lands every row there.
    op.add_column(_TABLE, sa.Column(_COLUMN, sa.String(length=20), nullable=True))
    op.create_check_constraint(
        _CHECK,
        _TABLE,
        f"{_COLUMN} IS NULL OR {_COLUMN} = 'role_only'",
    )
