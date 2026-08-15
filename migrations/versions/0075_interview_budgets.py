"""Teacher-configurable interview budgets.

Moves the two per-question budgets out of hardcoded orchestrator constants
(``DEFAULT_MAX_FOLLOWUPS_PER_QUESTION`` / ``MAX_CANNOT_ANSWER_HINTS``) into
``interview_configs`` columns so a teacher can tune how long the interviewer
may stay on one question before the server advances.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

# NOTE: alembic_version.version_num is varchar(32); keep revision id <= 32 chars.
revision = "0075_iv_budgets"
down_revision = "0074_career_path_versions"
branch_labels = None
depends_on = None

_TABLE = "interview_configs"
_COLUMNS = (
    ("max_follow_ups_per_question", sa.Integer(), "2"),
    ("max_hints_per_question", sa.Integer(), "3"),
)


def upgrade() -> None:
    for name, type_, default in _COLUMNS:
        op.add_column(
            _TABLE,
            sa.Column(name, type_, nullable=False, server_default=sa.text(default)),
        )
    op.create_check_constraint(
        "ck_interview_configs_max_follow_ups",
        _TABLE,
        "max_follow_ups_per_question BETWEEN 0 AND 10",
    )
    op.create_check_constraint(
        "ck_interview_configs_max_hints",
        _TABLE,
        "max_hints_per_question BETWEEN 0 AND 10",
    )


def downgrade() -> None:
    op.drop_constraint("ck_interview_configs_max_hints", _TABLE, type_="check")
    op.drop_constraint("ck_interview_configs_max_follow_ups", _TABLE, type_="check")
    for name, _, _ in _COLUMNS:
        op.drop_column(_TABLE, name)
