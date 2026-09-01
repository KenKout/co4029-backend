"""Switch the interview retake cooldown from hours to minutes.

``interview_configs.cooldown_hours`` could only express whole hours, so a
teacher could not ask for a short breather (15 / 30 min) between attempts — the
smallest gate available was a full hour. The column becomes
``cooldown_minutes`` and existing values are multiplied by 60 so every live
config keeps the exact same window.

The quiz-side ``quizzes.cooldown_hours`` is a SEPARATE column and is
deliberately untouched.

Revision ID: 0099_ic_cooldown_minutes
Revises: 0098_drop_ic_variant_strat
"""

from __future__ import annotations

from alembic import op

revision = "0099_ic_cooldown_minutes"
down_revision = "0098_drop_ic_variant_strat"
branch_labels = None
depends_on = None


_TABLE = "interview_configs"


def upgrade() -> None:
    op.alter_column(_TABLE, "cooldown_hours", new_column_name="cooldown_minutes")
    # Preserve every existing window exactly: 24h -> 1440min.
    op.execute(
        f"UPDATE {_TABLE} SET cooldown_minutes = cooldown_minutes * 60 "
        "WHERE cooldown_minutes IS NOT NULL"
    )
    op.create_check_constraint(
        "ck_interview_configs_cooldown_minutes",
        _TABLE,
        "cooldown_minutes IS NULL OR cooldown_minutes > 0",
    )


def downgrade() -> None:
    op.drop_constraint("ck_interview_configs_cooldown_minutes", _TABLE, type_="check")
    # Integer division: a sub-hour cooldown (the whole point of this migration)
    # cannot survive a downgrade. Round UP so a 30-minute window degrades to 1
    # hour rather than to 0, which the old code read as "no cooldown at all" —
    # silently dropping a gate is worse than making it slightly stricter.
    op.execute(
        f"UPDATE {_TABLE} SET cooldown_minutes = CEIL(cooldown_minutes / 60.0) "
        "WHERE cooldown_minutes IS NOT NULL"
    )
    op.alter_column(_TABLE, "cooldown_minutes", new_column_name="cooldown_hours")
