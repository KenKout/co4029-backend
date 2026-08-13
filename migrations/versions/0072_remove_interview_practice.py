"""Remove the interview practice-mode columns (revert 0065).

Drops the three columns that powered the rehearsal feature — the bank
partition flag, the config opt-in, and the per-session mode — together with
their CHECK constraint. The application code that read these columns was
removed in the same change; this migration only reclaims the schema.

Data note: existing practice rows lose their mode marker, so any session that
was ``practice`` reads as ``assessment`` afterwards. This is the intended,
irreversible cleanup of a feature being retired wholesale.

Revision ID: 0072_remove_interview_practice
Revises: 0071_user_disable_perm
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

# NOTE: alembic_version.version_num is varchar(32); keep this id <= 32 chars.
revision = "0072_remove_interview_practice"
down_revision = "0071_user_disable_perm"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_constraint(
        "ck_interview_sessions_session_mode",
        "interview_sessions",
        type_="check",
    )
    op.drop_column("interview_sessions", "session_mode")
    op.drop_column("interview_configs", "practice_mode_enabled")
    op.drop_column("interview_questions", "practice_only")


def downgrade() -> None:
    op.add_column(
        "interview_questions",
        sa.Column(
            "practice_only",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("FALSE"),
        ),
    )
    op.add_column(
        "interview_configs",
        sa.Column(
            "practice_mode_enabled",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("FALSE"),
        ),
    )
    op.add_column(
        "interview_sessions",
        sa.Column(
            "session_mode",
            sa.String(length=20),
            nullable=False,
            server_default=sa.text("'assessment'"),
        ),
    )
    op.create_check_constraint(
        "ck_interview_sessions_session_mode",
        "interview_sessions",
        "session_mode IN ('assessment', 'practice')",
    )
