"""Split interview onboarding into short, single-purpose turns.

Revision ID: 0028_interview_onboarding_steps
Revises: 0027_interview_onboarding
Create Date: 2026-07-19 00:00:00.000000
"""

from __future__ import annotations

from alembic import op

revision = "0028_interview_onboarding_steps"
down_revision = "0027_interview_onboarding"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_constraint(
        "ck_interview_sessions_onboarding_stage",
        "interview_sessions",
        type_="check",
    )
    op.create_check_constraint(
        "ck_interview_sessions_onboarding_stage",
        "interview_sessions",
        "onboarding_stage IN ('identity_check', 'audio_check', "
        "'language_check', 'preparation', 'readiness', 'completed')",
    )


def downgrade() -> None:
    # Intermediate stages cannot be represented by the previous constraint.
    # Restart their setup at identity confirmation before narrowing the enum.
    op.execute(
        "UPDATE interview_sessions SET onboarding_stage = 'identity_check' "
        "WHERE onboarding_stage IN "
        "('audio_check', 'language_check', 'preparation')"
    )
    op.drop_constraint(
        "ck_interview_sessions_onboarding_stage",
        "interview_sessions",
        type_="check",
    )
    op.create_check_constraint(
        "ck_interview_sessions_onboarding_stage",
        "interview_sessions",
        "onboarding_stage IN ('identity_check', 'readiness', 'completed')",
    )
