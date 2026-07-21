"""Add persisted candidate onboarding and assessed timer anchor.

Revision ID: 0027_interview_onboarding
Revises: 0026_interview_ceremony_turns
Create Date: 2026-07-19 00:00:00.000000
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0027_interview_onboarding"
down_revision = "0026_interview_ceremony_turns"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "interview_sessions",
        sa.Column("assessment_started_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "interview_sessions",
        sa.Column(
            "onboarding_stage",
            sa.String(length=20),
            nullable=False,
            server_default=sa.text("'identity_check'"),
        ),
    )
    op.add_column(
        "interview_sessions",
        sa.Column(
            "interview_language",
            sa.String(length=5),
            nullable=False,
            server_default=sa.text("'en'"),
        ),
    )

    # Sessions created by older application versions must resume directly in
    # their established question flow. Only newly-created sessions use setup.
    #
    # Some restored development databases contain legacy interview sessions
    # whose student was removed while FK triggers were disabled. PostgreSQL
    # rechecks the existing student FK when these rows are updated, even though
    # this migration does not change student_id. Restrict the data backfill to
    # sessions with a live user so those inaccessible orphan rows are preserved
    # at the new-column defaults instead of making the schema upgrade fail.
    op.execute(
        "UPDATE interview_sessions AS s SET interview_language = COALESCE(("
        "SELECT CASE WHEN m.metadata_json->>'language' = 'vi' THEN 'vi' ELSE 'en' END "
        "FROM interview_session_messages AS m "
        "WHERE m.session_id = s.id AND m.metadata_json->>'ceremony_key' = 'opening' "
        "ORDER BY m.created_at LIMIT 1), 'en') "
        "WHERE EXISTS (SELECT 1 FROM users AS u WHERE u.id = s.student_id)"
    )
    op.execute(
        "UPDATE interview_sessions AS s "
        "SET onboarding_stage = 'completed', assessment_started_at = started_at "
        "WHERE EXISTS (SELECT 1 FROM users AS u WHERE u.id = s.student_id)"
    )
    op.create_check_constraint(
        "ck_interview_sessions_onboarding_stage",
        "interview_sessions",
        "onboarding_stage IN ('identity_check', 'readiness', 'completed')",
    )
    op.create_check_constraint(
        "ck_interview_sessions_language",
        "interview_sessions",
        "interview_language IN ('en', 'vi')",
    )


def downgrade() -> None:
    op.drop_constraint(
        "ck_interview_sessions_language", "interview_sessions", type_="check"
    )
    op.drop_constraint(
        "ck_interview_sessions_onboarding_stage",
        "interview_sessions",
        type_="check",
    )
    op.drop_column("interview_sessions", "interview_language")
    op.drop_column("interview_sessions", "onboarding_stage")
    op.drop_column("interview_sessions", "assessment_started_at")
