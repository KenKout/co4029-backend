"""Persist interview recording consent decisions and completion time.

Revision ID: 0124_interview_recording_consent
Revises: 0123_interview_recordings
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0124_interview_recording_consent"
down_revision = "0123_interview_recordings"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "interview_sessions",
        sa.Column("recording_consent_status", sa.String(length=20), nullable=True),
    )
    op.add_column(
        "interview_sessions",
        sa.Column("recording_consent_policy_version", sa.String(length=32), nullable=True),
    )
    op.add_column(
        "interview_sessions",
        sa.Column("recording_consented_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "interview_sessions",
        sa.Column("recording_consent_scope", sa.String(length=20), nullable=True),
    )
    op.create_check_constraint(
        "ck_interview_sessions_recording_consent_status",
        "interview_sessions",
        "recording_consent_status IS NULL OR recording_consent_status IN ('accepted', 'declined')",
    )
    op.create_check_constraint(
        "ck_interview_sessions_recording_consent_scope",
        "interview_sessions",
        "recording_consent_scope IS NULL OR recording_consent_scope IN ('audio_only')",
    )
    op.add_column(
        "interview_recordings",
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("interview_recordings", "completed_at")
    op.drop_constraint(
        "ck_interview_sessions_recording_consent_scope", "interview_sessions", type_="check"
    )
    op.drop_constraint(
        "ck_interview_sessions_recording_consent_status", "interview_sessions", type_="check"
    )
    op.drop_column("interview_sessions", "recording_consent_scope")
    op.drop_column("interview_sessions", "recording_consented_at")
    op.drop_column("interview_sessions", "recording_consent_policy_version")
    op.drop_column("interview_sessions", "recording_consent_status")
