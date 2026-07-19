"""Add prompt-injection policy, versioning, flags, and redacted events.

Revision ID: 0025_interview_security_guard
Revises: 0024_user_profile_locale
Create Date: 2026-07-19 00:00:00.000000

All changes are additive. Existing sessions/configs receive conservative
defaults; runtime JSON remains lazily/backward compatible.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0025_interview_security_guard"
down_revision = "0024_user_profile_locale"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "interview_configs",
        sa.Column(
            "security_response_policy",
            sa.String(length=30),
            nullable=False,
            server_default=sa.text("'warn_and_continue'"),
        ),
    )
    op.add_column(
        "interview_configs",
        sa.Column(
            "security_max_consecutive_attempts",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("3"),
        ),
    )
    op.add_column(
        "interview_configs", sa.Column("security_custom_refusal_en", sa.Text(), nullable=True)
    )
    op.add_column(
        "interview_configs", sa.Column("security_custom_refusal_vi", sa.Text(), nullable=True)
    )
    op.add_column(
        "interview_configs",
        sa.Column(
            "security_incident_summary_enabled",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("TRUE"),
        ),
    )
    op.create_check_constraint(
        "ck_interview_configs_security_response_policy",
        "interview_configs",
        "security_response_policy IN "
        "('continue_and_log', 'warn_and_continue', 'end_and_flag')",
    )
    op.create_check_constraint(
        "ck_interview_configs_security_max_attempts",
        "interview_configs",
        "security_max_consecutive_attempts BETWEEN 2 AND 20",
    )

    for name, default in (
        ("security_policy_version", "2026-07-19"),
        ("security_rules_version", "1.0.0"),
        ("security_prompt_version", "1.0.0"),
        ("output_guard_version", "1.0.0"),
    ):
        op.add_column(
            "interview_sessions",
            sa.Column(
                name,
                sa.String(length=32),
                nullable=False,
                server_default=sa.text(f"'{default}'"),
            ),
        )
    op.add_column(
        "interview_sessions",
        sa.Column(
            "session_security_flagged",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("FALSE"),
        ),
    )

    op.create_table(
        "interview_security_events",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            nullable=False,
            server_default=sa.text("uuid_generate_v4()"),
        ),
        sa.Column(
            "session_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("interview_sessions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("turn_id", sa.String(length=255), nullable=False),
        sa.Column("event_type", sa.String(length=64), nullable=False),
        sa.Column("category", sa.String(length=64), nullable=False),
        sa.Column("confidence_band", sa.String(length=10), nullable=False),
        sa.Column("action", sa.String(length=40), nullable=False),
        sa.Column("attempt_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column(
            "fallback_status", sa.Boolean(), nullable=False, server_default=sa.text("FALSE")
        ),
        sa.Column("normalized_fingerprint", sa.String(length=64), nullable=True),
        sa.Column("protected_content_category", sa.String(length=64), nullable=True),
        sa.Column("policy_version", sa.String(length=32), nullable=False),
        sa.Column("rules_version", sa.String(length=32), nullable=False),
        sa.Column("prompt_version", sa.String(length=32), nullable=False),
        sa.Column("output_guard_version", sa.String(length=32), nullable=False),
        sa.Column("latency_ms", sa.Float(), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("NOW()")
        ),
        sa.CheckConstraint(
            "event_type IN ('interview.security.assessed', "
            "'interview.security.blocked', "
            "'interview.security.output_leakage_blocked', "
            "'interview.security.repeated_attempt', "
            "'interview.security.session_flagged')",
            name="ck_interview_security_events_type",
        ),
        sa.CheckConstraint(
            "confidence_band IN ('low', 'medium', 'high')",
            name="ck_interview_security_events_confidence_band",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_interview_security_events"),
        sa.UniqueConstraint(
            "session_id", "turn_id", "event_type", name="uq_interview_security_event"
        ),
    )
    op.create_index(
        "ix_interview_security_events_session_id",
        "interview_security_events",
        ["session_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_interview_security_events_session_id", table_name="interview_security_events")
    op.drop_table("interview_security_events")
    op.drop_column("interview_sessions", "session_security_flagged")
    for name in (
        "output_guard_version",
        "security_prompt_version",
        "security_rules_version",
        "security_policy_version",
    ):
        op.drop_column("interview_sessions", name)
    op.drop_constraint(
        "ck_interview_configs_security_max_attempts", "interview_configs", type_="check"
    )
    op.drop_constraint(
        "ck_interview_configs_security_response_policy", "interview_configs", type_="check"
    )
    op.drop_column("interview_configs", "security_incident_summary_enabled")
    op.drop_column("interview_configs", "security_custom_refusal_vi")
    op.drop_column("interview_configs", "security_custom_refusal_en")
    op.drop_column("interview_configs", "security_max_consecutive_attempts")
    op.drop_column("interview_configs", "security_response_policy")
