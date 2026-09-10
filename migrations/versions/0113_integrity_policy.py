"""Browser-integrity scoring policy + single-source custom refusal.

Two policy changes land together because both reshape the Security & Integrity
settings contract (FR-5.8 extension, cohort-fairness decision 2026-09-10):

1. Custom safe refusal collapses to ONE source: ``security_custom_refusal_en``.
   The Vietnamese column is dropped. A learner interviewing in Vietnamese now
   receives the teacher's English custom wording verbatim (their own choice of
   text, their own responsibility for its language); the platform fallback
   (no custom text) stays locale-specific inside ``safe_security_response``.
   The backfill is deliberate, not a blind copy: a config that only ever had a
   Vietnamese wording keeps that wording as its English-source text rather than
   silently losing the teacher's authored refusal.

2. Browser-integrity policy is scored server-side. Four new config columns
   (three event weights on a 1..5 scale + the warning threshold, 1..20) are
   frozen after publish like every other conduct knob — the default-freeze
   whitelist in ``services/published_freeze.py`` already covers them. Sessions
   persist an IMMUTABLE policy snapshot at start (``integrity_policy_snapshot``
   jsonb) plus the running score / warning state, so a config edited mid-cohort
   (draft → republish) never rewrites the rules an earlier attempt was scored
   under — the same fairness reasoning as the security provenance columns.

``assessment_integrity_events`` gains a nullable ``client_event_id`` with a
partial unique index over (interview_session_id, client_event_id): the browser
reporter retries on network loss, and without the index a retry double-scores
the same physical tab switch. Partial because reconnect/disconnect events (and
server-generated warning rows) legitimately carry no client id.

Verified before writing: 23 dev rows / 4 test rows in ``interview_configs``,
none with a custom refusal set — the backfill matches 0 rows today; it exists
so a future legacy row cannot lose its wording. No duplicate
(interview_session_id, client_event_id) pairs can pre-exist because the column
is new.

Revision ID: 0113_integrity_policy
Revises: 0112_remove_legacy_ownership
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "0113_integrity_policy"
down_revision = "0112_remove_legacy_ownership"
branch_labels = None
depends_on = None

_CLIENT_EVENT_INDEX = "uq_integrity_events_session_client"


def upgrade() -> None:
    # --- 1. single-source custom refusal -------------------------------------
    # Legacy Vietnamese-only wording becomes the English source. COALESCE keeps
    # an existing English text (both-languages rows keep EN) instead of
    # overwriting it.
    op.execute(
        "UPDATE interview_configs SET security_custom_refusal_en = "
        "COALESCE(security_custom_refusal_en, security_custom_refusal_vi) "
        "WHERE security_custom_refusal_vi IS NOT NULL"
    )
    op.drop_column("interview_configs", "security_custom_refusal_vi")

    # --- 2. browser-integrity policy knobs (frozen after publish by default) --
    op.add_column(
        "interview_configs",
        sa.Column(
            "integrity_weight_tab_switch",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("3"),
        ),
    )
    op.add_column(
        "interview_configs",
        sa.Column(
            "integrity_weight_focus_lost",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("1"),
        ),
    )
    op.add_column(
        "interview_configs",
        sa.Column(
            "integrity_weight_fullscreen_exit",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("2"),
        ),
    )
    op.add_column(
        "interview_configs",
        sa.Column(
            "integrity_score_threshold",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("3"),
        ),
    )
    op.create_check_constraint(
        "ck_interview_configs_integrity_weight_tab_switch",
        "interview_configs",
        "integrity_weight_tab_switch BETWEEN 1 AND 5",
    )
    op.create_check_constraint(
        "ck_interview_configs_integrity_weight_focus_lost",
        "interview_configs",
        "integrity_weight_focus_lost BETWEEN 1 AND 5",
    )
    op.create_check_constraint(
        "ck_interview_configs_integrity_weight_fullscreen_exit",
        "interview_configs",
        "integrity_weight_fullscreen_exit BETWEEN 1 AND 5",
    )
    op.create_check_constraint(
        "ck_interview_configs_integrity_score_threshold",
        "interview_configs",
        "integrity_score_threshold BETWEEN 1 AND 20",
    )

    # --- 3. per-session scored state ------------------------------------------
    op.add_column(
        "interview_sessions",
        sa.Column(
            "integrity_policy_snapshot",
            JSONB,
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
    )
    op.add_column(
        "interview_sessions",
        sa.Column(
            "integrity_score",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("0"),
        ),
    )
    op.add_column(
        "interview_sessions",
        sa.Column(
            "integrity_warning_issued",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("FALSE"),
        ),
    )
    op.add_column(
        "interview_sessions",
        sa.Column(
            "integrity_threshold_flagged_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
    )

    # --- 4. idempotent client reporting ---------------------------------------
    op.add_column(
        "assessment_integrity_events",
        sa.Column("client_event_id", sa.UUID(), nullable=True),
    )
    op.create_index(
        _CLIENT_EVENT_INDEX,
        "assessment_integrity_events",
        ["interview_session_id", "client_event_id"],
        unique=True,
        postgresql_where=sa.text("client_event_id IS NOT NULL"),
    )


def downgrade() -> None:
    op.drop_index(_CLIENT_EVENT_INDEX, table_name="assessment_integrity_events")
    op.drop_column("assessment_integrity_events", "client_event_id")

    op.drop_column("interview_sessions", "integrity_threshold_flagged_at")
    op.drop_column("interview_sessions", "integrity_warning_issued")
    op.drop_column("interview_sessions", "integrity_score")
    op.drop_column("interview_sessions", "integrity_policy_snapshot")

    op.drop_constraint(
        "ck_interview_configs_integrity_score_threshold",
        "interview_configs",
        type_="check",
    )
    op.drop_constraint(
        "ck_interview_configs_integrity_weight_fullscreen_exit",
        "interview_configs",
        type_="check",
    )
    op.drop_constraint(
        "ck_interview_configs_integrity_weight_focus_lost",
        "interview_configs",
        type_="check",
    )
    op.drop_constraint(
        "ck_interview_configs_integrity_weight_tab_switch",
        "interview_configs",
        type_="check",
    )
    op.drop_column("interview_configs", "integrity_score_threshold")
    op.drop_column("interview_configs", "integrity_weight_fullscreen_exit")
    op.drop_column("interview_configs", "integrity_weight_focus_lost")
    op.drop_column("interview_configs", "integrity_weight_tab_switch")

    # Lossy in the safe direction: the backfilled English text (which may
    # itself be an old Vietnamese wording) is restored as the Vietnamese
    # column, so a downgrade never loses the teacher's authored refusal —
    # it may re-materialise it under the pre-0112 name.
    op.add_column(
        "interview_configs",
        sa.Column("security_custom_refusal_vi", sa.Text(), nullable=True),
    )
    op.execute(
        "UPDATE interview_configs SET security_custom_refusal_vi = "
        "security_custom_refusal_en WHERE security_custom_refusal_en IS NOT NULL"
    )
