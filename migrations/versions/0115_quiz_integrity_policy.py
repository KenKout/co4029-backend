"""Browser-integrity scoring for quiz attempts (FR-5.8 parity with interviews).

Quiz attempts already COLLECTED browser signals — tab switch, focus loss,
fullscreen exit — into ``assessment_integrity_events`` with
``assessment_kind='quiz'``. Nothing ever read them except a read-only timeline
on the teacher's attempt-detail page: no weights, no threshold, no score, no
warning. The interview side has had all of that since 0113, so the same signal
meant "flagged, risk: high" in one assessment and nothing at all in the other.

This migration ports 0113's design to quizzes, deliberately unchanged:

1. ``quizzes`` gains the three event weights (1..5) and the warning threshold
   (1..20), same defaults as ``interview_configs`` — a signal should not be
   worth more in one assessment kind than the other.

2. ``quiz_attempts`` gains an IMMUTABLE policy snapshot frozen at attempt
   start, plus the running score and the one-shot warning state. The snapshot
   is the cohort-fairness guarantee: a teacher who edits the weights halfway
   through a cohort never re-scores an attempt that was taken under the older
   rules.

3. The idempotency index is EXTENDED to quizzes. 0113 added
   ``uq_integrity_events_session_client`` over (interview_session_id,
   client_event_id); the quiz reporter needs the same protection over
   (quiz_attempt_id, client_event_id), because the browser retries a batch
   after a network loss and without it one physical tab switch scores twice.
   A separate partial index rather than a widened one: the two columns are
   mutually exclusive per row (one is always NULL), so a single index over
   both would not constrain anything.

Backfill: none, and none is possible. Existing attempts have no snapshot, so
they score under the shipped defaults if any late event arrives — but events
are only accepted while an attempt is ``in_progress``, so in practice a closed
attempt keeps a score of 0. Their stored events remain readable exactly as
before; they simply carry no score, which is the truthful representation of
"this attempt was never scored" rather than inventing one retroactively.

Revision ID: 0115_quiz_integrity_policy
Revises: 0114_drop_quiz_browser_security
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "0115_quiz_integrity_policy"
down_revision = "0114_drop_quiz_browser_security"
branch_labels = None
depends_on = None

_QUIZ_CLIENT_EVENT_INDEX = "uq_integrity_events_attempt_client"

# (column, default, low, high) — the bounds mirror interview_configs exactly.
_WEIGHT_COLUMNS = (
    ("integrity_weight_tab_switch", 3, 1, 5),
    ("integrity_weight_focus_lost", 1, 1, 5),
    ("integrity_weight_fullscreen_exit", 2, 1, 5),
    ("integrity_score_threshold", 3, 1, 20),
)


def upgrade() -> None:
    # --- 1. per-quiz policy ---------------------------------------------------
    for column, default, low, high in _WEIGHT_COLUMNS:
        op.add_column(
            "quizzes",
            sa.Column(
                column,
                sa.Integer(),
                nullable=False,
                server_default=sa.text(str(default)),
            ),
        )
        op.create_check_constraint(
            f"ck_quizzes_{column}",
            "quizzes",
            f"{column} BETWEEN {low} AND {high}",
        )

    # --- 2. per-attempt snapshot + running state ------------------------------
    op.add_column(
        "quiz_attempts",
        sa.Column(
            "integrity_policy_snapshot",
            JSONB(),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
    )
    op.add_column(
        "quiz_attempts",
        sa.Column(
            "integrity_score",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("0"),
        ),
    )
    op.add_column(
        "quiz_attempts",
        sa.Column(
            "integrity_warning_issued",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("FALSE"),
        ),
    )
    op.add_column(
        "quiz_attempts",
        sa.Column(
            "integrity_threshold_flagged_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
    )

    # --- 3. idempotent client reporting for quiz attempts ---------------------
    op.create_index(
        _QUIZ_CLIENT_EVENT_INDEX,
        "assessment_integrity_events",
        ["quiz_attempt_id", "client_event_id"],
        unique=True,
        postgresql_where=sa.text("client_event_id IS NOT NULL"),
    )


def downgrade() -> None:
    op.drop_index(_QUIZ_CLIENT_EVENT_INDEX, table_name="assessment_integrity_events")
    op.drop_column("quiz_attempts", "integrity_threshold_flagged_at")
    op.drop_column("quiz_attempts", "integrity_warning_issued")
    op.drop_column("quiz_attempts", "integrity_score")
    op.drop_column("quiz_attempts", "integrity_policy_snapshot")
    for column, _default, _low, _high in reversed(_WEIGHT_COLUMNS):
        op.drop_constraint(f"ck_quizzes_{column}", "quizzes", type_="check")
        op.drop_column("quizzes", column)
