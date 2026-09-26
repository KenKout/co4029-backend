"""Require positive calibrated quiz expected times.

Revision ID: 0138_quiz_texp_positive
Revises: 0137_quiz_timing_snapshot
"""

from __future__ import annotations

from alembic import op

revision = "0138_quiz_texp_positive"
down_revision = "0137_quiz_timing_snapshot"
branch_labels = None
depends_on = None

_QUIZ_CONSTRAINT = "quiz_questions_expected_response_ms_check"
_BANK_CONSTRAINT = "ck_quiz_bank_expected_time_positive"


def upgrade() -> None:
    # Zero was previously accepted as an "uncalibrated" sentinel. NULL already
    # represents that draft state and cannot enter the SR scheduler, so
    # normalise legacy non-positive values before tightening the invariant.
    op.execute(
        "UPDATE quiz_questions "
        "SET expected_response_time_ms = NULL "
        "WHERE expected_response_time_ms <= 0"
    )
    op.execute(
        "UPDATE quiz_question_bank_items "
        "SET expected_response_time_ms = NULL "
        "WHERE expected_response_time_ms <= 0"
    )

    op.drop_constraint(_QUIZ_CONSTRAINT, "quiz_questions", type_="check")
    op.create_check_constraint(
        _QUIZ_CONSTRAINT,
        "quiz_questions",
        "expected_response_time_ms IS NULL OR expected_response_time_ms > 0",
    )
    op.create_check_constraint(
        _BANK_CONSTRAINT,
        "quiz_question_bank_items",
        "expected_response_time_ms IS NULL OR expected_response_time_ms > 0",
    )


def downgrade() -> None:
    op.drop_constraint(_BANK_CONSTRAINT, "quiz_question_bank_items", type_="check")
    op.drop_constraint(_QUIZ_CONSTRAINT, "quiz_questions", type_="check")
    op.create_check_constraint(
        _QUIZ_CONSTRAINT,
        "quiz_questions",
        "expected_response_time_ms IS NULL OR expected_response_time_ms >= 0",
    )
