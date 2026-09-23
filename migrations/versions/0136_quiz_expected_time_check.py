"""Allow the uncalibrated expected-time sentinel.

The T5.1 migration made ``quiz_questions.expected_response_time_ms`` nullable,
but the baseline ``> 0`` CHECK survived the column rename.  That left the
schema stricter than the publish/query contract: draft questions may use NULL
or zero while the publish gate rejects both as unusable.

Revision ID: 0136_quiz_expected_time_check
Revises: 0135_interview_start_idem
"""

from __future__ import annotations

from alembic import op

revision = "0136_quiz_expected_time_check"
down_revision = "0135_interview_start_idem"
branch_labels = None
depends_on = None

_CONSTRAINT = "quiz_questions_expected_response_ms_check"


def upgrade() -> None:
    op.drop_constraint(_CONSTRAINT, "quiz_questions", type_="check")
    op.create_check_constraint(
        _CONSTRAINT,
        "quiz_questions",
        "expected_response_time_ms IS NULL OR expected_response_time_ms >= 0",
    )


def downgrade() -> None:
    # Existing NULL/zero sentinel values cannot satisfy the original baseline
    # constraint, so restore a valid default before narrowing the constraint.
    op.execute(
        "UPDATE quiz_questions "
        "SET expected_response_time_ms = 60000 "
        "WHERE expected_response_time_ms IS NULL OR expected_response_time_ms = 0"
    )
    op.drop_constraint(_CONSTRAINT, "quiz_questions", type_="check")
    op.create_check_constraint(
        _CONSTRAINT,
        "quiz_questions",
        "expected_response_time_ms > 0",
    )
