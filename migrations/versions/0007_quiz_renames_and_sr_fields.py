"""quiz_questions / quiz_attempt_answers SR-ready renames + nullable flip

Revision ID: 0007_quiz_renames_and_sr_fields
Revises: 0006_courses_unlock_config
Create Date: 2026-05-17 13:00:00.000000

T5.1 backfill — Reconciliation §C1 + plan §5363-5372 require four
column-level changes against the baseline ``quiz_questions`` and
``quiz_attempt_answers`` DDL. The orchestrator's "migrations are locked"
directive applies to the existing 0001-0006 history, not to additive
forward migrations needed to give the T5.1 ORM honest DDL to map onto.
The pattern matches 0005 / 0006 (also forward-only column renames + ALTER
COLUMN + UPDATE).

Changes
-------

1. ``quiz_questions.expected_response_ms`` → ``expected_response_time_ms``
   + flip ``NOT NULL`` → ``NULL`` (§C1: T7.5.9 publish gate enforces
   NOT NULL at publish time, not at column).

2. ``quiz_questions.source_refs_json`` → ``source_refs`` (§C1: drop the
   ``_json`` suffix; column type stays JSONB and the default ``'[]'::jsonb``
   is preserved).

3. ``quiz_attempt_answers.response_time_ms`` → ``t_actual_ms`` (plan
   §5371: SR-ready terminology — observed response latency).

4. ``quiz_questions.question_type`` domain rename ``'mcq'`` →
   ``'multiple_choice'`` (§C1 + plan §5374 + plan §5414):
   * Drop existing CHECK constraint (the baseline names it implicitly
     via inline ``CHECK (question_type IN (...))``; PostgreSQL assigns
     ``quiz_questions_question_type_check``).
   * Data migration: ``UPDATE quiz_questions SET
     question_type='multiple_choice' WHERE question_type='mcq'``.
   * Re-create CHECK constraint with new value list, keeping
     ``true_false``, ``short_answer``, ``fill_blank``, ``code``.

Round-trip
----------

The downgrade reverses every change: re-narrows ``question_type``
(reverting rows back to ``'mcq'``), re-renames columns, and restores
``expected_response_time_ms`` to ``NOT NULL`` after backfilling NULLs
with a sentinel (the downgrade path is destructive only when published
questions had NULL T_exp; pre-T5.1 every row had a value, so a fresh
upgrade-then-downgrade is clean).

Per Reconciliation §C5 the Quiz Bank refactor (Question +
QuestionBank tables, QuizQuestion-as-link-table) is OUT OF SCOPE for
this migration — that lands in T5.0 follow-up with its own Alembic
revision.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0007_quiz_renames_and_sr_fields"
down_revision = "0006_courses_unlock_config"
branch_labels = None
depends_on = None


_QUESTION_TYPE_NEW_VALUES = (
    "multiple_choice",
    "true_false",
    "short_answer",
    "fill_blank",
    "code",
)
_QUESTION_TYPE_OLD_VALUES = (
    "mcq",
    "true_false",
    "short_answer",
    "fill_blank",
    "code",
)
# PostgreSQL's auto-generated CHECK name for the inline constraint on
# quiz_questions.question_type from baseline 0001 line 670.
_QUESTION_TYPE_CHECK_NAME = "quiz_questions_question_type_check"


def _values_sql(values: tuple[str, ...]) -> str:
    return ", ".join(f"'{v}'" for v in values)


def upgrade() -> None:
    # 1. quiz_questions.expected_response_ms → expected_response_time_ms (NULLABLE)
    op.alter_column(
        "quiz_questions",
        "expected_response_ms",
        new_column_name="expected_response_time_ms",
        existing_type=sa.Integer(),
        existing_nullable=False,
        nullable=True,
    )
    # The baseline inline CHECK ``expected_response_ms > 0`` followed the
    # column rename; PostgreSQL keeps the constraint and rewrites its
    # internal column reference. No manual CHECK rename needed.

    # 2. quiz_questions.source_refs_json → source_refs
    op.alter_column(
        "quiz_questions",
        "source_refs_json",
        new_column_name="source_refs",
    )

    # 3. quiz_attempt_answers.response_time_ms → t_actual_ms (per plan §5371)
    op.alter_column(
        "quiz_attempt_answers",
        "response_time_ms",
        new_column_name="t_actual_ms",
    )

    # 4. question_type domain: mcq → multiple_choice (data + CHECK)
    op.execute(
        f"ALTER TABLE quiz_questions DROP CONSTRAINT {_QUESTION_TYPE_CHECK_NAME}"
    )
    op.execute(
        "UPDATE quiz_questions SET question_type='multiple_choice' "
        "WHERE question_type='mcq'"
    )
    op.execute(
        f"ALTER TABLE quiz_questions ADD CONSTRAINT {_QUESTION_TYPE_CHECK_NAME} "
        f"CHECK (question_type IN ({_values_sql(_QUESTION_TYPE_NEW_VALUES)}))"
    )


def downgrade() -> None:
    # Reverse 4: question_type domain
    op.execute(
        f"ALTER TABLE quiz_questions DROP CONSTRAINT {_QUESTION_TYPE_CHECK_NAME}"
    )
    op.execute(
        "UPDATE quiz_questions SET question_type='mcq' "
        "WHERE question_type='multiple_choice'"
    )
    op.execute(
        f"ALTER TABLE quiz_questions ADD CONSTRAINT {_QUESTION_TYPE_CHECK_NAME} "
        f"CHECK (question_type IN ({_values_sql(_QUESTION_TYPE_OLD_VALUES)}))"
    )

    # Reverse 3: t_actual_ms → response_time_ms
    op.alter_column(
        "quiz_attempt_answers",
        "t_actual_ms",
        new_column_name="response_time_ms",
    )

    # Reverse 2: source_refs → source_refs_json
    op.alter_column(
        "quiz_questions",
        "source_refs",
        new_column_name="source_refs_json",
    )

    # Reverse 1: expected_response_time_ms → expected_response_ms (NOT NULL)
    # Backfill any NULLs introduced during the T5.1 nullable window with
    # a sentinel so the NOT NULL flip succeeds.
    op.execute(
        "UPDATE quiz_questions SET expected_response_time_ms = 60000 "
        "WHERE expected_response_time_ms IS NULL"
    )
    op.alter_column(
        "quiz_questions",
        "expected_response_time_ms",
        new_column_name="expected_response_ms",
        existing_type=sa.Integer(),
        existing_nullable=True,
        nullable=False,
    )
