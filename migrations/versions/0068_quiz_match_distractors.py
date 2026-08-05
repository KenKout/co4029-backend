"""Add ``quiz_questions.match_distractors`` for matching-question distractors.

Revision ID: 0068_quiz_match_distractors
Revises: 0067_lo_manage_perm
Create Date: 2026-08-02

A classic matching question is 1:1 — every right-hand value is the correct
partner of exactly one left prompt, so once a learner has matched all but one
prompt the last pairing is forced. Distractors fix that: they are extra
right-side values with NO left partner, added to the shuffled choice pool the
learner picks from. They make the question genuinely discriminating without
touching the answer key.

Storage
-------
A single nullable ``jsonb`` column holding a JSON array of strings, mirroring
how ``match_pairs`` / ``ordering_sequence`` are stored (raw JSONB, coerced by
the service layer). NULL or ``[]`` means "no distractors" — i.e. classic 1:1
matching, so every existing question keeps its exact current behaviour.

Grading is unaffected: ``_grade_matching`` only ever compares a learner's
chosen right value against each pair's stored ``right``. A distractor is never
the ``right`` of any pair, so selecting one can only ever be wrong — no grader
change is required for correctness, only the student-facing choice projection
grows.

Additive and fully reversible.
"""

from __future__ import annotations

from alembic import op

# NOTE: alembic_version.version_num is varchar(32); keep revision id <= 32 chars.
revision = "0068_quiz_match_distract"
down_revision = "0067_lo_manage_perm"
branch_labels = None
depends_on = None

_TABLE = "quiz_questions"
_COLUMN = "match_distractors"


def upgrade() -> None:
    op.execute(f"ALTER TABLE {_TABLE} ADD COLUMN {_COLUMN} jsonb")


def downgrade() -> None:
    op.execute(f"ALTER TABLE {_TABLE} DROP COLUMN {_COLUMN}")
