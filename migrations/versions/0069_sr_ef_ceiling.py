"""Cap ``student_card_state.ef`` at 2.5 (upper bound for SM-2 EF).

Revision ID: 0069_sr_ef_ceiling
Revises: 0068_quiz_match_distract
Create Date: 2026-08-03

Why
---
``update_ef`` only floored EF at 1.3, so a run of perfect reviews drifted EF
past 2.5 (2.6, 2.7, ...). Every EF consumer in the codebase assumes 2.5 as
the maximum — the initial EF, the lesson-unlock range (1.3..2.5), and the
KR normalisation denominator (2.5 - 1.3) in ``kr_estimate.sql`` /
``class_kr_distribution.sql`` — so an uncapped EF made the KR estimate
``(EF - 1.3) / (2.5 - 1.3)`` exceed 1.0 (a ">100%" retention figure). The
ceiling keeps EF in [1.3, 2.5] and KR in [0, 1] by construction.

Steps
-----
1. Clamp any existing over-limit rows to 2.5 (dev data only; the code path
   now caps on every write, so this is a one-time repair).
2. Replace the floor-only CHECK with the full range ``ef >= 1.3 AND
   ef <= 2.5`` (constraint renamed to ``ck_student_card_state_ef_range`` to
   match the widened semantics).

Reversible: downgrade restores the floor-only constraint and re-exports the
clamped rows untouched (the ceiling is enforced by code after this
migration, so dropping the CHECK alone cannot un-clamp anything).
"""

from __future__ import annotations

from alembic import op

# NOTE: alembic_version.version_num is varchar(32); keep revision id <= 32 chars.
revision = "0069_sr_ef_ceiling"
down_revision = "0068_quiz_match_distract"
branch_labels = None
depends_on = None

_TABLE = "student_card_state"
_OLD_CONSTRAINT = "ck_student_card_state_ef_floor"
_NEW_CONSTRAINT = "ck_student_card_state_ef_range"


def upgrade() -> None:
    # One-time repair of rows written by the uncapped code path.
    op.execute(f"UPDATE {_TABLE} SET ef = 2.5 WHERE ef > 2.5")
    op.execute(f"ALTER TABLE {_TABLE} DROP CONSTRAINT {_OLD_CONSTRAINT}")
    op.execute(
        f"ALTER TABLE {_TABLE} ADD CONSTRAINT {_NEW_CONSTRAINT} "
        "CHECK (ef >= 1.3 AND ef <= 2.5)"
    )


def downgrade() -> None:
    op.execute(f"ALTER TABLE {_TABLE} DROP CONSTRAINT {_NEW_CONSTRAINT}")
    op.execute(
        f"ALTER TABLE {_TABLE} ADD CONSTRAINT {_OLD_CONSTRAINT} "
        "CHECK (ef >= 1.3)"
    )
