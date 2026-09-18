"""Let a program decline to cap career paths, deferring to the org limit.

``learning_program_versions.max_career_paths_per_enrollment`` was NOT NULL with
a server default of 1, so every program had a per-program cap whether its
manager wanted one or not — and the only way to say "as many as the student is
allowed" was to type the maximum, which stops meaning that the moment the
maximum moves.

NULL now means the program sets no cap of its own. Such a student is bounded by
``learning_program.max_concurrent_paths_per_student`` (counted across every
program they are in) and by how many paths the program actually offers.

The existing CHECK is left alone: ``NULL BETWEEN 1 AND 10`` evaluates to NULL,
which a CHECK constraint accepts, so a nullable column needs no new constraint
to keep 1..10 for the rows that do set a value.

Existing values are NOT migrated. A program currently capped at 1 was capped at
1 by the old default rather than by a decision, but rewriting those to NULL
would widen live programs without anyone asking — a manager can clear the field
when they mean to. Only the default for FUTURE rows changes.

Revision ID: 0130_nullable_career_path_limit
Revises: 0129_drop_career_path_ceiling_setting
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0130_nullable_career_path_limit"
down_revision = "0129_drop_career_path_ceiling_setting"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column(
        "learning_program_versions",
        "max_career_paths_per_enrollment",
        existing_type=sa.Integer(),
        nullable=True,
        server_default=None,
    )


def downgrade() -> None:
    # Backfill before restoring NOT NULL. 1 is what the old server default
    # would have written, and it is the safest direction: it narrows an
    # uncapped program to a single path rather than silently granting every
    # student of it the organization maximum.
    op.execute(
        "UPDATE learning_program_versions "
        "SET max_career_paths_per_enrollment = 1 "
        "WHERE max_career_paths_per_enrollment IS NULL"
    )
    op.alter_column(
        "learning_program_versions",
        "max_career_paths_per_enrollment",
        existing_type=sa.Integer(),
        nullable=False,
        server_default=sa.text("1"),
    )
