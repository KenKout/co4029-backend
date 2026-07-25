"""Learning-outcome hierarchy: self-referential parent_id + per-parent position.

Revision ID: 0059_lo_hierarchy
Revises: 0058_interview_gen_notif
Create Date: 2026-07-25

Adds arbitrary-depth hierarchy to ``course_learning_outcomes`` (L.O.1 →
L.O.1.1 → L.O.1.1.1 …). Additive/reversible:

* ``parent_id`` self-FK (nullable → top-level). Existing rows stay
  top-level (parent_id IS NULL), so display codes are unchanged for
  current data.
* ``position`` semantics widen from "unique within course" to "unique
  within parent (siblings)". The old ``UNIQUE(course_id, position)``
  constraint is replaced by an expression unique index over
  ``COALESCE(parent_id, course_id)`` so top-level rows group by course
  and children group by their parent — one index covers both cases and
  Postgres does not treat the COALESCE result as NULL-distinct.

The running app (whose ORM does not yet map parent_id) keeps working:
the column is nullable and the new index is satisfied by the existing
(course_id, position) uniqueness of top-level rows.
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID as PGUUID

# NOTE: alembic_version.version_num is varchar(32); keep revision id <= 32 chars.
revision = "0059_lo_hierarchy"
down_revision = "0058_interview_gen_notif"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "course_learning_outcomes",
        sa.Column("parent_id", PGUUID(as_uuid=True), nullable=True),
    )
    op.create_foreign_key(
        "fk_course_learning_outcomes_parent",
        "course_learning_outcomes",
        "course_learning_outcomes",
        ["parent_id"],
        ["id"],
        ondelete="NO ACTION",
    )
    op.create_index(
        "ix_course_learning_outcomes_parent_id",
        "course_learning_outcomes",
        ["parent_id"],
    )
    # Swap the course-wide position uniqueness for a per-parent one. A single
    # expression index over COALESCE(parent_id, course_id) covers both:
    #   - top-level rows (parent_id NULL) group by course_id
    #   - child rows group by parent_id
    # Partial (deleted_at IS NULL) so soft-deleted rows don't pin a slot.
    # The old course-wide uniqueness is a unique INDEX
    # (course_learning_outcomes_course_id_position_key), auto-created by the
    # baseline UniqueConstraint. Drop whichever name exists (index or named
    # constraint) defensively, then create the per-parent expression index.
    op.execute(
        "ALTER TABLE course_learning_outcomes "
        "DROP CONSTRAINT IF EXISTS uq_course_learning_outcomes_position"
    )
    op.execute(
        "DROP INDEX IF EXISTS course_learning_outcomes_course_id_position_key"
    )
    op.execute(
        "CREATE UNIQUE INDEX uq_course_learning_outcomes_sibling_position "
        "ON course_learning_outcomes (COALESCE(parent_id, course_id), position) "
        "WHERE deleted_at IS NULL"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS uq_course_learning_outcomes_sibling_position")
    op.create_unique_constraint(
        "uq_course_learning_outcomes_position",
        "course_learning_outcomes",
        ["course_id", "position"],
    )
    op.drop_index(
        "ix_course_learning_outcomes_parent_id",
        table_name="course_learning_outcomes",
    )
    op.drop_constraint(
        "fk_course_learning_outcomes_parent",
        "course_learning_outcomes",
        type_="foreignkey",
    )
    op.drop_column("course_learning_outcomes", "parent_id")
