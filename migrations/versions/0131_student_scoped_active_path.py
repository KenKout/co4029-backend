"""Enforce one active attempt per (student, career path) in the database.

A student must not run the same Career Path through two Programs at once. That
rule has only ever lived in service code (``find_active_path_attempt_elsewhere``),
and the existing index cannot back it up:

    uq_program_path_attempts_active_path
    UNIQUE (program_enrollment_id, career_path_id) WHERE status = 'active'

It is scoped to ONE enrollment. Two concurrent selections in *different*
enrollments each read a clean state, each pass the service check, and both
insert — neither transaction can see the other's uncommitted row and no
constraint spans them. Acceptance item 10 of the gap review is unachievable
until an index is keyed on the student.

``student_id`` lives on ``program_enrollments``, one table away, so the index
needs the column denormalized onto the attempt. The danger of a denormalized
key is drift, so it is not merely copied: a composite foreign key ties the
pair back to its enrollment,

    (program_enrollment_id, student_id) -> program_enrollments (id, student_id)

which makes an attempt whose student disagrees with its enrollment's student
impossible to insert rather than merely unlikely. That requires a UNIQUE on
``program_enrollments (id, student_id)`` as the FK target; it is redundant
against the primary key, and it is what Postgres demands to reference the pair.

The older per-enrollment index is KEPT. The new one subsumes it logically, but
the two map to different messages and the narrower name is already registered
to ``path_already_selected``. Both messages stay true whichever fires.

Duplicates are reconciled before the index is built, since a unique index
cannot be created over data that violates it. The survivor is the earliest
attempt; the rest are closed, never deleted, and every reconciliation is
printed.

Revision ID: 0131_student_scoped_active_path
Revises: 0130_nullable_career_path_limit
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0131_student_scoped_active_path"
down_revision = "0130_nullable_career_path_limit"
branch_labels = None
depends_on = None

_INDEX = "uq_program_path_attempts_active_student_path"
_FK = "fk_program_path_attempts_enrollment_student"
_ENROLLMENT_PAIR = "uq_program_enrollments_id_student"


def upgrade() -> None:
    bind = op.get_bind()

    # -- 1. denormalize the student onto the attempt -----------------------
    op.add_column(
        "program_path_attempts",
        sa.Column("student_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.execute(
        """
        UPDATE program_path_attempts AS a
        SET student_id = e.student_id
        FROM program_enrollments AS e
        WHERE e.id = a.program_enrollment_id
        """
    )
    # The FK to program_enrollments guarantees every attempt has one, so a
    # surviving NULL here would mean the backfill itself was wrong.
    orphans = bind.scalar(
        sa.text("SELECT COUNT(*) FROM program_path_attempts WHERE student_id IS NULL")
    )
    if orphans:
        raise RuntimeError(
            f"{orphans} path attempt(s) could not be matched to an enrollment. "
            "Resolve those rows before enforcing the student-scoped index."
        )
    op.alter_column("program_path_attempts", "student_id", nullable=False)

    # -- 2. make drift impossible, not just unlikely -----------------------
    op.create_unique_constraint(
        _ENROLLMENT_PAIR, "program_enrollments", ["id", "student_id"]
    )
    op.create_foreign_key(
        _FK,
        "program_path_attempts",
        "program_enrollments",
        ["program_enrollment_id", "student_id"],
        ["id", "student_id"],
        ondelete="NO ACTION",
    )

    # -- 3. reconcile duplicates so the index can be built -----------------
    # Survivor: the earliest attempt. It has had the longest to accumulate
    # entitlements and progress, and "first claim wins" is a rule a student
    # can be told. selected_at first because it is what the student saw;
    # created_at and id only break exact ties, so the choice is deterministic
    # and the migration is repeatable.
    losers_sql = """
        SELECT id, student_id, career_path_id, program_enrollment_id, keep_id
        FROM (
            SELECT
                id,
                student_id,
                career_path_id,
                program_enrollment_id,
                row_number() OVER w AS rn,
                first_value(id) OVER w AS keep_id
            FROM program_path_attempts
            WHERE status = 'active'
            WINDOW w AS (
                PARTITION BY student_id, career_path_id
                ORDER BY selected_at, created_at, id
            )
        ) AS ranked
        WHERE rn > 1
    """
    duplicates = bind.execute(sa.text(losers_sql)).fetchall()

    for attempt_id, student_id, path_id, enrollment_id, kept_id in duplicates:
        print(  # noqa: T201
            f"reconciling duplicate active path: attempt {attempt_id} "
            f"(student {student_id}, path {path_id}, enrollment {enrollment_id}) "
            f"-> switched_out; keeping {kept_id}"
        )

    if duplicates:
        # Closed, not deleted. Progress snapshots are untouched: they record
        # work the student actually did and stay true whichever attempt
        # survives. The reason rides on the row itself so the audit does not
        # depend on anyone having read this migration.
        bind.execute(
            sa.text(
                f"""
                UPDATE program_path_attempts AS a
                SET status = 'switched_out',
                    ended_at = COALESCE(a.ended_at, now()),
                    exit_snapshot = COALESCE(a.exit_snapshot, '{{}}'::jsonb)
                        || jsonb_build_object(
                            'reconciled_by', '{revision}',
                            'reason', 'duplicate_active_path_across_programs',
                            'kept_attempt_id', losers.keep_id
                        ),
                    updated_at = now()
                FROM ({losers_sql}) AS losers
                WHERE losers.id = a.id
                """
            )
        )

    # -- 4. the constraint the rule actually needed ------------------------
    op.create_index(
        _INDEX,
        "program_path_attempts",
        ["student_id", "career_path_id"],
        unique=True,
        postgresql_where=sa.text("status = 'active'"),
    )


def downgrade() -> None:
    op.drop_index(_INDEX, table_name="program_path_attempts")
    op.drop_constraint(_FK, "program_path_attempts", type_="foreignkey")
    op.drop_constraint(_ENROLLMENT_PAIR, "program_enrollments", type_="unique")
    op.drop_column("program_path_attempts", "student_id")
    # Reconciled attempts are deliberately NOT reopened. Reactivating them
    # would recreate the duplicates this migration existed to remove, and the
    # exit_snapshot on each row still says what happened and why.
