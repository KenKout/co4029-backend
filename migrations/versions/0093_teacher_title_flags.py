"""teacher title flags

User decision 2026-08-30: a course may have MULTIPLE Course Instructors and
multiple Teacher Assistants, and one teacher may hold BOTH titles on the
same course. The single ``course_role`` column and the
``uq_course_teachers_one_instructor`` partial index (at most one CI) cannot
express that.

Replace the scalar with two independent booleans on the SAME assignment row
(one row per teacher, no duplicate rows):

* ``is_instructor``  — holds the Course Instructor title
* ``is_assistant``   — holds the Teacher Assistant title

Both true = "both titles". At least one must be true for a course-scoped
teacher row (CHECK below, same spirit as the old ``course_role`` enum
check). The at-least-one-instructor invariant moves fully into the
assignment service (a course with teachers keeps >= 1 instructor), since a
CHECK cannot enforce ">= 1 instructor across rows" without a subquery.
"""

import sqlalchemy as sa
from alembic import op

revision = "0093_teacher_title_flags"
down_revision = "0092_iq_bank_variant_groups"
branch_labels = None
depends_on = None

_TABLE = "user_role_assignments"

_BACKFILL_INSTRUCTOR = sa.text(
    "UPDATE user_role_assignments SET is_instructor = true "
    "WHERE course_role = 'course_instructor'"
)
_BACKFILL_ASSISTANT = sa.text(
    "UPDATE user_role_assignments SET is_assistant = true "
    "WHERE course_role = 'teacher_assistant'"
)
# Catch-all: legacy course-scoped rows that never got a course_role (e.g.
# soft-deleted rows 0079 deliberately skipped). The CHECK below requires
# every course-scoped row to hold >= 1 title, so sweep them to assistant.
_BACKFILL_TITELESS = sa.text(
    "UPDATE user_role_assignments SET is_assistant = true "
    "WHERE scope_kind = 'course' AND course_id IS NOT NULL "
    "  AND NOT is_instructor AND NOT is_assistant"
)


def upgrade() -> None:
    op.add_column(
        _TABLE,
        sa.Column("is_instructor", sa.Boolean(), nullable=False, server_default=sa.text("false")),
    )
    op.add_column(
        _TABLE,
        sa.Column("is_assistant", sa.Boolean(), nullable=False, server_default=sa.text("false")),
    )
    op.execute(_BACKFILL_INSTRUCTOR)
    op.execute(_BACKFILL_ASSISTANT)
    op.execute(_BACKFILL_TITELESS)

    # Course-scoped teacher rows must carry at least one title. The old enum
    # CHECK (``course_role IS NULL OR course_role IN (...)``) only constrained
    # values when present; this one makes "no title on a course row" invalid.
    op.create_check_constraint(
        "ck_user_role_assignments_course_title",
        _TABLE,
        "(scope_kind <> 'course') OR is_instructor OR is_assistant",
    )

    op.drop_index("uq_course_teachers_one_instructor", table_name=_TABLE)
    op.drop_constraint("ck_user_role_assignments_course_role", _TABLE, type_="check")
    op.drop_column(_TABLE, "course_role")


def downgrade() -> None:
    op.add_column(_TABLE, sa.Column("course_role", sa.String(30), nullable=True))
    op.execute(
        "UPDATE user_role_assignments SET course_role = 'course_instructor' "
        "WHERE is_instructor = true"
    )
    op.execute(
        "UPDATE user_role_assignments SET course_role = 'teacher_assistant' "
        "WHERE course_role IS NULL AND is_assistant = true"
    )
    op.create_check_constraint(
        "ck_user_role_assignments_course_role",
        _TABLE,
        "course_role IS NULL OR course_role IN ('course_instructor', 'teacher_assistant')",
    )
    op.create_index(
        "uq_course_teachers_one_instructor",
        _TABLE,
        ["course_id"],
        unique=True,
        postgresql_where=sa.text(
            "scope_kind = 'course' AND course_role = 'course_instructor' "
            "AND deleted_at IS NULL AND active_until IS NULL"
        ),
    )
    op.drop_constraint("ck_user_role_assignments_course_title", _TABLE, type_="check")
    op.drop_column(_TABLE, "is_instructor")
    op.drop_column(_TABLE, "is_assistant")
