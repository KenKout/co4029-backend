"""Course-scoped teacher title: Course Instructor / Teacher Assistant.

Adds a ``course_role`` column to course-scoped ``user_role_assignments`` so a
manager can title each teacher on a course (``course_instructor`` /
``teacher_assistant``) WITHOUT touching the global role catalog — titles are
per-course, "no catalog logic for titles".

Backfill (so existing courses keep working — the change never breaks them),
in dependency order so no course is left without an instructor:

1. Course OWNER (if they are an active teacher on it) -> ``course_instructor``.
2. Any course still without an instructor (owner not a teacher) -> its
   EARLIEST active teacher becomes ``course_instructor`` (a ``DISTINCT ON``
   over the rows whose ``course_role`` is still NULL, excluding courses that
   already gained an instructor in step 1).
3. Every remaining active course-scoped teacher -> ``teacher_assistant``.

At-least-one is guaranteed by steps 1-2 (a course with teachers always gets an
instructor). At-most-one is enforced later by a partial unique index.

Columns are table-qualified wherever a join introduces ambiguity (both
``user_role_assignments`` and ``courses`` have ``deleted_at``). SQL is plain
(un-formatted) string literals so bandit's S608 never fires.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

# NOTE: alembic_version.version_num is varchar(32); keep revision id <= 32 chars.
revision = "0079_course_teacher_role"
down_revision = "0078_raise_budget_bounds"
branch_labels = None
depends_on = None

_TABLE = "user_role_assignments"

_TEACHER_ROLE = (
    "SELECT id FROM roles WHERE code = 'teacher' AND deleted_at IS NULL LIMIT 1"
)

# Owner -> Course Instructor. All assignment columns are qualified ``a.``
# because ``courses c`` also carries ``deleted_at``.
_OWNER_TO_CI = (
    "UPDATE user_role_assignments a "
    "SET course_role = 'course_instructor' "
    "FROM courses c "
    "WHERE a.course_id = c.id "
    "AND a.scope_kind = 'course' AND a.course_role IS NULL "
    "AND a.role_id = (SELECT id FROM roles WHERE code = 'teacher' "
    "AND deleted_at IS NULL LIMIT 1) "
    "AND a.deleted_at IS NULL AND a.active_until IS NULL "
    "AND a.user_id = c.owner_user_id"
)

# Earliest remaining active teacher -> Course Instructor, only for courses
# that do NOT already have an instructor (owner not a teacher).
_EARLIEST_TO_CI = (
    "UPDATE user_role_assignments a "
    "SET course_role = 'course_instructor' "
    "FROM ("
    "  SELECT DISTINCT ON (u.course_id) u.course_id, u.id "
    "  FROM user_role_assignments u "
    "  WHERE u.scope_kind = 'course' AND u.course_id IS NOT NULL "
    "    AND u.course_role IS NULL "
    "    AND u.role_id = (SELECT id FROM roles WHERE code = 'teacher' "
    "      AND deleted_at IS NULL LIMIT 1) "
    "    AND u.deleted_at IS NULL AND u.active_until IS NULL "
    "    AND NOT EXISTS ("
    "      SELECT 1 FROM user_role_assignments ci "
    "      WHERE ci.course_id = u.course_id "
    "        AND ci.course_role = 'course_instructor' "
    "        AND ci.deleted_at IS NULL AND ci.active_until IS NULL)"
    "  ORDER BY u.course_id, u.active_from, u.id"
    ") sub WHERE a.id = sub.id"
)

# Everything left becomes a Teacher Assistant (no join -> no ambiguity).
_DEFAULT_TA = (
    "UPDATE user_role_assignments "
    "SET course_role = 'teacher_assistant' "
    "WHERE scope_kind = 'course' AND course_id IS NOT NULL "
    "  AND course_role IS NULL "
    "  AND role_id = (SELECT id FROM roles WHERE code = 'teacher' "
    "AND deleted_at IS NULL LIMIT 1) "
    "  AND deleted_at IS NULL AND active_until IS NULL"
)


def upgrade() -> None:
    op.add_column(_TABLE, sa.Column("course_role", sa.String(30), nullable=True))
    op.execute(_OWNER_TO_CI)
    op.execute(_EARLIEST_TO_CI)
    op.execute(_DEFAULT_TA)

    op.create_check_constraint(
        "ck_user_role_assignments_course_role",
        _TABLE,
        "course_role IS NULL OR course_role IN ('course_instructor', 'teacher_assistant')",
    )
    # Row-level guard: at most one active Course Instructor per course.
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


def downgrade() -> None:
    op.drop_index("uq_course_teachers_one_instructor", table_name=_TABLE)
    op.drop_constraint("ck_user_role_assignments_course_role", _TABLE, type_="check")
    op.drop_column(_TABLE, "course_role")
