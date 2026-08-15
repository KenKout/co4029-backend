"""Drop the unused 'pass' option from career_course_items.satisfied_by.

``'pass'`` has no evaluator anywhere in the codebase: ``satisfied`` is
always ``course_enrollments.status = 'completed'`` (queries/student.py),
no UI can set the value, and the model docstring documents it as reserved
for a "graded variant later" that never shipped. The CHECK accepted it
for forward compatibility; this migration reclaims the schema so the DB
rejects a value nothing can consume.

Data note: no row in the database carries ``'pass'`` (verified before
authoring), so the column value is untouched — only the CHECK narrows.
"""

from __future__ import annotations

from alembic import op

# NOTE: alembic_version.version_num is varchar(32); keep this id <= 32 chars.
revision = "0073_drop_satisfied_by_pass"
down_revision = "0072_remove_interview_practice"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_constraint(
        "career_course_items_satisfied_by_check",
        "career_course_items",
        type_="check",
    )
    op.create_check_constraint(
        "career_course_items_satisfied_by_check",
        "career_course_items",
        "satisfied_by IN ('completion')",
    )


def downgrade() -> None:
    op.drop_constraint(
        "career_course_items_satisfied_by_check",
        "career_course_items",
        type_="check",
    )
    op.create_check_constraint(
        "career_course_items_satisfied_by_check",
        "career_course_items",
        "satisfied_by IN ('completion','pass')",
    )
