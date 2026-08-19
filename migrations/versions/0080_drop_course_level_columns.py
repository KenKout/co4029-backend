"""Drop the course \u201clevel\u201d and \u201cexpected_completion_days\u201d columns.

User decision 2026-08-18: the course level is no longer user-defined — it is
DERIVED from the course's career-path placement (shown on the student view as
\u201cStage N \u2014 <title>\u201d). `expected_completion_days` was removed from the product
entirely. Both columns (and the `level` CHECK constraint) are dropped, along
with their ORM mappings, so nothing can read a value that no longer exists.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0080_drop_course_level_columns"
down_revision = "0079_course_teacher_role"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # The baseline template names the checks <table>_<col>_check (not the ORM's
    # ck_courses_level); drop them by their actual DB names, then the columns.
    op.drop_constraint("courses_level_check", "courses", type_="check")
    op.drop_constraint("courses_expected_completion_days_check", "courses", type_="check")
    op.drop_column("courses", "level")
    op.drop_column("courses", "expected_completion_days")


def downgrade() -> None:
    op.add_column("courses", sa.Column("expected_completion_days", sa.Integer(), nullable=True))
    op.add_column("courses", sa.Column("level", sa.String(20), nullable=True))
    op.create_check_constraint(
        "courses_level_check",
        "courses",
        "level IS NULL OR level IN ('beginner', 'intermediate', 'advanced')",
    )
    op.create_check_constraint(
        "courses_expected_completion_days_check",
        "courses",
        "expected_completion_days IS NULL OR expected_completion_days > 0",
    )
