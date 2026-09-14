"""Allow multiple active career paths in one learning-program enrollment.

Revision ID: 0118_multi_path_enrollments
Revises: 0117_path_change_note
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0118_multi_path_enrollments"
down_revision = "0117_path_change_note"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_index("uq_program_path_attempts_one_active", table_name="program_path_attempts")
    op.create_index(
        "uq_program_path_attempts_active_path",
        "program_path_attempts",
        ["program_enrollment_id", "career_path_id"],
        unique=True,
        postgresql_where=sa.text("status = 'active'"),
    )


def downgrade() -> None:
    op.drop_index("uq_program_path_attempts_active_path", table_name="program_path_attempts")
    op.create_index(
        "uq_program_path_attempts_one_active",
        "program_path_attempts",
        ["program_enrollment_id"],
        unique=True,
        postgresql_where=sa.text("status = 'active'"),
    )
