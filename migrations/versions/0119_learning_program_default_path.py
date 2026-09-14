"""Add a versioned default Career Path and selection provenance.

Revision ID: 0119_program_default_path
Revises: 0118_multi_path_enrollments

Published versions that already exist deliberately receive no default.  This
keeps their awaiting-path enrolments and historical behaviour unchanged.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0119_program_default_path"
down_revision = "0118_multi_path_enrollments"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "learning_program_version_paths",
        sa.Column("is_default", sa.Boolean(), nullable=False, server_default=sa.text("false")),
    )
    op.create_index(
        "uq_learning_program_version_paths_one_default",
        "learning_program_version_paths",
        ["program_version_id"],
        unique=True,
        postgresql_where=sa.text("is_default"),
    )
    op.add_column(
        "program_path_attempts",
        sa.Column(
            "selection_source",
            sa.String(length=20),
            nullable=False,
            server_default=sa.text("'student'"),
        ),
    )
    op.execute(
        "UPDATE program_path_attempts SET selection_source = 'path_change' "
        "WHERE previous_attempt_id IS NOT NULL"
    )
    op.create_check_constraint(
        "ck_program_path_attempts_selection_source",
        "program_path_attempts",
        "selection_source IN ('student','program_default','path_change')",
    )


def downgrade() -> None:
    op.drop_constraint(
        "ck_program_path_attempts_selection_source",
        "program_path_attempts",
        type_="check",
    )
    op.drop_column("program_path_attempts", "selection_source")
    op.drop_index(
        "uq_learning_program_version_paths_one_default",
        table_name="learning_program_version_paths",
    )
    op.drop_column("learning_program_version_paths", "is_default")
