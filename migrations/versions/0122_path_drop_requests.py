"""Let a student request dropping one Career Path, not only switching.

Revision ID: 0122_path_drop_requests
Revises: 0121_program_path_limit

A drop reuses ``path_change_requests`` rather than getting a table of its own:
it is the same review queue, the same single-open-request rule, the same
switch budget and the same rejection vocabulary. The only structural
difference is that a drop has no destination, so the two ``target_*`` columns
become nullable and a CHECK ties their presence to ``kind``.

Existing rows are all switches, which is what the ``'change'`` server default
backfills.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0122_path_drop_requests"
down_revision = "0121_program_path_limit"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "path_change_requests",
        sa.Column("kind", sa.String(length=20), nullable=False, server_default=sa.text("'change'")),
    )
    op.create_check_constraint(
        "ck_path_change_requests_kind",
        "path_change_requests",
        "kind IN ('change','drop')",
    )

    # A drop has nowhere to go, so the destination is optional at the column
    # level and mandatory-by-kind at the constraint level.
    op.alter_column("path_change_requests", "target_career_path_id", nullable=True)
    op.alter_column("path_change_requests", "target_career_path_version_id", nullable=True)
    op.create_check_constraint(
        "ck_path_change_requests_kind_target",
        "path_change_requests",
        "(kind = 'change' AND target_career_path_id IS NOT NULL "
        "AND target_career_path_version_id IS NOT NULL) "
        "OR (kind = 'drop' AND target_career_path_id IS NULL "
        "AND target_career_path_version_id IS NULL)",
    )


def downgrade() -> None:
    # Drop requests cannot be represented once the columns are mandatory
    # again, and silently deleting a student's academic record would be
    # worse than refusing. Terminal rows are kept by nulling nothing and
    # failing loudly if any drop row survives.
    rows = op.get_bind().scalar(
        sa.text("SELECT COUNT(*) FROM path_change_requests WHERE kind = 'drop'")
    )
    if rows:
        raise RuntimeError(
            f"{rows} path-drop request(s) exist; downgrading would lose them. "
            "Export or delete them before downgrading past 0122."
        )
    op.drop_constraint("ck_path_change_requests_kind_target", "path_change_requests", type_="check")
    op.alter_column("path_change_requests", "target_career_path_version_id", nullable=False)
    op.alter_column("path_change_requests", "target_career_path_id", nullable=False)
    op.drop_constraint("ck_path_change_requests_kind", "path_change_requests", type_="check")
    op.drop_column("path_change_requests", "kind")
