"""Remove legacy single-owner and single-faculty membership links.

Revision ID: 0112_remove_legacy_ownership
Revises: 0111_discussion_scope
Create Date: 2026-09-11 00:00:00.000000

Faculty authority is derived from scoped role assignments, so a Learning
Program must not pin one dean. Organization membership is the tenant boundary;
multi-faculty staff affiliation lives exclusively in
``user_faculty_assignments``.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0112_remove_legacy_ownership"
down_revision = "0111_discussion_scope"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_column("learning_programs", "owner_faculty_dean_id")
    op.drop_column("organization_memberships", "org_unit_id")


def downgrade() -> None:
    op.add_column(
        "organization_memberships",
        sa.Column(
            "org_unit_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("org_units.id", ondelete="SET NULL"),
            nullable=True,
        ),
    )
    op.add_column(
        "learning_programs",
        sa.Column(
            "owner_faculty_dean_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="NO ACTION"),
            nullable=True,
        ),
    )
