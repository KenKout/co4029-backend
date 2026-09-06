"""one live organization membership per user

Revision ID: 0109_one_org_per_user
Revises: 0108_policy_audience_literal
Create Date: 2026-09-06

Regular accounts have one tenant boundary.  A membership in any state except
``left`` reserves that boundary; inactive and suspended users must not become
attachable to another organization by accident.  Historical ``left`` and
soft-deleted rows remain available for audit purposes and do not reserve it.

The migration deliberately refuses to guess when existing data contains more
than one reserved membership for a user.  Picking a winning organization is a
business transfer, and this release does not support transfers.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0109_one_org_per_user"
down_revision = "0108_policy_audience_literal"
branch_labels = None
depends_on = None

_OLD_INDEX = "organization_memberships_user_id_organization_id_org_unit_i_key"
_NEW_INDEX = "uq_organization_memberships_one_live_org_per_user"


def upgrade() -> None:
    conn = op.get_bind()
    duplicate_count = conn.execute(
        sa.text(
            "SELECT COUNT(*) FROM ("
            " SELECT user_id FROM organization_memberships"
            " WHERE deleted_at IS NULL AND status <> 'left'"
            " GROUP BY user_id HAVING COUNT(*) > 1"
            ") duplicate_users"
        )
    ).scalar_one()
    if duplicate_count:
        raise RuntimeError(
            "Cannot enforce one organization per user: "
            f"{duplicate_count} user(s) have multiple non-left memberships. "
            "Resolve those memberships explicitly before rerunning the migration."
        )

    op.drop_index(_OLD_INDEX, table_name="organization_memberships")
    op.create_index(
        _NEW_INDEX,
        "organization_memberships",
        ["user_id"],
        unique=True,
        postgresql_where=sa.text("deleted_at IS NULL AND status <> 'left'"),
    )


def downgrade() -> None:
    op.drop_index(_NEW_INDEX, table_name="organization_memberships")
    op.create_index(
        _OLD_INDEX,
        "organization_memberships",
        ["user_id", "organization_id", "org_unit_id"],
        unique=True,
        postgresql_where=sa.text("deleted_at IS NULL"),
    )
