"""Audit trail and rollback source for runtime settings changes.

``system_settings`` records only the current value plus ``updated_by`` /
``updated_at``. That is enough to answer "what is it now" and nothing else: not
who changed it, not what it was before, not why, and — critically — not enough
to put it back. A global setting governs ingestion for every organization on
the deployment, so a bad value applied at 2am is an incident whose only
recovery today is somebody remembering the old number.

This table makes each change an explicit, reasoned, reversible event
(PRD ADM-031/033):

* ``before_value_json`` / ``after_value_json`` — NULL on either side is
  meaningful and is why both are nullable. NULL *before* means no override
  existed at that scope and the value was inherited; NULL *after* means the
  override was cleared and inheritance resumed. Storing 0 or the resolved
  inherited value instead would make "cleared" indistinguishable from "set to
  whatever it happened to inherit", and rollback would then re-pin a value
  nobody chose.
* ``reason`` — NOT NULL and enforced at the API. A change trail whose reason
  column is mostly empty is a log, not an audit.
* ``scope`` — carried explicitly rather than derived from
  ``organization_id IS NULL``, so a global row cannot be mistaken for an org
  row whose organization was later deleted.
* ``reverted_change_id`` — a rollback is itself a change, appended like any
  other and pointing at what it undid. Rolling back by deleting history would
  destroy the record of the incident being recovered from.

Append-only, like ``http_audit_log``: no soft-delete column, no UPDATE path.
Retention is a separate cleanup concern and is out of scope here.

Revision ID: 0089_setting_changes
Revises: 0088_interview_question_variant_groups
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision = "0089_setting_changes"
down_revision = "0088_interview_variant_groups"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "system_setting_changes",
        sa.Column(
            "id",
            UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("uuid_generate_v4()"),
        ),
        sa.Column("setting_key", sa.String(length=100), nullable=False),
        sa.Column(
            "organization_id",
            UUID(as_uuid=True),
            sa.ForeignKey("organizations.id", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column("scope", sa.String(length=20), nullable=False),
        sa.Column("action", sa.String(length=20), nullable=False),
        sa.Column("before_value_json", JSONB(), nullable=True),
        sa.Column("after_value_json", JSONB(), nullable=True),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column(
            "actor_id",
            UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "source",
            sa.String(length=30),
            nullable=False,
            server_default=sa.text("'admin_console'"),
        ),
        sa.Column(
            "reverted_change_id",
            UUID(as_uuid=True),
            sa.ForeignKey("system_setting_changes.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.CheckConstraint(
            "scope IN ('global', 'organization')",
            name="ck_system_setting_changes_scope",
        ),
        sa.CheckConstraint(
            "action IN ('set', 'clear', 'rollback')",
            name="ck_system_setting_changes_action",
        ),
        # A global change must not carry an organization, and an org-scoped one
        # must. Without this the scope column could disagree with the FK and
        # the history would show a "global" change filed under one tenant.
        sa.CheckConstraint(
            "(scope = 'global' AND organization_id IS NULL) OR "
            "(scope = 'organization' AND organization_id IS NOT NULL)",
            name="ck_system_setting_changes_scope_org",
        ),
        # An empty reason satisfies NOT NULL and defeats the point.
        sa.CheckConstraint(
            "length(btrim(reason)) > 0",
            name="ck_system_setting_changes_reason_nonempty",
        ),
    )
    # History for one setting, newest first — the rollback picker's read.
    op.create_index(
        "ix_system_setting_changes_key_created_at",
        "system_setting_changes",
        ["setting_key", sa.text("created_at DESC")],
    )
    # One tenant's change history.
    op.create_index(
        "ix_system_setting_changes_org_created_at",
        "system_setting_changes",
        ["organization_id", sa.text("created_at DESC")],
        postgresql_where=sa.text("organization_id IS NOT NULL"),
    )
    # The unfiltered recent-changes feed.
    op.create_index(
        "ix_system_setting_changes_created_at",
        "system_setting_changes",
        [sa.text("created_at DESC")],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_system_setting_changes_created_at", table_name="system_setting_changes"
    )
    op.drop_index(
        "ix_system_setting_changes_org_created_at",
        table_name="system_setting_changes",
    )
    op.drop_index(
        "ix_system_setting_changes_key_created_at",
        table_name="system_setting_changes",
    )
    op.drop_table("system_setting_changes")
