"""Policy documents as a versioned, admin-editable entity.

Replaces the five hardcoded front-end ``PolicyDocument`` constants with three
tables, on the same identity/version split migration 0074 gave Career Path.

Uniqueness is PARTIAL throughout (``WHERE deleted_at IS NULL``), matching the
0002 house rule: archiving a policy must free its slug rather than reserve it
forever.

``policy_audience_roles.role_id`` points at ``roles.id``, NOT ``roles.code`` —
0002 replaced the full unique constraint on ``code`` with a partial unique
index, and PostgreSQL cannot back a foreign key with a partial index.

Revision ID: 0102_policies
Revises: 0101_discussion_notify_category
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0102_policies"
down_revision = "0101_discussion_notify_category"
branch_labels = None
depends_on = None


def _audit_columns() -> list[sa.Column]:
    """The TimestampMixin + AuditedByMixin + SoftDeleteMixin column set."""
    return [
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.Column("created_by", sa.Uuid(), nullable=True),
        sa.Column("updated_by", sa.Uuid(), nullable=True),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("deleted_by", sa.Uuid(), nullable=True),
    ]


def _audit_fks() -> list[sa.ForeignKeyConstraint]:
    return [
        sa.ForeignKeyConstraint(["created_by"], ["users.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["updated_by"], ["users.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["deleted_by"], ["users.id"], ondelete="SET NULL"),
    ]


def upgrade() -> None:
    op.create_table(
        "policies",
        sa.Column("id", sa.Uuid(), nullable=False, server_default=sa.text("uuid_generate_v4()")),
        sa.Column("slug", sa.String(100), nullable=False),
        sa.Column("category", sa.String(20), nullable=False),
        *_audit_columns(),
        sa.CheckConstraint("category IN ('legal', 'academic')", name="ck_policies_category"),
        *_audit_fks(),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_policies_slug", "policies", ["slug"])
    op.create_index(
        "uq_policies_slug",
        "policies",
        ["slug"],
        unique=True,
        postgresql_where=sa.text("deleted_at IS NULL"),
    )

    op.create_table(
        "policy_versions",
        sa.Column("id", sa.Uuid(), nullable=False, server_default=sa.text("uuid_generate_v4()")),
        sa.Column("policy_id", sa.Uuid(), nullable=False),
        sa.Column("version_no", sa.Integer(), nullable=False),
        sa.Column("language", sa.String(10), nullable=False, server_default="en"),
        sa.Column("status", sa.String(20), nullable=False, server_default="draft"),
        sa.Column("title", sa.String(255), nullable=False),
        sa.Column("body", sa.Text(), nullable=False),
        sa.Column("format", sa.String(20), nullable=False, server_default="markdown"),
        sa.Column("changelog", sa.Text(), nullable=True),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("published_by", sa.Uuid(), nullable=True),
        *_audit_columns(),
        sa.CheckConstraint(
            "status IN ('draft', 'published', 'archived')",
            name="ck_policy_versions_status",
        ),
        sa.CheckConstraint("version_no > 0", name="ck_policy_versions_version_no"),
        sa.CheckConstraint(
            "(status = 'published') = (published_at IS NOT NULL)",
            name="ck_policy_versions_published_at",
        ),
        sa.ForeignKeyConstraint(["policy_id"], ["policies.id"], ondelete="NO ACTION"),
        sa.ForeignKeyConstraint(["published_by"], ["users.id"], ondelete="SET NULL"),
        *_audit_fks(),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_policy_versions_policy", "policy_versions", ["policy_id"])
    op.create_index(
        "uq_policy_versions_policy_language_version",
        "policy_versions",
        ["policy_id", "language", "version_no"],
        unique=True,
        postgresql_where=sa.text("deleted_at IS NULL"),
    )

    op.create_table(
        "policy_audience_roles",
        sa.Column("id", sa.Uuid(), nullable=False, server_default=sa.text("uuid_generate_v4()")),
        sa.Column("policy_id", sa.Uuid(), nullable=False),
        sa.Column("role_id", sa.Uuid(), nullable=False),
        *_audit_columns(),
        sa.ForeignKeyConstraint(["policy_id"], ["policies.id"], ondelete="NO ACTION"),
        sa.ForeignKeyConstraint(["role_id"], ["roles.id"], ondelete="NO ACTION"),
        *_audit_fks(),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_policy_audience_roles_policy", "policy_audience_roles", ["policy_id"])
    op.create_index("idx_policy_audience_roles_role", "policy_audience_roles", ["role_id"])
    op.create_index(
        "uq_policy_audience_roles_policy_role",
        "policy_audience_roles",
        ["policy_id", "role_id"],
        unique=True,
        postgresql_where=sa.text("deleted_at IS NULL"),
    )


def downgrade() -> None:
    op.drop_table("policy_audience_roles")
    op.drop_table("policy_versions")
    op.drop_table("policies")
