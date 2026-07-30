"""Per-organization runtime settings.

``system_settings`` was a single global key/value table, and the ingestion
knobs it should have held were not in it at all — ``CHUNKING_MAX_TOKENS``,
``CHUNKING_SEMANTIC_GLUE_THRESHOLD`` and friends were in ``.env`` but read by
nothing, so the values in force were the hard-coded defaults in
``SemanticChunker.__init__``.

This makes the table org-scoped so each organization can tune its own
ingestion behaviour, with ``organization_id IS NULL`` as the deployment-wide
default. Resolution order is org row -> global row -> environment -> code
default; see ``abridgeai/core/settings_registry.py``.

The unique constraint moves from ``setting_key`` alone to
``(organization_id, setting_key)``. A plain composite UNIQUE would NOT enforce
one global row per key, because Postgres treats NULLs as distinct — two rows
with ``organization_id IS NULL`` and the same key would both be accepted, and
the resolver would then pick between them arbitrarily. Two partial unique
indexes are used instead: one covering org rows, one covering the global rows.

Revision ID: 0066_org_settings
Revises: 0065_interview_practice
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0066_org_settings"
down_revision = "0065_interview_practice"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "system_settings",
        sa.Column("organization_id", sa.dialects.postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.create_foreign_key(
        "fk_system_settings_organization_id",
        "system_settings",
        "organizations",
        ["organization_id"],
        ["id"],
        ondelete="CASCADE",
    )

    # The baseline's UNIQUE on setting_key alone would reject a per-org row
    # whose key already exists globally, which is exactly the override case.
    op.drop_constraint("system_settings_setting_key_key", "system_settings", type_="unique")

    op.create_index(
        "uq_system_settings_org_key",
        "system_settings",
        ["organization_id", "setting_key"],
        unique=True,
        postgresql_where=sa.text("organization_id IS NOT NULL"),
    )
    op.create_index(
        "uq_system_settings_global_key",
        "system_settings",
        ["setting_key"],
        unique=True,
        postgresql_where=sa.text("organization_id IS NULL"),
    )
    # The resolver reads every row for one org plus every global row in a
    # single query; this index serves the org half of that.
    op.create_index(
        "ix_system_settings_organization_id",
        "system_settings",
        ["organization_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_system_settings_organization_id", table_name="system_settings")
    op.drop_index("uq_system_settings_global_key", table_name="system_settings")
    op.drop_index("uq_system_settings_org_key", table_name="system_settings")

    # Per-org rows cannot survive the return to a globally unique key: two orgs
    # overriding the same setting would collide. Drop them and keep the global
    # defaults, which is the pre-migration behaviour.
    op.execute(sa.text("DELETE FROM system_settings WHERE organization_id IS NOT NULL"))

    op.drop_constraint("fk_system_settings_organization_id", "system_settings", type_="foreignkey")
    op.drop_column("system_settings", "organization_id")
    op.create_unique_constraint(
        "system_settings_setting_key_key", "system_settings", ["setting_key"]
    )
