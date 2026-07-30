"""Add lesson_knowledge_graphs table (teacher-curated, publishable KG)

Revision ID: 0062_lesson_kg
Revises: 0061_course_contact
Create Date: 2026-07-25

Introduces a teacher-authored knowledge graph per lesson, stored as JSON in
Postgres — deliberately SEPARATE from the AI-generated concept graph in Neo4j
(which stays read-only and behind ``knowledge_graph_enabled``). This table is
what the teacher edits (CRUD nodes / edges / node detail) and publishes to the
student reading-lesson view.

Model
-----
One row per lesson (``lesson_id`` UNIQUE):

* ``draft_json``       JSONB NOT NULL — the working copy the teacher edits. Shape:
  ``{"nodes": [{id,label,type,definition,weight}], "edges": [{source,target,relation}]}``.
* ``primary_node_id``  VARCHAR(200) — the single required centre node's id, kept
  as a top-level column (not buried in JSON) so the "exactly one primary" rule
  is queryable and enforceable. Nullable only for the brief pre-first-save state.
* ``published_json``   JSONB NULL — snapshot shown to students. NULL until the
  first publish. Re-publishing overwrites it with the current draft.
* ``published_primary_node_id`` VARCHAR(200) NULL — primary node id captured at
  publish time (draft's primary may drift before the next publish).
* ``published_at``     TIMESTAMPTZ NULL — when the draft was last snapshotted.

Audit + soft-delete via the shared mixins (created_by / updated_by /
deleted_at / deleted_by) for consistency with the rest of the materials
aggregate.

Additive table — fully reversible.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PGUUID

# NOTE: alembic_version.version_num is varchar(32); keep revision id <= 32 chars.
revision = "0062_lesson_kg"
down_revision = "0061_course_contact"
branch_labels = None
depends_on = None

_TABLE = "lesson_knowledge_graphs"


def upgrade() -> None:
    op.create_table(
        _TABLE,
        sa.Column(
            "id",
            PGUUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("uuid_generate_v4()"),
        ),
        sa.Column(
            "lesson_id",
            PGUUID(as_uuid=True),
            sa.ForeignKey("lessons.id", ondelete="NO ACTION"),
            nullable=False,
        ),
        sa.Column("draft_json", JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("primary_node_id", sa.String(length=200), nullable=True),
        sa.Column("published_json", JSONB(), nullable=True),
        sa.Column("published_primary_node_id", sa.String(length=200), nullable=True),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=True),
        # Timestamp mixin
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("NOW()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("NOW()")),
        # Audit mixin
        sa.Column(
            "created_by",
            PGUUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "updated_by",
            PGUUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
        ),
        # Soft-delete mixin
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "deleted_by",
            PGUUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
        ),
    )
    # One curated KG per lesson.
    op.create_unique_constraint(
        "uq_lesson_knowledge_graphs_lesson", _TABLE, ["lesson_id"]
    )
    op.create_index(
        "ix_lesson_knowledge_graphs_deleted_at", _TABLE, ["deleted_at"]
    )


def downgrade() -> None:
    op.drop_index("ix_lesson_knowledge_graphs_deleted_at", table_name=_TABLE)
    op.drop_constraint("uq_lesson_knowledge_graphs_lesson", _TABLE, type_="unique")
    op.drop_table(_TABLE)
