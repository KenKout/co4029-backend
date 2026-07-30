"""Preprocessing quarantine + per-material preprocess mode.

Backs ``abridgeai/ai/preprocessing``: the cascade that strips running
headers/footers, blank pages, cover/instructor blocks, tables of contents,
bibliographies and closing slides between extraction and chunking.

Nothing the cascade removes is hard-deleted. Every removal lands here with
the exact text, a reason code and a rule score, so a threshold that turns out
to be wrong is reverted with a config change and a reprocess rather than a
re-upload — and a teacher can restore an individual unit.

Unlike ``document_chunks`` (deleted and rebuilt on every ingest), this table
SURVIVES re-ingest: it carries teacher decisions, and a restore must not have
to be repeated after each reprocess. The pipeline upserts on
``(material_version_id, unit_kind, ordinal)`` and leaves ``teacher_action*``
untouched.

Revision ID: 0064_preprocess_quar
Revises: 0063_iq_embedding
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0064_preprocess_quar"
down_revision = "0063_iq_embedding"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "learning_materials",
        sa.Column(
            "preprocess_mode",
            sa.String(length=16),
            nullable=False,
            server_default=sa.text("'full'"),
        ),
    )
    op.create_check_constraint(
        "learning_materials_preprocess_mode_check",
        "learning_materials",
        "preprocess_mode IN ('full', 'normalize_only', 'off')",
    )

    op.create_table(
        "material_preprocess_quarantine",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "material_version_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("learning_material_versions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "course_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("courses.id", ondelete="NO ACTION"),
            nullable=False,
        ),
        sa.Column("unit_kind", sa.String(length=8), nullable=False),
        sa.Column("page_number", sa.Integer(), nullable=True),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        # The point of the table: what was actually removed.
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("occurrences", sa.Integer(), nullable=False, server_default=sa.text("1")),
        sa.Column("rule_name", sa.String(length=64), nullable=False),
        sa.Column("reason_code", sa.String(length=48), nullable=False),
        sa.Column("action", sa.String(length=24), nullable=False),
        sa.Column("rule_score", sa.Float(), nullable=True),
        sa.Column("detector_stage", sa.String(length=16), nullable=False),
        sa.Column("teacher_action", sa.String(length=16), nullable=True),
        sa.Column(
            "teacher_action_by",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("teacher_action_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.UniqueConstraint(
            "material_version_id",
            "unit_kind",
            "ordinal",
            name="material_preprocess_quarantine_unit_key",
        ),
        sa.CheckConstraint(
            "unit_kind IN ('page', 'line', 'chunk')",
            name="material_preprocess_quarantine_unit_kind_check",
        ),
        sa.CheckConstraint(
            "teacher_action IS NULL OR teacher_action IN ('restore', 'confirm')",
            name="material_preprocess_quarantine_teacher_action_check",
        ),
    )
    op.create_index(
        "ix_material_preprocess_quarantine_version",
        "material_preprocess_quarantine",
        ["material_version_id"],
    )
    # Powers the course-wide "what is the filter eating across this course"
    # audit that measures filter precision.
    op.create_index(
        "ix_material_preprocess_quarantine_course_reason",
        "material_preprocess_quarantine",
        ["course_id", "reason_code"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_material_preprocess_quarantine_course_reason",
        table_name="material_preprocess_quarantine",
    )
    op.drop_index(
        "ix_material_preprocess_quarantine_version",
        table_name="material_preprocess_quarantine",
    )
    op.drop_table("material_preprocess_quarantine")
    op.drop_constraint(
        "learning_materials_preprocess_mode_check",
        "learning_materials",
        type_="check",
    )
    op.drop_column("learning_materials", "preprocess_mode")
