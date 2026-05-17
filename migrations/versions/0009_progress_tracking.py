"""Progress tracking tables (T7.2)

Revision ID: 0009_progress_tracking
Revises: 0008_enrollments_audit_cols
Create Date: 2026-05-17 16:00:00.000000

T7.2 introduces two new tables for the per-user progress feature that the
new ``abridgeai/features/progress/`` slice owns:

* ``lesson_progress`` — one row per (user, lesson). Carries
  ``UUIDPrimaryKey + Timestamp + AuditedBy``. Distinct from the baseline
  ``student_lesson_progress`` table (composite (student_id, lesson_id) PK,
  different column names like ``progress_percent`` /
  ``time_spent_seconds``); per Reconciliation §A13 we leave the legacy
  table untouched and add this new shape alongside it.
* ``material_engagement`` — append-only event row per material-version
  view. Carries ``UUIDPrimaryKey + CreatedAt`` only (no updates, no
  audit-by, no soft-delete).

Both tables match the SQLAlchemy models in
``abridgeai/features/progress/models.py`` exactly. Constraint names match
the model ``__table_args__`` so ORM creates round-trip with this DDL.

Downgrade drops both tables.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID

revision = "0009_progress_tracking"
down_revision = "0008_enrollments_audit_cols"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1. lesson_progress
    op.create_table(
        "lesson_progress",
        sa.Column(
            "id",
            UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("uuid_generate_v4()"),
        ),
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
        sa.Column(
            "created_by",
            UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "updated_by",
            UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "user_id",
            UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "lesson_id",
            UUID(as_uuid=True),
            sa.ForeignKey("lessons.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "status",
            sa.String(length=20),
            nullable=False,
            server_default=sa.text("'not_started'"),
        ),
        sa.Column(
            "completion_percent",
            sa.Numeric(5, 2),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column("last_activity_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "total_time_seconds",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.UniqueConstraint(
            "user_id", "lesson_id", name="uq_lesson_progress_user_lesson"
        ),
        sa.CheckConstraint(
            "status IN ('not_started', 'in_progress', 'completed')",
            name="ck_lesson_progress_status",
        ),
        sa.CheckConstraint(
            "completion_percent >= 0 AND completion_percent <= 100",
            name="ck_lesson_progress_completion_percent",
        ),
        sa.CheckConstraint(
            "total_time_seconds >= 0",
            name="ck_lesson_progress_total_time_nonneg",
        ),
    )
    op.create_index(
        "ix_lesson_progress_user_id", "lesson_progress", ["user_id"]
    )
    op.create_index(
        "ix_lesson_progress_lesson_id", "lesson_progress", ["lesson_id"]
    )

    # 2. material_engagement
    op.create_table(
        "material_engagement",
        sa.Column(
            "id",
            UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("uuid_generate_v4()"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.Column(
            "user_id",
            UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "material_version_id",
            UUID(as_uuid=True),
            sa.ForeignKey(
                "learning_material_versions.id", ondelete="CASCADE"
            ),
            nullable=False,
        ),
        sa.Column("engagement_seconds", sa.Integer(), nullable=False),
        sa.Column("scroll_position_percent", sa.Numeric(5, 2), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("ended_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "engagement_seconds >= 0",
            name="ck_material_engagement_seconds_nonneg",
        ),
        sa.CheckConstraint(
            "scroll_position_percent IS NULL "
            "OR (scroll_position_percent >= 0 AND scroll_position_percent <= 100)",
            name="ck_material_engagement_scroll_range",
        ),
    )
    op.create_index(
        "ix_material_engagement_user_id", "material_engagement", ["user_id"]
    )
    op.create_index(
        "ix_material_engagement_material_version_id",
        "material_engagement",
        ["material_version_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_material_engagement_material_version_id",
        table_name="material_engagement",
    )
    op.drop_index(
        "ix_material_engagement_user_id", table_name="material_engagement"
    )
    op.drop_table("material_engagement")

    op.drop_index(
        "ix_lesson_progress_lesson_id", table_name="lesson_progress"
    )
    op.drop_index("ix_lesson_progress_user_id", table_name="lesson_progress")
    op.drop_table("lesson_progress")
