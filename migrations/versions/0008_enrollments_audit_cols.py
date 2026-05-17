"""Enrollments + InvitationCode mixin alignment (T7.1)

Revision ID: 0008_enrollments_invitation_codes
Revises: 0007_quiz_renames_and_sr_fields
Create Date: 2026-05-17 14:00:00.000000

T7.1 backfill — the baseline schema (0001) already creates
``course_enrollments`` and ``course_invitation_codes`` (lines 404-432) but
neither table is in ``SOFT_DELETE_TABLES``, so neither received the
audit-column bulk loop. The :class:`Enrollment` model carries
``UUIDPrimaryKey + Timestamp + AuditedBy`` (no SoftDelete; uses
``status`` lifecycle), and :class:`InvitationCode` carries the same plus
``SoftDelete``. The plan also widens the invitation-code shape with
``organization_id`` (multi-tenant tracking) and ``current_uses`` (audit
counter for Manager-handed codes).

Changes
-------

1. ``course_enrollments``
   * ``+ created_by`` UUID FK users ON DELETE SET NULL
   * ``+ updated_by`` UUID FK users ON DELETE SET NULL

2. ``course_invitation_codes``
   * ``+ updated_by`` UUID FK users ON DELETE SET NULL
   * ``+ deleted_at`` TIMESTAMPTZ (indexed)
   * ``+ deleted_by`` UUID FK users ON DELETE SET NULL
   * ``+ organization_id`` UUID FK organizations ON DELETE CASCADE
     (NOT NULL after backfill from courses.organization_id)
   * ``+ current_uses`` INT NOT NULL DEFAULT 0

The ``created_by`` column already exists on ``course_invitation_codes``
(per :data:`TABLES_WITH_EXISTING_CREATED_BY` in 0001).

Backfill
--------

* ``course_invitation_codes.organization_id`` is populated from
  ``courses.organization_id`` for every existing row before the NOT NULL
  flip. Test environments rarely have rows here pre-T7.1, but the
  backfill is defensive.

Round-trip
----------

The downgrade reverses every change. Drops the indexes, then the columns
(reverse order). The ``current_uses`` and ``organization_id`` removals
lose data; that is intentional for round-trip parity.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID

revision = "0008_enrollments_audit_cols"
down_revision = "0007_quiz_renames_and_sr_fields"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1. course_enrollments: created_by + updated_by
    op.add_column(
        "course_enrollments",
        sa.Column(
            "created_by",
            UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
        ),
    )
    op.add_column(
        "course_enrollments",
        sa.Column(
            "updated_by",
            UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
        ),
    )

    # 2. course_invitation_codes: updated_by + soft-delete pair
    op.add_column(
        "course_invitation_codes",
        sa.Column(
            "updated_by",
            UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
        ),
    )
    op.add_column(
        "course_invitation_codes",
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "course_invitation_codes",
        sa.Column(
            "deleted_by",
            UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
        ),
    )
    op.create_index(
        "ix_course_invitation_codes_deleted_at",
        "course_invitation_codes",
        ["deleted_at"],
    )

    # 3. course_invitation_codes: organization_id (nullable -> backfill -> NOT NULL)
    op.add_column(
        "course_invitation_codes",
        sa.Column(
            "organization_id",
            UUID(as_uuid=True),
            sa.ForeignKey("organizations.id", ondelete="CASCADE"),
            nullable=True,
        ),
    )
    op.execute(
        """
        UPDATE course_invitation_codes ic
        SET organization_id = c.organization_id
        FROM courses c
        WHERE ic.course_id = c.id
          AND ic.organization_id IS NULL
        """
    )
    op.alter_column(
        "course_invitation_codes",
        "organization_id",
        existing_type=UUID(as_uuid=True),
        nullable=False,
    )

    # 4. course_invitation_codes: current_uses counter
    op.add_column(
        "course_invitation_codes",
        sa.Column(
            "current_uses",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("0"),
        ),
    )


def downgrade() -> None:
    # Reverse 4
    op.drop_column("course_invitation_codes", "current_uses")

    # Reverse 3
    op.drop_column("course_invitation_codes", "organization_id")

    # Reverse 2
    op.drop_index(
        "ix_course_invitation_codes_deleted_at",
        table_name="course_invitation_codes",
    )
    op.drop_column("course_invitation_codes", "deleted_by")
    op.drop_column("course_invitation_codes", "deleted_at")
    op.drop_column("course_invitation_codes", "updated_by")

    # Reverse 1
    op.drop_column("course_enrollments", "updated_by")
    op.drop_column("course_enrollments", "created_by")
