"""Course import from a syllabus PDF: attempt log + stored source document.

Adds ``course_syllabus_imports``, one row per import attempt by a manager.

Why a table rather than just returning the parse result:

* the uploaded syllabus has to stay downloadable by teachers and students
  after the import, so the ``storage_objects`` row needs an owner;
* a FAILED attempt has no course to hang off (``course_id`` stays NULL),
  but the manager still gets a notification naming the reason, and that
  notification needs something durable behind it;
* parser warnings on an otherwise successful import (outcomes renumbered
  because the source syllabus skipped a code, no total-hours row) have to
  survive past the HTTP response.

``course_id`` is ``ON DELETE SET NULL`` on purpose: deleting an imported
course must not take the audit row — or the archived source document —
with it. Same reasoning for ``storage_object_id``.

No backfill: the table starts empty, and nothing reads it until the
import endpoint lands.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# NOTE: alembic_version.version_num is varchar(32); keep revision id <= 32 chars.
revision = "0082_course_syllabus_imports"
down_revision = "0081_hod_rename_faculty_dean"
branch_labels = None
depends_on = None

_TABLE = "course_syllabus_imports"


def upgrade() -> None:
    op.create_table(
        _TABLE,
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "organization_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("organizations.id", ondelete="NO ACTION"),
            nullable=False,
        ),
        sa.Column(
            "course_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("courses.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "storage_object_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("storage_objects.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "imported_by",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="NO ACTION"),
            nullable=False,
        ),
        sa.Column("language", sa.String(length=2), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("original_filename", sa.String(length=255), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column(
            "warnings",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column("outcome_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
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
        sa.CheckConstraint(
            "status IN ('succeeded', 'failed')",
            name="ck_course_syllabus_imports_status",
        ),
        sa.CheckConstraint(
            "language IN ('vi', 'en')",
            name="ck_course_syllabus_imports_language",
        ),
        # A successful import always produced a course; a failed one may not
        # have got that far. Keeps "succeeded but no course" unrepresentable.
        sa.CheckConstraint(
            "(status = 'succeeded' AND course_id IS NOT NULL) OR status = 'failed'",
            name="ck_course_syllabus_imports_course_on_success",
        ),
    )
    op.create_index(
        "ix_course_syllabus_imports_organization_id", _TABLE, ["organization_id"]
    )
    # The download path looks up "the syllabus of THIS course", newest first.
    op.create_index("ix_course_syllabus_imports_course_id", _TABLE, ["course_id"])


def downgrade() -> None:
    op.drop_index("ix_course_syllabus_imports_course_id", table_name=_TABLE)
    op.drop_index("ix_course_syllabus_imports_organization_id", table_name=_TABLE)
    op.drop_table(_TABLE)
