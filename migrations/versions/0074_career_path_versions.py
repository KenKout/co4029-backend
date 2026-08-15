"""Gap 3 chunk 1: career-path versioning — versions table + re-parent.

Adds ``career_path_versions`` and re-parents the route tables so an
enrollment is pinned to the version it started on:

* ``career_path_stages.career_path_id`` → ``version_id``
* ``career_course_items.career_path_id`` → ``version_id`` (PK re-keyed
  from ``(career_path_id, course_id)`` to ``(version_id, course_id)``:
  a course may appear in two VERSIONS of one path, but still only once
  per version)
* ``student_career_enrollments`` gains ``version_id`` — THE pin (D3(a):
  an enrollment never moves versions)
* ``career_readiness_snapshots`` gains ``version_id`` so a score's
  meaning is version-dependent as the plan requires

Backfill: every existing path gets ``version_no = 1`` (published if the
path is published, else draft), and every stage/item/enrollment/snapshot
row points at it — behaviour-preserving. ``student_stage_progress`` is
untouched: it latches ``stage_id``, and with D3(a) a student only ever
walks ONE version's stages, so latches never need to cross versions.

Down-revision 0073; revision id <= 32 chars.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0074_career_path_versions"
down_revision = "0073_drop_satisfied_by_pass"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1. The versions table. Status is draft|published; published_at is the
    #    activation timestamp of a published version (NULL while draft).
    op.execute(
        """
        CREATE TABLE career_path_versions (
            id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            career_path_id UUID NOT NULL
                REFERENCES career_paths(id) ON DELETE NO ACTION,
            version_no INTEGER NOT NULL,
            status VARCHAR(20) NOT NULL DEFAULT 'draft',
            published_at TIMESTAMPTZ,
            created_by UUID REFERENCES users(id) ON DELETE SET NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_by UUID REFERENCES users(id) ON DELETE SET NULL,
            deleted_at TIMESTAMPTZ,
            deleted_by UUID REFERENCES users(id) ON DELETE SET NULL,
            CONSTRAINT career_path_versions_path_version_key
                UNIQUE (career_path_id, version_no),
            CONSTRAINT ck_career_path_versions_status
                CHECK (status IN ('draft', 'published')),
            CONSTRAINT career_path_versions_version_no_check
                CHECK (version_no > 0)
        )
        """
    )
    op.create_index(
        "ix_career_path_versions_career_path_id",
        "career_path_versions",
        ["career_path_id"],
    )

    # 2. Backfill: every existing path becomes version 1. A published path
    #    gets a published v1 (enrollments pin to it); drafts get a draft v1.
    op.execute(
        """
        INSERT INTO career_path_versions
            (id, career_path_id, version_no, status, published_at, created_by)
        SELECT uuid_generate_v4(), id, 1,
               CASE WHEN status = 'published' THEN 'published' ELSE 'draft' END,
               CASE WHEN status = 'published' THEN NOW() ELSE NULL END,
               created_by
        FROM career_paths
        """
    )

    # 3. Stages: career_path_id → version_id.
    op.add_column("career_path_stages", sa.Column("version_id", sa.dialects.postgresql.UUID(as_uuid=True), nullable=True))
    op.execute(
        """
        UPDATE career_path_stages s
        SET version_id = v.id
        FROM career_path_versions v
        WHERE v.career_path_id = s.career_path_id AND v.version_no = 1
        """
    )
    op.alter_column("career_path_stages", "version_id", nullable=False)
    op.create_foreign_key(
        "career_path_stages_version_id_fkey",
        "career_path_stages",
        "career_path_versions",
        ["version_id"],
        ["id"],
        ondelete="NO ACTION",
    )
    op.create_index(
        "ix_career_path_stages_version_id",
        "career_path_stages",
        ["version_id"],
    )
    op.drop_index("ix_career_path_stages_career_path_id", table_name="career_path_stages")
    op.drop_constraint(
        "career_path_stages_career_path_id_fkey",
        "career_path_stages",
        type_="foreignkey",
    )
    op.drop_column("career_path_stages", "career_path_id")

    # 4. Course items: re-key PK (career_path_id, course_id) →
    #    (version_id, course_id). The old career_path_id FK (ON DELETE
    #    CASCADE) goes with the column; the new version FK is also CASCADE so
    #    deleting a (draft) version cleans its items, while published
    #    versions are protected at the service layer.
    op.add_column("career_course_items", sa.Column("version_id", sa.dialects.postgresql.UUID(as_uuid=True), nullable=True))
    op.execute(
        """
        UPDATE career_course_items ci
        SET version_id = v.id
        FROM career_path_versions v
        WHERE v.career_path_id = ci.career_path_id AND v.version_no = 1
        """
    )
    op.drop_constraint("career_course_items_pkey", "career_course_items", type_="primary")
    op.alter_column("career_course_items", "version_id", nullable=False)
    op.create_primary_key("career_course_items_pkey", "career_course_items", ["version_id", "course_id"])
    op.create_foreign_key(
        "career_course_items_version_id_fkey",
        "career_course_items",
        "career_path_versions",
        ["version_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.drop_column("career_course_items", "career_path_id")

    # 5. Enrollments: the version pin (D3(a): stays forever).
    op.add_column("student_career_enrollments", sa.Column("version_id", sa.dialects.postgresql.UUID(as_uuid=True), nullable=True))
    op.execute(
        """
        UPDATE student_career_enrollments e
        SET version_id = v.id
        FROM career_path_versions v
        WHERE v.career_path_id = e.career_path_id AND v.version_no = 1
        """
    )
    op.alter_column("student_career_enrollments", "version_id", nullable=False)
    op.create_foreign_key(
        "student_career_enrollments_version_id_fkey",
        "student_career_enrollments",
        "career_path_versions",
        ["version_id"],
        ["id"],
        ondelete="NO ACTION",
    )
    op.create_index(
        "ix_student_career_enrollments_version_id",
        "student_career_enrollments",
        ["version_id"],
    )

    # 6. Readiness snapshots: record which version produced the score.
    op.add_column("career_readiness_snapshots", sa.Column("version_id", sa.dialects.postgresql.UUID(as_uuid=True), nullable=True))
    op.execute(
        """
        UPDATE career_readiness_snapshots rs
        SET version_id = v.id
        FROM career_path_versions v
        WHERE v.career_path_id = rs.career_path_id AND v.version_no = 1
        """
    )
    op.alter_column("career_readiness_snapshots", "version_id", nullable=False)
    op.create_foreign_key(
        "career_readiness_snapshots_version_id_fkey",
        "career_readiness_snapshots",
        "career_path_versions",
        ["version_id"],
        ["id"],
        ondelete="NO ACTION",
    )
    op.create_index(
        "ix_career_readiness_snapshots_version_id",
        "career_readiness_snapshots",
        ["version_id"],
    )


def downgrade() -> None:
    # Reverse: restore career_path_id on stages/items (via the pinned v1
    # version's path), drop the version pin from enrollments/snapshots, and
    # drop the versions table. Best-effort: versioned data created AFTER the
    # upgrade is flattened onto the path's v1 version.
    op.execute(
        """
        UPDATE career_readiness_snapshots rs
        SET version_id = NULL
        """
    )
    op.drop_index("ix_career_readiness_snapshots_version_id", table_name="career_readiness_snapshots")
    op.drop_constraint(
        "career_readiness_snapshots_version_id_fkey",
        "career_readiness_snapshots",
        type_="foreignkey",
    )
    op.drop_column("career_readiness_snapshots", "version_id")

    op.execute(
        """
        UPDATE student_career_enrollments e
        SET version_id = NULL
        """
    )
    op.drop_index("ix_student_career_enrollments_version_id", table_name="student_career_enrollments")
    op.drop_constraint(
        "student_career_enrollments_version_id_fkey",
        "student_career_enrollments",
        type_="foreignkey",
    )
    op.drop_column("student_career_enrollments", "version_id")

    op.add_column("career_course_items", sa.Column("career_path_id", sa.dialects.postgresql.UUID(as_uuid=True), nullable=True))
    op.execute(
        """
        UPDATE career_course_items ci
        SET career_path_id = v.career_path_id
        FROM career_path_versions v
        WHERE v.id = ci.version_id AND v.version_no = 1
        """
    )
    op.drop_constraint("career_course_items_pkey", "career_course_items", type_="primary")
    op.alter_column("career_course_items", "career_path_id", nullable=False)
    op.create_primary_key("career_course_items_pkey", "career_course_items", ["career_path_id", "course_id"])
    op.create_foreign_key(
        "career_course_items_career_path_id_fkey",
        "career_course_items",
        "career_paths",
        ["career_path_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.drop_constraint(
        "career_course_items_version_id_fkey",
        "career_course_items",
        type_="foreignkey",
    )
    op.drop_column("career_course_items", "version_id")

    op.add_column("career_path_stages", sa.Column("career_path_id", sa.dialects.postgresql.UUID(as_uuid=True), nullable=True))
    op.execute(
        """
        UPDATE career_path_stages s
        SET career_path_id = v.career_path_id
        FROM career_path_versions v
        WHERE v.id = s.version_id AND v.version_no = 1
        """
    )
    op.alter_column("career_path_stages", "career_path_id", nullable=False)
    op.create_foreign_key(
        "career_path_stages_career_path_id_fkey",
        "career_path_stages",
        "career_paths",
        ["career_path_id"],
        ["id"],
        ondelete="NO ACTION",
    )
    op.create_index(
        "ix_career_path_stages_career_path_id",
        "career_path_stages",
        ["career_path_id"],
    )
    op.drop_index("ix_career_path_stages_version_id", table_name="career_path_stages")
    op.drop_constraint(
        "career_path_stages_version_id_fkey",
        "career_path_stages",
        type_="foreignkey",
    )
    op.drop_column("career_path_stages", "version_id")

    op.execute("DELETE FROM career_path_versions")
    op.drop_index("ix_career_path_versions_career_path_id", table_name="career_path_versions")
    op.drop_table("career_path_versions")
