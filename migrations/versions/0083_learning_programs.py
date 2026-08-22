"""Versioned learning programs and student-selected career paths.

Managers and faculty deans enrol students into a program.  A learner then
selects one path from the immutable program version; later switches are
approved by the owning faculty dean and recorded as new attempts.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0083_learning_programs"
down_revision = "0082_course_syllabus_imports"
branch_labels = None
depends_on = None


def _uuid_pk() -> sa.Column:
    return sa.Column(
        "id",
        postgresql.UUID(as_uuid=True),
        primary_key=True,
        server_default=sa.text("gen_random_uuid()"),
    )


def _timestamps() -> list[sa.Column]:
    return [
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
        sa.Column(
            "created_by",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
        ),
        sa.Column(
            "updated_by",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
        ),
    ]


def upgrade() -> None:
    op.create_table(
        "learning_programs",
        _uuid_pk(),
        sa.Column(
            "organization_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("organizations.id", ondelete="NO ACTION"),
            nullable=False,
        ),
        sa.Column(
            "faculty_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("org_units.id", ondelete="NO ACTION"),
            nullable=False,
        ),
        sa.Column(
            "owner_faculty_dean_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="NO ACTION"),
            nullable=True,
        ),
        sa.Column("slug", sa.String(100), nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("description", sa.Text()),
        sa.Column("status", sa.String(20), nullable=False, server_default=sa.text("'draft'")),
        sa.Column("deleted_at", sa.DateTime(timezone=True)),
        sa.Column(
            "deleted_by",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
        ),
        *_timestamps(),
        sa.CheckConstraint(
            "status IN ('draft','published','archived')", name="ck_learning_programs_status"
        ),
    )
    op.create_index(
        "ix_learning_programs_organization_id", "learning_programs", ["organization_id"]
    )
    op.create_index("ix_learning_programs_faculty_id", "learning_programs", ["faculty_id"])
    op.create_index(
        "uq_learning_programs_org_slug",
        "learning_programs",
        ["organization_id", "slug"],
        unique=True,
        postgresql_where=sa.text("deleted_at IS NULL"),
    )

    op.create_table(
        "learning_program_versions",
        _uuid_pk(),
        sa.Column(
            "learning_program_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("learning_programs.id", ondelete="NO ACTION"),
            nullable=False,
        ),
        sa.Column("version_no", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default=sa.text("'draft'")),
        sa.Column("max_path_switches", sa.Integer(), nullable=False, server_default=sa.text("3")),
        sa.Column("published_at", sa.DateTime(timezone=True)),
        sa.Column("deleted_at", sa.DateTime(timezone=True)),
        sa.Column(
            "deleted_by",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
        ),
        *_timestamps(),
        sa.UniqueConstraint(
            "learning_program_id", "version_no", name="uq_learning_program_versions_no"
        ),
        sa.CheckConstraint(
            "status IN ('draft','published')", name="ck_learning_program_versions_status"
        ),
        sa.CheckConstraint("version_no > 0", name="ck_learning_program_versions_no"),
        sa.CheckConstraint("max_path_switches >= 0", name="ck_learning_program_versions_switches"),
    )
    op.create_index(
        "ix_learning_program_versions_program_id",
        "learning_program_versions",
        ["learning_program_id"],
    )

    op.create_table(
        "learning_program_version_paths",
        sa.Column(
            "program_version_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("learning_program_versions.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column(
            "career_path_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("career_paths.id", ondelete="NO ACTION"),
            primary_key=True,
        ),
        sa.Column(
            "career_path_version_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("career_path_versions.id", ondelete="NO ACTION"),
            nullable=False,
        ),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.UniqueConstraint(
            "program_version_id", "position", name="uq_learning_program_version_paths_position"
        ),
        sa.CheckConstraint("position > 0", name="ck_learning_program_version_paths_position"),
    )

    op.create_table(
        "program_enrollments",
        _uuid_pk(),
        sa.Column(
            "learning_program_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("learning_programs.id", ondelete="NO ACTION"),
            nullable=False,
        ),
        sa.Column(
            "program_version_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("learning_program_versions.id", ondelete="NO ACTION"),
            nullable=False,
        ),
        sa.Column(
            "student_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "status", sa.String(20), nullable=False, server_default=sa.text("'awaiting_path'")
        ),
        sa.Column(
            "enrolled_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column("completed_at", sa.DateTime(timezone=True)),
        sa.Column("withdrawn_at", sa.DateTime(timezone=True)),
        sa.Column("withdrawal_reason", sa.Text()),
        *_timestamps(),
        sa.UniqueConstraint(
            "learning_program_id", "student_id", name="uq_program_enrollments_program_student"
        ),
        sa.CheckConstraint(
            "status IN ('awaiting_path','active','completed','withdrawn','cancelled')",
            name="ck_program_enrollments_status",
        ),
    )
    op.create_index("ix_program_enrollments_student_id", "program_enrollments", ["student_id"])
    op.create_index(
        "ix_program_enrollments_program_id", "program_enrollments", ["learning_program_id"]
    )

    op.create_table(
        "program_path_attempts",
        _uuid_pk(),
        sa.Column(
            "program_enrollment_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("program_enrollments.id", ondelete="NO ACTION"),
            nullable=False,
        ),
        sa.Column(
            "career_path_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("career_paths.id", ondelete="NO ACTION"),
            nullable=False,
        ),
        sa.Column(
            "career_path_version_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("career_path_versions.id", ondelete="NO ACTION"),
            nullable=False,
        ),
        sa.Column(
            "previous_attempt_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("program_path_attempts.id", ondelete="NO ACTION"),
        ),
        sa.Column("status", sa.String(20), nullable=False, server_default=sa.text("'active'")),
        sa.Column(
            "selected_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column("ended_at", sa.DateTime(timezone=True)),
        sa.Column("exit_snapshot", postgresql.JSONB(astext_type=sa.Text())),
        *_timestamps(),
        sa.CheckConstraint(
            "status IN ('active','completed','switched_out','cancelled')",
            name="ck_program_path_attempts_status",
        ),
    )
    op.create_index(
        "ix_program_path_attempts_enrollment_id", "program_path_attempts", ["program_enrollment_id"]
    )
    op.create_index(
        "uq_program_path_attempts_one_active",
        "program_path_attempts",
        ["program_enrollment_id"],
        unique=True,
        postgresql_where=sa.text("status = 'active'"),
    )

    op.create_table(
        "path_change_requests",
        _uuid_pk(),
        sa.Column(
            "program_enrollment_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("program_enrollments.id", ondelete="NO ACTION"),
            nullable=False,
        ),
        sa.Column(
            "from_attempt_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("program_path_attempts.id", ondelete="NO ACTION"),
            nullable=False,
        ),
        sa.Column(
            "target_career_path_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("career_paths.id", ondelete="NO ACTION"),
            nullable=False,
        ),
        sa.Column(
            "target_career_path_version_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("career_path_versions.id", ondelete="NO ACTION"),
            nullable=False,
        ),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default=sa.text("'pending'")),
        sa.Column(
            "reviewed_by",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="NO ACTION"),
        ),
        sa.Column("reviewed_at", sa.DateTime(timezone=True)),
        sa.Column("decision_reason", sa.Text()),
        sa.Column(
            "new_attempt_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("program_path_attempts.id", ondelete="NO ACTION"),
        ),
        *_timestamps(),
        sa.CheckConstraint(
            "status IN ('pending','approved','rejected','cancelled','invalidated')",
            name="ck_path_change_requests_status",
        ),
    )
    op.create_index(
        "uq_path_change_requests_one_pending",
        "path_change_requests",
        ["program_enrollment_id"],
        unique=True,
        postgresql_where=sa.text("status = 'pending'"),
    )

    op.create_table(
        "course_completion_awards",
        _uuid_pk(),
        sa.Column(
            "student_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "course_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("courses.id", ondelete="NO ACTION"),
            nullable=False,
        ),
        sa.Column(
            "awarded_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "source_enrollment_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("course_enrollments.id", ondelete="SET NULL"),
        ),
        sa.Column("revoked_at", sa.DateTime(timezone=True)),
        sa.Column(
            "revoked_by",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="NO ACTION"),
        ),
        sa.Column("revocation_reason", sa.Text()),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.UniqueConstraint(
            "student_id", "course_id", name="uq_course_completion_awards_student_course"
        ),
    )

    op.create_table(
        "course_enrollment_entitlements",
        _uuid_pk(),
        sa.Column(
            "course_enrollment_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("course_enrollments.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("source_type", sa.String(30), nullable=False),
        sa.Column("source_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "granted_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column("revoked_at", sa.DateTime(timezone=True)),
        sa.Column(
            "created_by",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
        ),
        sa.UniqueConstraint(
            "course_enrollment_id",
            "source_type",
            "source_id",
            name="uq_course_enrollment_entitlements_source",
        ),
        sa.CheckConstraint(
            "source_type IN ('path_attempt','manual','invitation','legacy')",
            name="ck_course_enrollment_entitlements_source_type",
        ),
    )

    # Preserve existing direct path enrolments as one legacy program per path.
    # Old tenants may not have a faculty unit; create a scoped legacy faculty
    # so every path and enrolment is migrated instead of silently skipped.
    op.execute(
        sa.text("""
        INSERT INTO org_units (
            id, organization_id, unit_type, name, code, created_at, updated_at
        )
        SELECT gen_random_uuid(), o.id, 'faculty', 'Legacy Learning Programs',
               'legacy-learning-programs', NOW(), NOW()
        FROM organizations o
        WHERE o.deleted_at IS NULL
          AND EXISTS (
              SELECT 1 FROM career_paths cp
              WHERE cp.organization_id = o.id AND cp.deleted_at IS NULL
          )
          AND NOT EXISTS (
              SELECT 1 FROM org_units ou
              WHERE ou.organization_id = o.id
                AND ou.unit_type = 'faculty' AND ou.deleted_at IS NULL
          )
    """)
    )
    op.execute(
        sa.text("""
        WITH RECURSIVE unit_ancestors AS (
            SELECT cp.id AS path_id, ou.id, ou.parent_unit_id,
                   ou.unit_type, 1 AS depth
            FROM career_paths cp
            JOIN org_units ou ON ou.id = cp.org_unit_id AND ou.deleted_at IS NULL
            WHERE cp.deleted_at IS NULL
            UNION ALL
            SELECT ua.path_id, parent.id, parent.parent_unit_id,
                   parent.unit_type, ua.depth + 1
            FROM unit_ancestors ua
            JOIN org_units parent
              ON parent.id = ua.parent_unit_id AND parent.deleted_at IS NULL
        ), path_faculties AS (
            SELECT DISTINCT ON (path_id) path_id, id AS faculty_id
            FROM unit_ancestors
            WHERE unit_type = 'faculty'
            ORDER BY path_id, depth
        )
        INSERT INTO learning_programs (
            id, organization_id, faculty_id, owner_faculty_dean_id, slug, name,
            description, status, created_at, updated_at, created_by, updated_by
        )
        SELECT cp.id, cp.organization_id,
               COALESCE(pf.faculty_id,
                        (SELECT ou.id FROM org_units ou WHERE ou.organization_id = cp.organization_id AND ou.unit_type = 'faculty' AND ou.deleted_at IS NULL ORDER BY ou.created_at LIMIT 1)),
               NULL, 'legacy-' || cp.slug, cp.name, cp.description,
               CASE WHEN cp.status = 'archived' THEN 'archived' ELSE 'published' END,
               cp.created_at, cp.updated_at, cp.created_by, cp.updated_by
        FROM career_paths cp
        LEFT JOIN path_faculties pf ON pf.path_id = cp.id
        WHERE cp.deleted_at IS NULL
    """)
    )
    op.execute(
        sa.text("""
        INSERT INTO learning_program_versions (
            id, learning_program_id, version_no, status, max_path_switches,
            published_at, created_at, updated_at, created_by, updated_by
        )
        SELECT cpv.id, cpv.career_path_id, cpv.version_no,
               CASE WHEN cpv.status = 'published' THEN 'published' ELSE 'draft' END,
               3, cpv.published_at, cpv.created_at, cpv.updated_at, cpv.created_by, cpv.updated_by
        FROM career_path_versions cpv
        JOIN learning_programs lp ON lp.id = cpv.career_path_id
        WHERE cpv.deleted_at IS NULL
    """)
    )
    op.execute(
        sa.text("""
        INSERT INTO learning_program_version_paths (
            program_version_id, career_path_id, career_path_version_id, position
        )
        SELECT cpv.id, cpv.career_path_id, cpv.id, 1
        FROM career_path_versions cpv
        JOIN learning_program_versions lpv ON lpv.id = cpv.id
    """)
    )
    op.execute(
        sa.text("""
        INSERT INTO program_enrollments (
            id, learning_program_id, program_version_id, student_id, status,
            enrolled_at, completed_at, withdrawn_at, created_at, updated_at,
            created_by, updated_by
        )
        SELECT sce.id, sce.career_path_id, sce.version_id, sce.student_id,
               CASE sce.status WHEN 'active' THEN 'active' WHEN 'completed' THEN 'completed' ELSE 'withdrawn' END,
               sce.started_at, sce.completed_at,
               CASE WHEN sce.status = 'dropped' THEN sce.updated_at ELSE NULL END,
               sce.created_at, sce.updated_at, sce.created_by, sce.updated_by
        FROM student_career_enrollments sce
        JOIN learning_program_versions lpv ON lpv.id = sce.version_id
        WHERE sce.deleted_at IS NULL
    """)
    )
    op.execute(
        sa.text("""
        INSERT INTO program_path_attempts (
            id, program_enrollment_id, career_path_id, career_path_version_id,
            status, selected_at, ended_at, created_at, updated_at, created_by, updated_by
        )
        SELECT pe.id, pe.id, pe.learning_program_id, pe.program_version_id,
               CASE pe.status WHEN 'active' THEN 'active' WHEN 'completed' THEN 'completed' ELSE 'cancelled' END,
               pe.enrolled_at, COALESCE(pe.completed_at, pe.withdrawn_at),
               pe.created_at, pe.updated_at, pe.created_by, pe.updated_by
        FROM program_enrollments pe
    """)
    )
    op.execute(
        sa.text("""
        INSERT INTO course_completion_awards (
            student_id, course_id, awarded_at, source_enrollment_id
        )
        SELECT student_id, course_id, COALESCE(completed_at, updated_at), id
        FROM course_enrollments
        WHERE status = 'completed'
        ON CONFLICT (student_id, course_id) DO NOTHING
    """)
    )

    # Catalog permissions are intentionally NOT granted to admin. Admin's
    # historical ALL seed is additionally blocked by role/scope checks in the
    # learning-program service.
    op.execute(
        sa.text("""
        INSERT INTO permissions (id, code, name, description)
        VALUES
          (gen_random_uuid(), 'learning_program.read', 'Read learning programs', 'Read learning programs within assigned scope'),
          (gen_random_uuid(), 'learning_program.manage', 'Manage learning programs', 'Create, edit, publish, and archive learning programs'),
          (gen_random_uuid(), 'learning_program.enroll', 'Enroll in learning programs', 'Enroll or withdraw students in learning programs'),
          (gen_random_uuid(), 'learning_program.switch.review', 'Review path switches', 'Approve or reject career-path switches as the owning Faculty Dean')
        ON CONFLICT (code) WHERE deleted_at IS NULL DO NOTHING
    """)
    )
    op.execute(
        sa.text("""
        INSERT INTO role_permissions (role_id, permission_id)
        SELECT r.id, p.id
        FROM roles r CROSS JOIN permissions p
        WHERE r.code IN ('manager','hod')
          AND p.code IN ('learning_program.read','learning_program.manage','learning_program.enroll')
          AND r.deleted_at IS NULL AND p.deleted_at IS NULL
        ON CONFLICT DO NOTHING
    """)
    )
    op.execute(
        sa.text("""
        INSERT INTO role_permissions (role_id, permission_id)
        SELECT r.id, p.id FROM roles r CROSS JOIN permissions p
        WHERE r.code = 'hod' AND p.code = 'learning_program.switch.review'
          AND r.deleted_at IS NULL AND p.deleted_at IS NULL
        ON CONFLICT DO NOTHING
    """)
    )


def downgrade() -> None:
    op.drop_table("course_enrollment_entitlements")
    op.drop_table("course_completion_awards")
    op.drop_table("path_change_requests")
    op.drop_table("program_path_attempts")
    op.drop_table("program_enrollments")
    op.drop_table("learning_program_version_paths")
    op.drop_table("learning_program_versions")
    op.drop_table("learning_programs")
    op.execute(
        sa.text("""
        DELETE FROM role_permissions WHERE permission_id IN (
          SELECT id FROM permissions WHERE code LIKE 'learning_program.%'
        )
    """)
    )
    op.execute(sa.text("DELETE FROM permissions WHERE code LIKE 'learning_program.%'"))
