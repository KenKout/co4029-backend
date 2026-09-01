"""Flatten organization units into faculties and add faculty ownership.

The physical ``org_units`` table and RBAC ``org_unit`` scope are retained for
compatibility, but every live row is now a top-level faculty.  Staff
affiliation moves to a many-to-many table, while courses receive one immutable
owning faculty (nullable for organization-wide courses).
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0094_flat_faculties"
down_revision = "0093_teacher_title_flags"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()

    # Every legacy tree has a usable faculty root.  Existing faculty rows are
    # detached first; non-faculty roots are promoted so orphan trees still
    # have a deterministic target rather than being widened to org scope.
    bind.execute(sa.text("UPDATE org_units SET parent_unit_id = NULL WHERE unit_type = 'faculty'"))
    bind.execute(
        sa.text(
            "UPDATE org_units SET unit_type = 'faculty' "
            "WHERE parent_unit_id IS NULL AND deleted_at IS NULL"
        )
    )

    # Resolve each live unit to its nearest faculty ancestor and rewrite all
    # references before non-faculty descendants are archived.
    bind.execute(
        sa.text(
            """
            CREATE TEMP TABLE _unit_faculty_map ON COMMIT DROP AS
            WITH RECURSIVE ancestors AS (
                SELECT id AS source_id, id AS ancestor_id, parent_unit_id,
                       unit_type, 0 AS depth
                FROM org_units WHERE deleted_at IS NULL
                UNION ALL
                SELECT a.source_id, p.id, p.parent_unit_id, p.unit_type,
                       a.depth + 1
                FROM ancestors a
                JOIN org_units p ON p.id = a.parent_unit_id
                WHERE p.deleted_at IS NULL
            ), ranked AS (
                SELECT source_id, ancestor_id AS faculty_id,
                       ROW_NUMBER() OVER (PARTITION BY source_id ORDER BY depth) AS rn
                FROM ancestors WHERE unit_type = 'faculty'
            )
            SELECT source_id, faculty_id FROM ranked WHERE rn = 1
            """
        )
    )
    for table in ("organization_memberships", "user_role_assignments", "user_permission_grants"):
        bind.execute(
            sa.text(
                f"UPDATE {table} t SET org_unit_id = m.faculty_id "
                "FROM _unit_faculty_map m WHERE t.org_unit_id = m.source_id"
            )
        )
    bind.execute(
        sa.text(
            "UPDATE courses c SET org_unit_id = m.faculty_id "
            "FROM _unit_faculty_map m WHERE c.org_unit_id = m.source_id"
        )
    )
    # Career paths are organization-wide and intentionally have no faculty.
    op.drop_column("career_paths", "org_unit_id")

    bind.execute(
        sa.text(
            "UPDATE org_units SET deleted_at = COALESCE(deleted_at, NOW()), "
            "updated_at = NOW() WHERE unit_type <> 'faculty' AND deleted_at IS NULL"
        )
    )
    bind.execute(
        sa.text(
            "UPDATE org_units SET parent_unit_id = NULL "
            "WHERE unit_type = 'faculty' AND deleted_at IS NULL"
        )
    )
    op.create_check_constraint(
        "ck_org_units_live_faculty_root",
        "org_units",
        "deleted_at IS NOT NULL OR (unit_type = 'faculty' AND parent_unit_id IS NULL)",
    )

    op.alter_column("courses", "org_unit_id", new_column_name="faculty_id")
    op.create_index("ix_courses_faculty_id", "courses", ["faculty_id"])

    op.create_table(
        "user_faculty_assignments",
        sa.Column("id", sa.Uuid(), nullable=False, server_default=sa.text("uuid_generate_v4()")),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("organization_id", sa.Uuid(), nullable=False),
        sa.Column("faculty_id", sa.Uuid(), nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default="active"),
        sa.Column(
            "active_from",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.Column("active_until", sa.DateTime(timezone=True), nullable=True),
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
        sa.CheckConstraint(
            "status IN ('active', 'inactive')", name="ck_user_faculty_assignments_status"
        ),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"], ondelete="NO ACTION"),
        sa.ForeignKeyConstraint(["faculty_id"], ["org_units.id"], ondelete="NO ACTION"),
        sa.ForeignKeyConstraint(["created_by"], ["users.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["updated_by"], ["users.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["deleted_by"], ["users.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_user_faculty_assignments_user_id", "user_faculty_assignments", ["user_id"])
    op.create_index(
        "ix_user_faculty_assignments_organization_id",
        "user_faculty_assignments",
        ["organization_id"],
    )
    op.create_index(
        "ix_user_faculty_assignments_faculty_id", "user_faculty_assignments", ["faculty_id"]
    )
    op.create_index(
        "uq_user_faculty_assignments_active",
        "user_faculty_assignments",
        ["user_id", "faculty_id"],
        unique=True,
        postgresql_where=sa.text(
            "deleted_at IS NULL AND status = 'active' AND active_until IS NULL"
        ),
    )

    # Backfill affiliation from the legacy primary membership and every
    # faculty-scoped role, thereby retaining existing multi-faculty deans.
    bind.execute(
        sa.text(
            """
            INSERT INTO user_faculty_assignments
                (id, user_id, organization_id, faculty_id, status,
                 active_from, created_at, updated_at)
            SELECT gen_random_uuid(), x.user_id, x.organization_id, x.faculty_id,
                   'active', NOW(), NOW(), NOW()
            FROM (
                SELECT user_id, organization_id, org_unit_id AS faculty_id
                FROM organization_memberships om
                WHERE om.org_unit_id IS NOT NULL
                  AND om.deleted_at IS NULL AND om.status = 'active'
                  AND EXISTS (
                      SELECT 1
                      FROM user_role_assignments ura
                      JOIN roles r ON r.id = ura.role_id
                      WHERE ura.user_id = om.user_id
                        AND ura.organization_id = om.organization_id
                        AND ura.deleted_at IS NULL
                        AND (ura.active_until IS NULL OR ura.active_until > NOW())
                        AND r.code IN ('hod', 'manager', 'teacher')
                        AND r.deleted_at IS NULL
                  )
                UNION
                SELECT ura.user_id, ura.organization_id, ura.org_unit_id AS faculty_id
                FROM user_role_assignments ura
                JOIN roles r ON r.id = ura.role_id
                WHERE ura.scope_kind = 'org_unit' AND ura.org_unit_id IS NOT NULL
                  AND ura.deleted_at IS NULL
                  AND (ura.active_until IS NULL OR ura.active_until > NOW())
                  AND r.code IN ('hod', 'manager', 'teacher')
                  AND r.deleted_at IS NULL
            ) x
            JOIN org_units f ON f.id = x.faculty_id
            WHERE f.unit_type = 'faculty' AND f.deleted_at IS NULL
            ON CONFLICT DO NOTHING
            """
        )
    )


def downgrade() -> None:
    op.drop_table("user_faculty_assignments")
    op.drop_index("ix_courses_faculty_id", table_name="courses")
    op.alter_column("courses", "faculty_id", new_column_name="org_unit_id")
    op.add_column(
        "career_paths",
        sa.Column(
            "org_unit_id",
            sa.Uuid(),
            sa.ForeignKey("org_units.id", ondelete="SET NULL"),
            nullable=True,
        ),
    )
    op.drop_constraint("ck_org_units_live_faculty_root", "org_units", type_="check")
