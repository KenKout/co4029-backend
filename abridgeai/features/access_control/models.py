"""Access control + organization ORM models.

Ported from legacy ``backend/app/models/{access_control,organization}.py``
to the feature-first layout. The schema source of truth is the baseline
Alembic migration (``migrations/versions/0001_baseline_schema.py``); legacy
ORM names were updated to match the migration where they had drifted.

Key invariants preserved verbatim from baseline DDL:

* ``ck_user_role_assignments_scope_kind`` and
  ``ck_user_permission_grants_scope_kind`` enforce the 4-scope coverage
  (global / organization / org_unit / course) that T1.4a / FIX-CRIT-2 will
  rely on. **Do not paraphrase** — text matches the migration exactly.
* Status / kind enums are enforced via ``CHECK`` constraints, mirroring
  baseline DDL.
* Soft-delete eligibility follows the migration's ``SOFT_DELETE_TABLES``
  inventory (Reconciliation §A13). RolePermission is the only join here
  without ``deleted_at``.
* T0.14 cascade flip: every FK whose parent is soft-deletable uses
  ``ondelete='NO ACTION'``. FKs to ``users.id`` / ``courses.id`` follow
  the migration choice (CASCADE for owner identity, SET NULL for
  ``granted_by``, NO ACTION for ``course_id`` after the flip).
* Per Reconciliation §A1, the M2M between :class:`CareerPath` and
  ``courses`` lives in ``features/career_paths/`` and is added in
  Phase 7 (T7.3). Not declared here.
* Per Reconciliation §B6, :class:`CareerPath.description` is a net-new
  column added in T0.9 and is declared on the ORM here.

Import-linter contract: this module only imports ``sqlalchemy`` and
``abridgeai.core.db``. FK references to ``users.id`` / ``courses.id``
use string targets so we do not pull cross-feature ORM modules in.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    text,
)
from sqlalchemy.dialects.postgresql import CITEXT
from sqlalchemy.orm import Mapped, mapped_column, relationship

from abridgeai.core.db import (
    PGUUID,
    AuditedByMixin,
    Base,
    CreatedAtMixin,
    SoftDeleteMixin,
    TimestampMixin,
    UUIDPrimaryKeyMixin,
)

# CHECK constraint text shared between ``user_role_assignments`` and
# ``user_permission_grants``. Copied verbatim from
# migrations/versions/0001_baseline_schema.py (lines 286-291 / 308-313).
_SCOPE_KIND_CHECK = (
    "(scope_kind = 'global' AND organization_id IS NULL "
    "AND org_unit_id IS NULL AND course_id IS NULL) OR "
    "(scope_kind = 'organization' AND organization_id IS NOT NULL "
    "AND org_unit_id IS NULL AND course_id IS NULL) OR "
    "(scope_kind = 'org_unit' AND organization_id IS NOT NULL "
    "AND org_unit_id IS NOT NULL AND course_id IS NULL) OR "
    "(scope_kind = 'course' AND organization_id IS NOT NULL "
    "AND course_id IS NOT NULL)"
)


# ---------------------------------------------------------------------------
# Tenancy: Organization, OrgUnit, OrganizationDomain, OrganizationMembership
# ---------------------------------------------------------------------------


class Organization(UUIDPrimaryKeyMixin, TimestampMixin, AuditedByMixin, SoftDeleteMixin, Base):
    """Top-level tenant.

    ``slug`` uniqueness is enforced by a partial unique index
    (``WHERE deleted_at IS NULL``) installed by T0.16
    (migration 0002). The mapped column is therefore not declared
    ``unique=True`` here.
    """

    __tablename__ = "organizations"
    __table_args__ = (
        CheckConstraint(
            "status IN ('active', 'inactive', 'archived')",
            name="ck_organizations_status",
        ),
    )

    slug: Mapped[str] = mapped_column(String(100), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, server_default=text("'active'"))


class OrgUnit(UUIDPrimaryKeyMixin, TimestampMixin, AuditedByMixin, SoftDeleteMixin, Base):
    """Faculty / department / program / campus inside an organization.

    Self-referential via ``parent_unit_id``. Uniqueness on
    ``(organization_id, code)`` is a partial unique index from T0.16.
    """

    __tablename__ = "org_units"
    __table_args__ = (
        CheckConstraint(
            "unit_type IN ('faculty', 'department', 'office', 'program', 'campus', 'other')",
            name="ck_org_units_unit_type",
        ),
    )

    organization_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("organizations.id", ondelete="NO ACTION"),
        nullable=False,
        index=True,
    )
    parent_unit_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("org_units.id", ondelete="NO ACTION"),
    )
    unit_type: Mapped[str] = mapped_column(String(20), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    code: Mapped[str | None] = mapped_column(String(50))

    children: Mapped[list[OrgUnit]] = relationship(
        back_populates="parent",
        remote_side="OrgUnit.parent_unit_id",
        foreign_keys=[parent_unit_id],
        post_update=True,
    )
    parent: Mapped[OrgUnit | None] = relationship(
        back_populates="children",
        remote_side="OrgUnit.id",
        foreign_keys=[parent_unit_id],
        post_update=True,
    )


class OrganizationDomain(
    UUIDPrimaryKeyMixin, TimestampMixin, AuditedByMixin, SoftDeleteMixin, Base
):
    """Email domain → organization auto-routing for OAuth provisioning.

    ``domain`` uniqueness is a T0.16 partial unique index.
    """

    __tablename__ = "organization_domains"

    organization_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("organizations.id", ondelete="NO ACTION"),
        nullable=False,
    )
    domain: Mapped[str] = mapped_column(CITEXT(), nullable=False)
    auto_provision: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("FALSE")
    )


class OrganizationMembership(
    UUIDPrimaryKeyMixin, TimestampMixin, AuditedByMixin, SoftDeleteMixin, Base
):
    """User ↔ organization membership with optional org_unit scope."""

    __tablename__ = "organization_memberships"
    __table_args__ = (
        CheckConstraint(
            "status IN ('active', 'invited', 'inactive', 'suspended', 'left')",
            name="ck_organization_memberships_status",
        ),
    )

    user_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    organization_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("organizations.id", ondelete="NO ACTION"),
        nullable=False,
        index=True,
    )
    org_unit_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("org_units.id", ondelete="SET NULL"),
    )
    status: Mapped[str] = mapped_column(String(20), nullable=False, server_default=text("'active'"))
    student_code: Mapped[str | None] = mapped_column(String(50))
    employee_code: Mapped[str | None] = mapped_column(String(50))
    joined_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("NOW()")
    )
    left_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


# ---------------------------------------------------------------------------
# RBAC catalog: Permission, Role, RolePermission
# ---------------------------------------------------------------------------


class Permission(UUIDPrimaryKeyMixin, TimestampMixin, AuditedByMixin, SoftDeleteMixin, Base):
    """Permission catalog entry (e.g. ``course.read``).

    ``code`` uniqueness is a T0.16 partial unique index.
    """

    __tablename__ = "permissions"

    code: Mapped[str] = mapped_column(String(100), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)


class Role(UUIDPrimaryKeyMixin, TimestampMixin, AuditedByMixin, SoftDeleteMixin, Base):
    """Role definition (``student`` | ``teacher`` | ``hod`` | ``manager``
    | ``admin``).

    ``code`` uniqueness is a T0.16 partial unique index.
    """

    __tablename__ = "roles"

    code: Mapped[str] = mapped_column(String(50), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    is_system_role: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("TRUE")
    )


class RolePermission(CreatedAtMixin, Base):
    """Composite-PK link table between roles and permissions.

    No soft-delete (matches baseline migration; not in
    ``SOFT_DELETE_TABLES``).
    """

    __tablename__ = "role_permissions"

    role_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("roles.id", ondelete="NO ACTION"),
        primary_key=True,
    )
    permission_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("permissions.id", ondelete="NO ACTION"),
        primary_key=True,
    )


# ---------------------------------------------------------------------------
# RBAC assignments: UserRoleAssignment, UserPermissionGrant
# ---------------------------------------------------------------------------


class UserRoleAssignment(
    UUIDPrimaryKeyMixin, TimestampMixin, AuditedByMixin, SoftDeleteMixin, Base
):
    """Assigns a role to a user at one of four scopes.

    The ``ck_user_role_assignments_scope_kind`` CHECK constraint enforces
    the 4-scope coverage matrix and is the foundation T1.4a (FIX-CRIT-2)
    builds on.
    """

    __tablename__ = "user_role_assignments"
    __table_args__ = (
        CheckConstraint(
            "scope_kind IN ('global', 'organization', 'org_unit', 'course')",
            name="ck_user_role_assignments_scope_kind_enum",
        ),
        CheckConstraint(
            _SCOPE_KIND_CHECK,
            name="ck_user_role_assignments_scope_kind",
        ),
    )

    user_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    role_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("roles.id", ondelete="NO ACTION"),
        nullable=False,
    )
    scope_kind: Mapped[str] = mapped_column(String(20), nullable=False)
    organization_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("organizations.id", ondelete="NO ACTION"),
    )
    org_unit_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("org_units.id", ondelete="NO ACTION"),
    )
    course_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("courses.id", ondelete="NO ACTION"),
    )
    granted_by: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
    )
    active_from: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("NOW()")
    )
    active_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    # Course-scoped TEACHER title (user decision 2026-08-18): "no catalog logic
    # for titles". NULL (the default) for every non-course or non-teacher row;
    # 'course_instructor' | 'teacher_assistant' for course-scoped teacher rows.
    # At most one Course Instructor per course is enforced by the partial
    # unique index ``uq_course_teachers_one_instructor``; at least one is
    # enforced by the assignment service (a course with teachers has exactly
    # one Course Instructor, others are Teacher Assistants).
    course_role: Mapped[str | None] = mapped_column(String(30))


class UserPermissionGrant(
    UUIDPrimaryKeyMixin, TimestampMixin, AuditedByMixin, SoftDeleteMixin, Base
):
    """Direct permission grant to a user (bypassing the role catalog).

    Same 4-scope CHECK pattern as :class:`UserRoleAssignment`.
    """

    __tablename__ = "user_permission_grants"
    __table_args__ = (
        CheckConstraint(
            "scope_kind IN ('global', 'organization', 'org_unit', 'course')",
            name="ck_user_permission_grants_scope_kind_enum",
        ),
        CheckConstraint(
            _SCOPE_KIND_CHECK,
            name="ck_user_permission_grants_scope_kind",
        ),
    )

    user_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    permission_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("permissions.id", ondelete="NO ACTION"),
        nullable=False,
    )
    scope_kind: Mapped[str] = mapped_column(String(20), nullable=False)
    organization_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("organizations.id", ondelete="NO ACTION"),
    )
    org_unit_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("org_units.id", ondelete="NO ACTION"),
    )
    course_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("courses.id", ondelete="NO ACTION"),
    )
    granted_by: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
    )
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


# ---------------------------------------------------------------------------
# Career paths: CareerPath, StudentCareerEnrollment
#
# Note: ``CareerCourseItem`` (the M2M between CareerPath and Course) lives
# in ``features/career_paths/models.py`` per Reconciliation §A1. It is
# added in Phase 7 (T7.3) and is intentionally NOT declared here.
# ---------------------------------------------------------------------------


class CareerPath(UUIDPrimaryKeyMixin, TimestampMixin, AuditedByMixin, SoftDeleteMixin, Base):
    """Career track owned by an organization.

    ``description`` is the Reconciliation §B6 net-new column added in the
    T0.9 baseline. ``(organization_id, slug)`` uniqueness is a T0.16
    partial unique index.
    """

    __tablename__ = "career_paths"
    __table_args__ = (
        CheckConstraint(
            "status IN ('draft', 'published', 'archived')",
            name="ck_career_paths_status",
        ),
        CheckConstraint(
            "max_concurrent IS NULL OR max_concurrent > 0",
            name="career_paths_max_concurrent_check",
        ),
    )

    organization_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("organizations.id", ondelete="NO ACTION"),
        nullable=False,
    )
    org_unit_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("org_units.id", ondelete="SET NULL"),
    )
    slug: Mapped[str] = mapped_column(String(100), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(20), nullable=False, server_default=text("'draft'"))
    max_concurrent: Mapped[int | None] = mapped_column(Integer)
    """Attention cap: how many courses of THIS path a student should have in
    flight at once. ``NULL`` ⇒ unlimited.

    Path-level on purpose. It is evaluated against a **path-wide** count of
    the student's ``active`` course enrollments, so a stage-level column
    would have compared a stage-scoped cap to a path-scoped count. It is
    advisory in the strict sense: exceeding it returns a warning flag, never
    a 4xx — not even under ``enforcement='hard'``, which governs stage
    *lock* only.
    """

    enrollments: Mapped[list[StudentCareerEnrollment]] = relationship(
        back_populates="career_path",
        cascade="save-update, merge, refresh-expire, expunge",
    )


class StudentCareerEnrollment(
    UUIDPrimaryKeyMixin, TimestampMixin, AuditedByMixin, SoftDeleteMixin, Base
):
    """Student → career path enrollment with lifecycle status.

    ``(career_path_id, student_id)`` uniqueness is a T0.16 partial unique
    index.
    """

    __tablename__ = "student_career_enrollments"
    __table_args__ = (
        CheckConstraint(
            "status IN ('active', 'completed', 'dropped')",
            name="ck_student_career_enrollments_status",
        ),
    )

    career_path_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("career_paths.id", ondelete="NO ACTION"),
        nullable=False,
    )
    # Gap 3 (0074): THE version pin. D3(a) — the student finishes the
    # version they started on; a manager editing the path edits a NEW
    # version, so this enrollment's route never changes under them.
    version_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("career_path_versions.id", ondelete="NO ACTION"),
        nullable=False,
        index=True,
    )
    student_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    status: Mapped[str] = mapped_column(String(20), nullable=False, server_default=text("'active'"))
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("NOW()")
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    career_path: Mapped[CareerPath] = relationship(back_populates="enrollments")


__all__ = [
    "CareerPath",
    "Organization",
    "OrganizationDomain",
    "OrganizationMembership",
    "OrgUnit",
    "Permission",
    "Role",
    "RolePermission",
    "StudentCareerEnrollment",
    "UserPermissionGrant",
    "UserRoleAssignment",
]
