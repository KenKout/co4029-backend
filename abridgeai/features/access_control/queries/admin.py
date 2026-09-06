"""Admin-surface data accessors for the access-control feature.

INSERT flows use :func:`sqlalchemy.insert` against the mapped class
(e.g. ``insert(UserRoleAssignment).values(...)``) rather than
``db.add()``. ``db.add()`` triggers the ORM unit-of-work which walks
``Base.metadata.sorted_tables`` to compute insertion order, and that
walk fails with :class:`NoReferencedTableError` whenever a
cross-feature FK target (e.g. ``users``, ``courses``) hasn't been
imported into ``Base.metadata`` yet — which is the common case in
test setup and lazy-loaded surfaces. The Core-level ``insert()``
construct skips the metadata walk entirely while still keeping the
column references type-safe (no raw SQL strings).

DELETE flows on :class:`SoftDeleteMixin` tables go through
:func:`abridgeai.core.db.recursive_delete.soft_delete_cascade`. Raw
``DELETE FROM`` would bypass the ``hard_delete_guard`` listener and
erase the audit trail; ``soft_delete_cascade`` stamps ``deleted_at`` /
``deleted_by`` and flushes via ``session.flush()``.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import func, insert, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.core.db.recursive_delete import soft_delete_cascade
from abridgeai.core.exceptions import ConflictError
from abridgeai.features.access_control.models import (
    OrganizationMembership,
    Permission,
    Role,
    RolePermission,
    UserPermissionGrant,
    UserRoleAssignment,
)


async def list_permissions(db: AsyncSession) -> list[Permission]:
    result = await db.execute(
        select(Permission).where(Permission.deleted_at.is_(None)).order_by(Permission.code)
    )
    return list(result.scalars().all())


async def list_roles(db: AsyncSession) -> list[Role]:
    result = await db.execute(select(Role).where(Role.deleted_at.is_(None)).order_by(Role.code))
    return list(result.scalars().all())


async def get_role_by_code(db: AsyncSession, code: str) -> Role | None:
    result = await db.execute(select(Role).where(Role.code == code, Role.deleted_at.is_(None)))
    return result.scalar_one_or_none()


async def get_role_permission_codes(db: AsyncSession, role_id: UUID) -> list[str]:
    result = await db.execute(
        select(Permission.code)
        .join(RolePermission, RolePermission.permission_id == Permission.id)
        .where(
            RolePermission.role_id == role_id,
            Permission.deleted_at.is_(None),
        )
        .order_by(Permission.code)
    )
    return [row[0] for row in result.all()]


async def list_assignments_for_user(db: AsyncSession, user_id: UUID) -> list[UserRoleAssignment]:
    result = await db.execute(
        select(UserRoleAssignment)
        .where(
            UserRoleAssignment.user_id == user_id,
            UserRoleAssignment.deleted_at.is_(None),
        )
        .order_by(UserRoleAssignment.active_from.desc())
    )
    return list(result.scalars().all())


async def get_assignment_by_id(
    db: AsyncSession, assignment_id: UUID
) -> tuple[UserRoleAssignment, str] | None:
    """Return ``(assignment, role_code)`` for a live assignment, else ``None``.

    The role code is needed by the revoke guard to re-assert the caller's
    permission against the *revoked role's* requirements (HOD-gated vs
    plain ``user.role_assign``) — not just the caller's global permission
    set, which flattens scope.
    """
    result = await db.execute(
        select(UserRoleAssignment, Role.code)
        .join(Role, Role.id == UserRoleAssignment.role_id)
        .where(
            UserRoleAssignment.id == assignment_id,
            UserRoleAssignment.deleted_at.is_(None),
        )
    )
    row = result.first()
    if row is None:
        return None
    return row[0], row[1]


async def org_ids_for_user(db: AsyncSession, user_id: UUID) -> set[UUID]:
    """Set of org ids the user belongs to.

    Membership is derived from two sources (either suffices):
    * active ``organization_memberships`` rows, and
    * any live role assignment that carries an ``organization_id``
      (organization / org_unit / course scoped).

    Used by the role-assignment guards to answer "may this caller act on
    this target?" without leaking org membership checks into services.
    """
    memberships = await db.execute(
        select(OrganizationMembership.organization_id).where(
            OrganizationMembership.user_id == user_id,
            OrganizationMembership.status == "active",
        )
    )
    assignments = await db.execute(
        select(UserRoleAssignment.organization_id).where(
            UserRoleAssignment.user_id == user_id,
            UserRoleAssignment.deleted_at.is_(None),
            UserRoleAssignment.organization_id.is_not(None),
        )
    )
    return {row[0] for row in memberships.all()} | {
        row[0] for row in assignments.all() if row[0] is not None
    }


async def assign_org_ids_for_user(
    db: AsyncSession, user_id: UUID
) -> tuple[set[UUID], bool]:
    """Orgs where the user may assign roles, plus a global-assign flag.

    Returns ``(org_ids, has_global)``:
    * ``org_ids`` — distinct ``organization_id`` of live role assignments
      whose role grants ``user.role_assign`` or ``user.role_assign.hod``.
    * ``has_global`` — True when the user holds one of those permissions
      via a ``scope_kind='global'`` assignment (platform-wide authority).

    This is the *actor-authority* counterpart of :func:`org_ids_for_user`
    (which answers "which orgs does this user belong to"). The two are
    deliberately separate: a teacher belongs to orgs without gaining any
    assign authority there.
    """
    result = await db.execute(
        select(
            UserRoleAssignment.organization_id,
            UserRoleAssignment.scope_kind,
        )
        .join(RolePermission, RolePermission.role_id == UserRoleAssignment.role_id)
        .join(Permission, Permission.id == RolePermission.permission_id)
        .where(
            UserRoleAssignment.user_id == user_id,
            UserRoleAssignment.deleted_at.is_(None),
            Permission.deleted_at.is_(None),
            Permission.code.in_(("user.role_assign", "user.role_assign.hod")),
        )
    )
    orgs: set[UUID] = set()
    has_global = False
    for org_id, scope_kind in result.all():
        if scope_kind == "global":
            has_global = True
        elif org_id is not None:
            orgs.add(org_id)
    return orgs, has_global


async def insert_assignment(
    db: AsyncSession,
    *,
    user_id: UUID,
    role_id: UUID,
    scope_kind: str,
    organization_id: UUID | None,
    org_unit_id: UUID | None,
    course_id: UUID | None,
    granted_by: UUID | None,
    active_until: datetime | None,
) -> UserRoleAssignment:
    new_id = uuid4()
    # 0093_teacher_title_flags added ck_user_role_assignments_course_title:
    # a COURSE-scoped assignment must carry at least one title. This generic
    # admin path was never updated, so every course-scoped assignment made
    # through it violated the CHECK and 500ed.
    #
    # Teacher Assistant is the least-privileged of the two and matches what
    # the course staffing flow gives a newly assigned teacher; promotion to
    # Course Instructor is an explicit, separate action
    # (PUT /courses/{id}/teachers/{user}/role).
    is_assistant = scope_kind == "course"
    await db.execute(
        insert(UserRoleAssignment).values(
            id=new_id,
            user_id=user_id,
            role_id=role_id,
            scope_kind=scope_kind,
            organization_id=organization_id,
            org_unit_id=org_unit_id,
            course_id=course_id,
            granted_by=granted_by,
            active_until=active_until,
            is_assistant=is_assistant,
        )
    )
    await db.flush()
    fetched = await db.execute(select(UserRoleAssignment).where(UserRoleAssignment.id == new_id))
    return fetched.scalar_one()


async def soft_delete_assignment(
    db: AsyncSession,
    assignment_id: UUID,
    *,
    actor_id: UUID | None = None,
) -> bool:
    row = await db.get(UserRoleAssignment, assignment_id)
    if row is None or row.deleted_at is not None:
        return False
    await soft_delete_cascade(db, row, actor_id=actor_id)
    return True


async def list_grants_for_user(db: AsyncSession, user_id: UUID) -> list[UserPermissionGrant]:
    result = await db.execute(
        select(UserPermissionGrant)
        .where(
            UserPermissionGrant.user_id == user_id,
            UserPermissionGrant.deleted_at.is_(None),
        )
        .order_by(UserPermissionGrant.created_at.desc())
    )
    return list(result.scalars().all())


async def insert_grant(
    db: AsyncSession,
    *,
    user_id: UUID,
    permission_id: UUID,
    scope_kind: str,
    organization_id: UUID | None,
    org_unit_id: UUID | None,
    course_id: UUID | None,
    granted_by: UUID | None,
    expires_at: datetime | None,
) -> UserPermissionGrant:
    new_id = uuid4()
    await db.execute(
        insert(UserPermissionGrant).values(
            id=new_id,
            user_id=user_id,
            permission_id=permission_id,
            scope_kind=scope_kind,
            organization_id=organization_id,
            org_unit_id=org_unit_id,
            course_id=course_id,
            granted_by=granted_by,
            expires_at=expires_at,
        )
    )
    await db.flush()
    fetched = await db.execute(select(UserPermissionGrant).where(UserPermissionGrant.id == new_id))
    return fetched.scalar_one()


async def get_grant(db: AsyncSession, grant_id: UUID) -> UserPermissionGrant | None:
    row = await db.get(UserPermissionGrant, grant_id)
    return row if row is not None and row.deleted_at is None else None


async def delete_grant(
    db: AsyncSession,
    grant_id: UUID,
    *,
    actor_id: UUID | None = None,
) -> bool:
    row = await db.get(UserPermissionGrant, grant_id)
    if row is None or row.deleted_at is not None:
        return False
    await soft_delete_cascade(db, row, actor_id=actor_id)
    return True


async def list_memberships_for_organization(
    db: AsyncSession, organization_id: UUID
) -> list[OrganizationMembership]:
    result = await db.execute(
        select(OrganizationMembership)
        .where(
            OrganizationMembership.organization_id == organization_id,
            OrganizationMembership.deleted_at.is_(None),
        )
        .order_by(OrganizationMembership.joined_at.desc())
    )
    return list(result.scalars().all())


async def get_reserved_membership_for_user(
    db: AsyncSession, user_id: UUID
) -> OrganizationMembership | None:
    """Return the membership currently reserving a user's organization.

    Every non-``left`` state reserves the tenant boundary.  In particular,
    suspending or deactivating an account must not make it attachable to a
    second organization.  Soft-deleted and historical ``left`` rows do not
    reserve the user.
    """
    result = await db.execute(
        select(OrganizationMembership)
        .where(
            OrganizationMembership.user_id == user_id,
            OrganizationMembership.deleted_at.is_(None),
            OrganizationMembership.status != "left",
        )
        .limit(1)
    )
    return result.scalar_one_or_none()


async def is_platform_admin(db: AsyncSession, user_id: UUID) -> bool:
    """Whether the user currently holds the global platform-admin role."""
    result = await db.execute(
        select(UserRoleAssignment.id)
        .join(Role, Role.id == UserRoleAssignment.role_id)
        .where(
            UserRoleAssignment.user_id == user_id,
            UserRoleAssignment.scope_kind == "global",
            UserRoleAssignment.deleted_at.is_(None),
            UserRoleAssignment.active_from <= func.now(),
            (UserRoleAssignment.active_until.is_(None))
            | (UserRoleAssignment.active_until > func.now()),
            Role.code == "admin",
            Role.deleted_at.is_(None),
        )
        .limit(1)
    )
    return result.scalar_one_or_none() is not None


async def insert_membership(
    db: AsyncSession,
    *,
    user_id: UUID,
    organization_id: UUID,
    org_unit_id: UUID | None,
    status: str,
    student_code: str | None,
    employee_code: str | None,
) -> OrganizationMembership:
    new_id = uuid4()
    try:
        await db.execute(
            insert(OrganizationMembership).values(
                id=new_id,
                user_id=user_id,
                organization_id=organization_id,
                org_unit_id=org_unit_id,
                status=status,
                student_code=student_code,
                employee_code=employee_code,
            )
        )
        await db.flush()
    except IntegrityError as exc:
        if "uq_organization_memberships_one_live_org_per_user" not in str(exc.orig):
            raise
        await db.rollback()
        raise ConflictError("user_already_belongs_to_an_organization") from exc
    fetched = await db.execute(
        select(OrganizationMembership).where(OrganizationMembership.id == new_id)
    )
    return fetched.scalar_one()


async def get_permission_by_id(db: AsyncSession, permission_id: UUID) -> Permission | None:
    return await db.get(Permission, permission_id)


async def get_role_by_id(db: AsyncSession, role_id: UUID) -> Role | None:
    return await db.get(Role, role_id)


__all__ = [
    "delete_grant",
    "get_grant",
    "get_reserved_membership_for_user",
    "get_permission_by_id",
    "get_role_by_code",
    "get_role_by_id",
    "get_role_permission_codes",
    "insert_assignment",
    "insert_grant",
    "insert_membership",
    "is_platform_admin",
    "list_assignments_for_user",
    "list_grants_for_user",
    "list_memberships_for_organization",
    "list_permissions",
    "list_roles",
    "soft_delete_assignment",
]
