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

from sqlalchemy import insert, select
from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.core.db.recursive_delete import soft_delete_cascade
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
    "get_permission_by_id",
    "get_role_by_code",
    "get_role_by_id",
    "get_role_permission_codes",
    "insert_assignment",
    "insert_grant",
    "insert_membership",
    "list_assignments_for_user",
    "list_grants_for_user",
    "list_memberships_for_organization",
    "list_permissions",
    "list_roles",
    "soft_delete_assignment",
]
