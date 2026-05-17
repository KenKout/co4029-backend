"""Resolve caller's organization scope for admin queries (T7.5).

A caller with ``system.administer`` bypasses scoping (returns ``None`` ->
global view). A caller with only ``system.stats.read`` (Manager / HOD) gets
their organization derived from ``organization_memberships`` (preferred) or
``user_role_assignments`` as a fallback. Returns ``None`` when no
organization can be resolved -- callers MUST treat this as "no data
visible" rather than as global access.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import text

from abridgeai.core.security import CurrentUser

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


_RESOLVE_ORG_SQL = text(
    """
    SELECT om.organization_id
    FROM organization_memberships om
    WHERE om.user_id = :user_id
      AND om.deleted_at IS NULL
      AND om.status = 'active'
    UNION
    SELECT ura.organization_id
    FROM user_role_assignments ura
    WHERE ura.user_id = :user_id
      AND ura.deleted_at IS NULL
      AND ura.organization_id IS NOT NULL
      AND ura.active_from <= NOW()
      AND (ura.active_until IS NULL OR ura.active_until > NOW())
    LIMIT 1
    """
)


async def resolve_admin_scope(db: AsyncSession, current_user: CurrentUser) -> UUID | None:
    """Return the organization filter for ``current_user``.

    * IT Admin (holds ``system.administer``) -> ``None`` (global).
    * Other holders -> their resolved organization or ``None`` if unresolved.
    """
    if "system.administer" in current_user.permissions:
        return None
    row = (await db.execute(_RESOLVE_ORG_SQL, {"user_id": current_user.user_id})).first()
    if row is None:
        return None
    org_id: UUID = row[0]
    return org_id


__all__ = ["resolve_admin_scope"]
