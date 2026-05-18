"""Resolve caller's organization scope for admin queries (T7.5).

A caller with ``system.administer`` bypasses scoping (returns ``None`` ->
global view). A caller with only ``system.stats.read`` (Manager / HOD) gets
their organization derived from active ``user_role_assignments`` via
:func:`access_control.api.public.get_user_primary_org`. Returns ``None``
when no organization can be resolved -- callers MUST treat this as "no
data visible" rather than as global access.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

from abridgeai.core.security import CurrentUser
from abridgeai.features.access_control.api import public as access_control_api

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


async def resolve_admin_scope(db: AsyncSession, current_user: CurrentUser) -> UUID | None:
    """Return the organization filter for ``current_user``.

    * IT Admin (holds ``system.administer``) -> ``None`` (global).
    * Other holders -> their resolved organization or ``None`` if unresolved.
    """
    if "system.administer" in current_user.permissions:
        return None
    org = await access_control_api.get_user_primary_org(db, current_user.user_id)
    return org.id if org is not None else None


__all__ = ["resolve_admin_scope"]
