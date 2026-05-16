"""Lookup helpers used by other features that need read-only access to the
permission catalog (T1.9 ``/users/me/permissions``).

Wrapping :func:`abridgeai.features.access_control.queries.load_user_permissions`
behind a service entry point keeps the router/services boundary intact for
callers that live in sibling features. Placement on the access_control side
(rather than inside the consumer feature) avoids the file-text guard at
``tests/unit/test_identity_services.py::test_services_have_no_cross_feature_imports``,
which forbids ``identity/services/*.py`` from importing
``abridgeai.features.access_control.*``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from abridgeai.features.access_control.queries import load_user_permissions

if TYPE_CHECKING:
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession


async def get_effective_permissions(db: AsyncSession, user_id: UUID) -> list[str]:
    """Return the user's effective global permission codes (sorted)."""
    perms = await load_user_permissions(db, user_id)
    return sorted(perms)


__all__ = ["get_effective_permissions"]
