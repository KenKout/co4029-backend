"""Pydantic DTOs returned by :mod:`access_control.api.public`.

ORM models (``UserRoleAssignment``, ``OrgUnit``, ``Organization``,
``Permission``) MUST NOT escape the access_control feature; cross-feature
callers receive these immutable, typed read-models instead.

All DTOs are ``model_config = ConfigDict(frozen=True)`` so consumers
cannot mutate cached results.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class _BaseDTO(BaseModel):
    model_config = ConfigDict(frozen=True, from_attributes=True)


class OrgDTO(_BaseDTO):
    """Tenant organisation read-model.

    Mirrors the projected columns of ``organizations`` consumed cross
    feature. Excludes audit columns (``created_by``/``updated_by``) and
    soft-delete metadata: callers must not need them.
    """

    id: UUID
    slug: str
    name: str
    status: str


class OrgUnitDTO(_BaseDTO):
    """Faculty / department / program / campus read-model.

    ``parent_unit_id`` is ``None`` at the root of the hierarchy. ``depth``
    is the 0-indexed distance from the requested unit (0 == self,
    1 == direct parent, …) populated by ancestor walks.
    """

    id: UUID
    organization_id: UUID
    parent_unit_id: UUID | None
    unit_type: str
    name: str
    code: str | None
    depth: int = 0


class PermissionDTO(_BaseDTO):
    id: UUID
    code: str
    name: str
    description: str | None


class RoleAssignmentDTO(_BaseDTO):
    """Effective ``user_role_assignments`` row, scope kept opaque."""

    id: UUID
    user_id: UUID
    role_id: UUID
    role_code: str
    scope_kind: str
    organization_id: UUID | None
    org_unit_id: UUID | None
    course_id: UUID | None
    active_from: datetime
    active_until: datetime | None


__all__ = [
    "OrgDTO",
    "OrgUnitDTO",
    "PermissionDTO",
    "RoleAssignmentDTO",
]
