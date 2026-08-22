"""Pydantic v2 schemas for the access-control admin router (T1.10)."""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

ScopeKind = Literal["global", "organization", "org_unit", "course"]
OrganizationStatus = Literal["active", "inactive", "archived"]
UnitType = Literal["faculty", "department", "office", "program", "campus", "other"]
MembershipStatus = Literal["active", "inactive", "suspended"]


class _ORM(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class PermissionRead(_ORM):
    id: UUID
    code: str
    name: str
    description: str | None = None


class RoleRead(_ORM):
    id: UUID
    code: str
    name: str
    is_system_role: bool


class RoleWithPermissionsRead(BaseModel):
    role: RoleRead
    permissions: list[str]


class RoleAssignmentCreate(BaseModel):
    role_code: str = Field(min_length=1, max_length=50)
    scope_kind: ScopeKind
    organization_id: UUID | None = None
    org_unit_id: UUID | None = None
    course_id: UUID | None = None
    active_until: datetime | None = None


class RoleAssignmentRead(_ORM):
    id: UUID
    user_id: UUID
    role_id: UUID
    scope_kind: str
    organization_id: UUID | None = None
    org_unit_id: UUID | None = None
    course_id: UUID | None = None
    granted_by: UUID | None = None
    active_from: datetime
    active_until: datetime | None = None


class GrantCreate(BaseModel):
    permission_id: UUID
    scope_kind: ScopeKind
    organization_id: UUID | None = None
    org_unit_id: UUID | None = None
    course_id: UUID | None = None
    expires_at: datetime | None = None


class GrantRead(_ORM):
    id: UUID
    user_id: UUID
    permission_id: UUID
    scope_kind: str
    organization_id: UUID | None = None
    org_unit_id: UUID | None = None
    course_id: UUID | None = None
    granted_by: UUID | None = None
    expires_at: datetime | None = None


class MembershipCreate(BaseModel):
    user_id: UUID
    org_unit_id: UUID | None = None
    status: MembershipStatus = "active"
    student_code: str | None = Field(default=None, max_length=50)
    employee_code: str | None = Field(default=None, max_length=50)


class MembershipPatch(BaseModel):
    """Partial update for a membership row.

    All fields are optional. ``status`` transitions to ``left`` should
    additionally stamp ``left_at``; the service layer applies that.
    """

    model_config = ConfigDict(extra="forbid")

    org_unit_id: UUID | None = None
    status: MembershipStatus | None = None
    student_code: str | None = Field(default=None, max_length=50)
    employee_code: str | None = Field(default=None, max_length=50)


class MembershipRead(_ORM):
    id: UUID
    user_id: UUID
    organization_id: UUID
    org_unit_id: UUID | None = None
    status: str
    student_code: str | None = None
    employee_code: str | None = None
    joined_at: datetime
    left_at: datetime | None = None


# ---------------------------------------------------------------------------
# Organizations
# ---------------------------------------------------------------------------


class OrganizationCreate(BaseModel):
    """Create payload for ``POST /admin/organizations``.

    ``slug`` must be unique among non-deleted rows (partial unique index
    from migration 0002). ``status`` defaults to ``active`` server-side.
    """

    model_config = ConfigDict(extra="forbid")

    slug: str = Field(min_length=1, max_length=100, pattern=r"^[a-z0-9][a-z0-9-]*$")
    name: str = Field(min_length=1, max_length=255)
    status: OrganizationStatus = "active"


class OrganizationPatch(BaseModel):
    """Partial update for ``PATCH /admin/organizations/{id}``."""

    model_config = ConfigDict(extra="forbid")

    slug: str | None = Field(
        default=None,
        min_length=1,
        max_length=100,
        pattern=r"^[a-z0-9][a-z0-9-]*$",
    )
    name: str | None = Field(default=None, min_length=1, max_length=255)
    status: OrganizationStatus | None = None


class OrganizationRead(_ORM):
    id: UUID
    slug: str
    name: str
    status: str
    created_at: datetime
    updated_at: datetime


class OrganizationListPage(BaseModel):
    """Cursor-paginated organisation listing.

    ``next_cursor`` is opaque and round-trips through subsequent calls.
    Set when the page filled to ``limit`` (more rows may exist); ``None``
    otherwise. Reconciliation §A10/§D2: cursor pagination, not offset.
    """

    items: list[OrganizationRead]
    next_cursor: str | None = None


# ---------------------------------------------------------------------------
# Organization domains
# ---------------------------------------------------------------------------


class OrganizationDomainCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    domain: str = Field(min_length=1, max_length=255)
    auto_provision: bool = False


class OrganizationDomainPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    domain: str | None = Field(default=None, min_length=1, max_length=255)
    auto_provision: bool | None = None


class OrganizationDomainRead(_ORM):
    id: UUID
    organization_id: UUID
    domain: str
    auto_provision: bool
    created_at: datetime
    updated_at: datetime


# ---------------------------------------------------------------------------
# Org units
# ---------------------------------------------------------------------------


class OrgUnitCreate(BaseModel):
    """Create payload for ``POST /admin/organizations/{org_id}/units``."""

    model_config = ConfigDict(extra="forbid")

    parent_unit_id: UUID | None = None
    unit_type: UnitType
    name: str = Field(min_length=1, max_length=255)
    code: str | None = Field(default=None, max_length=50)


class OrgUnitPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    parent_unit_id: UUID | None = None
    unit_type: UnitType | None = None
    name: str | None = Field(default=None, min_length=1, max_length=255)
    code: str | None = Field(default=None, max_length=50)


class OrgUnitNode(_ORM):
    """One node of the nested org tree returned by ``GET .../units/tree``.

    Same columns as :class:`OrgUnitRead` plus the two the tree UI needs and
    a flat list cannot supply:

    ``children``
        Populated depth-first by the service; siblings sorted by name.
    ``descendant_count``
        Total units BELOW this one. Drives the "deleting this also deletes
        N sub-units" confirmation — the delete cascades down the subtree,
        so the count has to be visible before the click.
    """

    id: UUID
    organization_id: UUID
    parent_unit_id: UUID | None = None
    unit_type: str
    name: str
    code: str | None = None
    created_at: datetime
    updated_at: datetime
    children: list[OrgUnitNode] = Field(default_factory=list)
    descendant_count: int = 0


class OrgUnitRead(_ORM):
    """Standalone org-unit row (different from the cross-feature
    :class:`abridgeai.features.access_control.api._dto.OrgUnitDTO` which
    is read-only and excludes audit columns).
    """

    id: UUID
    organization_id: UUID
    parent_unit_id: UUID | None = None
    unit_type: str
    name: str
    code: str | None = None
    created_at: datetime
    updated_at: datetime


__all__ = [
    "GrantCreate",
    "GrantRead",
    "MembershipCreate",
    "MembershipPatch",
    "MembershipRead",
    "MembershipStatus",
    "OrgUnitCreate",
    "OrgUnitPatch",
    "OrgUnitRead",
    "OrganizationCreate",
    "OrganizationDomainCreate",
    "OrganizationDomainPatch",
    "OrganizationDomainRead",
    "OrganizationPatch",
    "OrganizationRead",
    "OrganizationStatus",
    "PermissionRead",
    "RoleAssignmentCreate",
    "RoleAssignmentRead",
    "RoleRead",
    "RoleWithPermissionsRead",
    "ScopeKind",
    "UnitType",
]
