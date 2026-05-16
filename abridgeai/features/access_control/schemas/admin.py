"""Pydantic v2 schemas for the access-control admin router (T1.10)."""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

ScopeKind = Literal["global", "organization", "org_unit", "course"]


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
    status: Literal["active", "invited", "inactive", "suspended", "left"] = "active"
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


__all__ = [
    "GrantCreate",
    "GrantRead",
    "MembershipCreate",
    "MembershipRead",
    "PermissionRead",
    "RoleAssignmentCreate",
    "RoleAssignmentRead",
    "RoleRead",
    "RoleWithPermissionsRead",
    "ScopeKind",
]
