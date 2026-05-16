"""Access control feature: RBAC, tenancy, career paths.

Re-exports the ORM models so callers can use the public path
``abridgeai.features.access_control``. Service / repository / router
modules land in later Phase 1 tasks.
"""

from abridgeai.features.access_control.models import (
    CareerPath,
    Organization,
    OrganizationDomain,
    OrganizationMembership,
    OrgUnit,
    Permission,
    Role,
    RolePermission,
    StudentCareerEnrollment,
    UserPermissionGrant,
    UserRoleAssignment,
)

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
