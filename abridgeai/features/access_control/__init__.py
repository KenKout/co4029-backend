"""Access control feature: RBAC, tenancy, career paths.

Re-exports ORM models and FastAPI permission policies under the public
package path ``abridgeai.features.access_control``.
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
from abridgeai.features.access_control.policies import (
    PermissionDependency,
    can_manage_course,
    require_any_permission,
    require_course_permission,
    require_org_unit_permission,
    require_permission,
)

__all__ = [
    "CareerPath",
    "Organization",
    "OrganizationDomain",
    "OrganizationMembership",
    "OrgUnit",
    "Permission",
    "PermissionDependency",
    "Role",
    "RolePermission",
    "StudentCareerEnrollment",
    "UserPermissionGrant",
    "UserRoleAssignment",
    "can_manage_course",
    "require_any_permission",
    "require_course_permission",
    "require_org_unit_permission",
    "require_permission",
]
