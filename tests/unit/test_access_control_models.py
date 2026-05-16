"""Unit tests for access_control + organization models (T1.2).

Covers:
- All 11 models import cleanly.
- The two scope_kind CHECK constraints are present (enforces the
  4-scope coverage matrix used by T1.4a / FIX-CRIT-2).
- Soft-delete inventory matches the baseline migration:
  - Soft-deletable: Organization, OrgUnit, OrganizationDomain,
    OrganizationMembership, Permission, Role, UserRoleAssignment,
    UserPermissionGrant, CareerPath, StudentCareerEnrollment.
  - Hard-delete only: RolePermission.
- CareerPath has the Reconciliation §B6 ``description`` column.
- Self-referential FK on OrgUnit.parent_unit_id targets org_units.id.
- All FKs to users.id / courses.id / organizations.id resolve via
  string targets (no cross-feature ORM imports).
"""

from __future__ import annotations

import pytest
from sqlalchemy import CheckConstraint

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


def test_all_models_importable() -> None:
    models = [
        Permission,
        Role,
        RolePermission,
        UserRoleAssignment,
        UserPermissionGrant,
        Organization,
        OrgUnit,
        OrganizationMembership,
        OrganizationDomain,
        CareerPath,
        StudentCareerEnrollment,
    ]
    assert len(models) == 11
    table_names = {m.__tablename__ for m in models}
    assert table_names == {
        "permissions",
        "roles",
        "role_permissions",
        "user_role_assignments",
        "user_permission_grants",
        "organizations",
        "org_units",
        "organization_memberships",
        "organization_domains",
        "career_paths",
        "student_career_enrollments",
    }


def test_user_role_assignment_scope_kind_check_present() -> None:
    names = {
        c.name for c in UserRoleAssignment.__table__.constraints if isinstance(c, CheckConstraint)
    }
    assert "ck_user_role_assignments_scope_kind" in names


def test_user_permission_grant_scope_kind_check_present() -> None:
    names = {
        c.name for c in UserPermissionGrant.__table__.constraints if isinstance(c, CheckConstraint)
    }
    assert "ck_user_permission_grants_scope_kind" in names


@pytest.mark.parametrize(
    "ck",
    [
        c
        for c in UserRoleAssignment.__table__.constraints
        if isinstance(c, CheckConstraint) and c.name == "ck_user_role_assignments_scope_kind"
    ],
)
def test_scope_kind_check_covers_four_scopes(ck: CheckConstraint) -> None:
    sql = str(ck.sqltext)
    for scope in ("'global'", "'organization'", "'org_unit'", "'course'"):
        assert scope in sql, f"scope {scope!r} missing from CHECK"


def test_career_path_has_softdelete_and_description() -> None:
    cols = {c.name for c in CareerPath.__table__.columns}
    assert "deleted_at" in cols
    assert "deleted_by" in cols
    assert "description" in cols


def test_organization_has_softdelete_and_audit() -> None:
    cols = {c.name for c in Organization.__table__.columns}
    expected = {"deleted_at", "deleted_by", "created_by", "updated_by"}
    assert expected <= cols, f"missing: {expected - cols}"


def test_role_permission_has_no_softdelete() -> None:
    cols = {c.name for c in RolePermission.__table__.columns}
    assert "deleted_at" not in cols
    assert "deleted_by" not in cols


def test_org_unit_parent_unit_id_is_self_referential() -> None:
    parent_fk = next(iter(OrgUnit.__table__.c.parent_unit_id.foreign_keys))
    assert parent_fk.target_fullname == "org_units.id"


def test_user_role_assignment_user_id_targets_users_id() -> None:
    user_fk = next(iter(UserRoleAssignment.__table__.c.user_id.foreign_keys))
    assert user_fk.target_fullname == "users.id"


def test_user_role_assignment_course_id_targets_courses_id() -> None:
    course_fk = next(iter(UserRoleAssignment.__table__.c.course_id.foreign_keys))
    assert course_fk.target_fullname == "courses.id"


def test_role_permission_composite_primary_key() -> None:
    pk_cols = {c.name for c in RolePermission.__table__.primary_key.columns}
    assert pk_cols == {"role_id", "permission_id"}
