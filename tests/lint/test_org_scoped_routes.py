"""Every route addressing an org-owned resource must check the org itself.

Why this file exists
--------------------
``load_user_permissions`` flattens role assignments **without regard to
``scope_kind``** (see ``access_control/api/public.py::_ACTIVE_PERMISSIONS_SQL``).
A role granted to a manager at ``scope_kind='organization'`` for org B produces
exactly the same permission codes as a global grant, so
``require_permission("course.update")`` cannot keep that manager out of org A.

Two families of resource exist:

* **course-owned** — protected already. ``require_course_permission`` /
  ``require_material_authoring_access`` re-resolve scope against the course's
  own organization via ``load_course_permissions``, which *does* honour
  ``scope_kind``.
* **org-owned but course-less** — career paths, invitation codes,
  organizations, org units, domains, memberships. There is no course to
  re-resolve against, so the handler must call ``require_org_access`` (or a
  module-local wrapper) after loading the resource. Nothing structural forces
  that, which is how ten of twelve career-path routes and every route in the
  organizations router shipped without it.

This test is the forcing function. It reads handler source rather than
exercising routes, so a newly added endpoint fails here on the day it lands.

Adding a legitimately global endpoint? Put it in ``_GLOBAL_BY_DESIGN`` with a
reason. That list is short on purpose — each entry is an endpoint where any
holder of the permission may act across every tenant.
"""

from __future__ import annotations

import ast
from pathlib import Path

_BACKEND_ROOT = Path(__file__).resolve().parents[2]
_FEATURES = _BACKEND_ROOT / "abridgeai" / "features"

# Path/query parameters that name an org-owned, course-less resource. A handler
# taking one of these is reaching a specific tenant's row.
_ORG_SCOPED_PARAMS = frozenset(
    {
        "career_path_id",
        "code_id",
        "domain_id",
        "membership_id",
        "org_id",
        "organization_id",
        "unit_id",
    }
)

# Call names that constitute an org check. Module-local wrappers are accepted
# because the resolution step (resource -> organization_id) is necessarily
# feature-specific; what matters is that one of them runs.
_ORG_CHECK_CALLS = frozenset(
    {
        "require_org_access",
        "_require_access_to",
        "_ensure_caller_in_path_org",
        "_ensure_caller_in_code_org",
        "_visible_org_ids",
        # org_unit-scoped dependency already resolves the owning organization
        # and walks the unit ancestry.
        "require_org_unit_permission",
    }
)

# (module suffix, handler name) -> why no org check is needed.
#
# Two shapes qualify. Either the endpoint genuinely has no tenant to scope to,
# or the query is already keyed on ``current_user.user_id``, which bounds the
# result to the caller's own rows no matter which id they pass — a learner
# naming another org's career path gets an empty answer, not that org's data.
_GLOBAL_BY_DESIGN: dict[tuple[str, str], str] = {
    (
        "access_control/routers/organizations.py",
        "create_organization_endpoint",
    ): "Creates a new organization; there is no existing tenant to scope to.",
    (
        "career_paths/routers/learner.py",
        "get_my_career_path_progress",
    ): "Keyed on student_id=current_user.user_id; returns only the caller's own enrolment.",
    (
        "career_paths/routers/learner.py",
        "get_my_readiness_history",
    ): "Keyed on student_id=current_user.user_id; returns only the caller's own snapshots.",
    (
        "admin/routers/stats.py",
        "get_dashboard",
    ): (
        "organization_id is a NARROWING filter, honoured only for "
        "system.administer; every other caller is pinned by "
        "resolve_admin_scope to their own org and the parameter is ignored. "
        "It can never widen what a scoped caller sees."
    ),
    (
        "career_paths/routers/authoring.py",
        "enroll_student_in_path",
    ): (
        "Disabled stub: raises 409 unconditionally and never touches the "
        "database. Direct path enrolment moved to Learning Programs, so "
        "there is no tenant row to reach."
    ),
    (
        "career_paths/routers/authoring.py",
        "unenroll_student_from_path",
    ): (
        "Disabled stub: raises 409 unconditionally and never touches the "
        "database. Withdrawal moved to the Learning Program enrolment."
    ),
    (
        "career_paths/routers/learner.py",
        "start_course_in_path",
    ): (
        "Student-scoped, not permission-scoped: student_id comes from the "
        "JWT and the service 403s unless the course sits in a stage of a "
        "path the caller is ALREADY actively enrolled in. A student cannot "
        "be enrolled in another tenant's path, so the cross-tenant reach "
        "this check guards against is unreachable here."
    ),
    (
        "admin/routers/security.py",
        "get_security_summary",
    ): (
        "Same narrowing-only organization_id as get_dashboard: honoured for "
        "system.administer, ignored for everyone else."
    ),
}


def _iter_router_modules() -> list[Path]:
    return sorted(
        p
        for p in _FEATURES.rglob("*.py")
        if ("/routers/" in p.as_posix() or "/api/" in p.as_posix())
        and p.name != "__init__.py"
        and "__pycache__" not in p.parts
    )


def _has_org_check(node: ast.AsyncFunctionDef | ast.FunctionDef) -> bool:
    for sub in ast.walk(node):
        if not isinstance(sub, ast.Call):
            continue
        func = sub.func
        name = (
            func.attr
            if isinstance(func, ast.Attribute)
            else func.id
            if isinstance(func, ast.Name)
            else None
        )
        if name in _ORG_CHECK_CALLS:
            return True
    return False


def _is_route(node: ast.AsyncFunctionDef | ast.FunctionDef) -> bool:
    """True when the function carries a ``@router.<verb>(...)`` decorator."""
    for dec in node.decorator_list:
        call = dec.func if isinstance(dec, ast.Call) else dec
        if isinstance(call, ast.Attribute) and call.attr in {
            "get",
            "post",
            "put",
            "patch",
            "delete",
        }:
            return True
    return False


def test_org_scoped_routes_check_the_organization() -> None:
    violations: list[str] = []

    for module in _iter_router_modules():
        rel = module.relative_to(_BACKEND_ROOT / "abridgeai" / "features").as_posix()
        tree = ast.parse(module.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.AsyncFunctionDef | ast.FunctionDef):
                continue
            if not _is_route(node):
                continue
            params = {a.arg for a in node.args.args} | {a.arg for a in node.args.kwonlyargs}
            scoped = params & _ORG_SCOPED_PARAMS
            if not scoped:
                continue
            if _GLOBAL_BY_DESIGN.get((rel, node.name)):
                continue
            if _has_org_check(node):
                continue
            violations.append(f"{rel}::{node.name} (takes {sorted(scoped)})")

    assert violations == [], (
        "Routes addressing an org-owned resource without an organization check:\n  "
        + "\n  ".join(violations)
        + "\n\nThe permission dependency alone is NOT sufficient: the flat permission "
        "set ignores scope_kind, so a grant made inside one organization satisfies it "
        "everywhere. Resolve the resource's organization_id and call "
        "`require_org_access(...)`, or add the endpoint to _GLOBAL_BY_DESIGN with a "
        "reason if it genuinely acts across tenants."
    )


def test_global_by_design_entries_still_exist() -> None:
    """An allowlist that outlives its endpoint silently widens the check."""
    stale: list[str] = []
    for (rel, handler), _reason in _GLOBAL_BY_DESIGN.items():
        module = _FEATURES / rel
        if not module.exists():
            stale.append(f"{rel} (module gone)")
            continue
        tree = ast.parse(module.read_text(encoding="utf-8"))
        names = {
            n.name
            for n in ast.walk(tree)
            if isinstance(n, ast.AsyncFunctionDef | ast.FunctionDef)
        }
        if handler not in names:
            stale.append(f"{rel}::{handler} (handler gone)")
    assert stale == [], f"Stale _GLOBAL_BY_DESIGN entries: {stale}"


def test_every_org_check_resolves_permissions_not_just_membership() -> None:
    """``require_org_access`` must be called WITH the route's permission codes.

    Membership alone is the weaker question. It cannot separate a student of
    org A from a manager of org B who also studies at A — and the flat set
    behind the route dependency has already accepted that manager's code. The
    ``permissions=`` fallback exists so an unconverted call site degrades to
    the old behaviour instead of to nothing; this test stops one from being
    added by accident.
    """
    unscoped: list[str] = []

    for module in _iter_router_modules():
        rel = module.relative_to(_BACKEND_ROOT / "abridgeai" / "features").as_posix()
        tree = ast.parse(module.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            if name != "require_org_access":
                continue
            if not any(kw.arg == "permissions" for kw in node.keywords):
                unscoped.append(f"{rel}:{node.lineno}")

    assert unscoped == [], (
        "require_org_access called without `permissions=` at:\n  "
        + "\n  ".join(unscoped)
        + "\n\nPass the same codes the route's dependency declares, so the check "
        "resolves them against the owning organization instead of falling back "
        "to bare membership."
    )
