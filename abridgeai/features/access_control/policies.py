"""FastAPI permission policies (FIX-CRIT-1).

Permission-enforcement dependency factories that compose with
:func:`get_current_user` from ``core.security`` and the canonical
scope-aware queries from ``features.access_control.queries`` (T1.4a).

Five public functions:

* :func:`require_permission` -- global / non-scoped permission check.
* :func:`require_any_permission` -- passes if the principal holds any of
  the listed codes.
* :func:`require_course_permission` -- course-scoped check that resolves
  all four ``scope_kind`` values via :func:`load_course_permissions`.
  Course owner short-circuits the permission lookup (Reconciliation
  §A1 -- ownership is an additional allow on top of explicit grants).
* :func:`require_org_unit_permission` -- org_unit-scoped check that
  walks the unit's ancestor chain in Python.
* :func:`require_org_access` -- organization-scoped check for org-owned
  resources that are NOT course-owned, where there is no course to
  re-resolve scope against. Resolves the caller's permissions FOR that
  organization via :func:`load_org_permissions`. Mandatory for those.
* :func:`can_manage_course` -- bool helper for non-route call sites
  (cron jobs, bulk operations) -- never raises.

**FIX-CRIT-1 invariant**: this module never short-circuits permission
enforcement based on environment, feature flag, or build mode. Every
call either resolves a real permission set against the catalog or
short-circuits on the documented ownership rule. The companion test
``test_no_devmode_permission_flag`` enforces absence of the legacy
footgun by grepping the source tree.

**Import-linter posture**: this module lives in
``features.access_control.policies`` -- not under ``services/`` --
which means the contract ``services -> sqlalchemy`` (T0.4) does not
apply here. Importing ``sqlalchemy.text`` for the inline ``courses``
SELECT is therefore allowed. ``features.courses`` does not exist yet
(Phase 3 T3.1+); the inline SELECT bridges that gap and is documented
to move to ``features/courses/queries`` once the feature lands.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from typing import Annotated
from uuid import UUID

from fastapi import Depends, HTTPException, Request, status
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.core.db import get_db
from abridgeai.core.security import CurrentUser, get_current_user
from abridgeai.features.access_control.queries import (
    load_course_permissions,
    load_org_permissions,
    load_user_permissions,
)

PermissionDependency = Callable[..., Awaitable[CurrentUser]]


_COURSE_OWNER_CTX_SQL = text(
    """
    SELECT owner_user_id, organization_id, faculty_id
    FROM courses
    WHERE id = :course_id
      AND deleted_at IS NULL
    """
)


_ORG_UNIT_ANCESTORS_SQL = text(
    """
    WITH RECURSIVE org_unit_tree AS (
        SELECT id AS unit_id, parent_unit_id, organization_id
        FROM org_units
        WHERE id = :org_unit_id
          AND deleted_at IS NULL

        UNION

        SELECT ou.id, ou.parent_unit_id, ou.organization_id
        FROM org_units ou
        JOIN org_unit_tree t ON ou.id = t.parent_unit_id
        WHERE ou.deleted_at IS NULL
    )
    SELECT unit_id, organization_id FROM org_unit_tree
    """
)


_ORG_MEMBERSHIP_SQL = text(
    """
    SELECT 1
    FROM organization_memberships
    WHERE user_id = :user_id
      AND organization_id = :organization_id
      AND status = 'active'
      AND deleted_at IS NULL
    LIMIT 1
    """
)


async def user_is_org_member(
    db: AsyncSession,
    *,
    user_id: UUID,
    organization_id: UUID,
) -> bool:
    """True iff the user holds an active, non-deleted membership in the org."""
    row = (
        await db.execute(
            _ORG_MEMBERSHIP_SQL,
            {"user_id": str(user_id), "organization_id": str(organization_id)},
        )
    ).first()
    return row is not None


async def require_org_access(
    db: AsyncSession,
    current_user: CurrentUser,
    organization_id: UUID,
    *,
    resource: str,
    resource_id: UUID,
    permissions: Sequence[str] = (),
) -> None:
    """Raise 404 unless the caller may act on ``organization_id``.

    The mandatory second half of authorising an **org-owned resource that is
    not course-owned**. The permission dependencies above answer only "does
    this principal hold code X anywhere", because :func:`load_user_permissions`
    flattens role assignments without regard to ``scope_kind``. Course-owned
    resources are fine — they route through :func:`require_course_permission`,
    which re-resolves scope against the course's own organization. Anything
    org-owned but course-less has nothing to re-resolve against, so it must
    call this after loading the resource.

    Pass ``permissions`` — the same codes the route's dependency checks — and
    the question becomes *"was one of these granted FOR this organization?"*,
    resolved through :func:`load_org_permissions`. That is the correct check
    and what every call site should use.

    Omit them and it falls back to bare organization membership, which is
    necessary but NOT sufficient. Membership cannot separate a student of org A
    from a manager of org B who also happens to study at A — and the flat set
    behind the dependency has already accepted that manager's ``course.update``.
    The fallback exists so an unconverted call site degrades to the previous
    behaviour rather than to nothing.

    ``system.administer`` passes either way — it is the platform-operator
    permission and is only ever granted globally.

    **404, not 403.** A 403 would confirm the resource exists, which turns any
    id endpoint into an existence oracle across tenants: enumerate ids, read
    the status code, learn which career paths or invitation codes another
    organization owns. The not-found shape matches what the caller would see
    for a genuinely absent id, so the two are indistinguishable.

    Call it *inside* the handler's ``try`` block where one exists, so a missing
    resource still maps to the router's own 404 shape.
    """
    if current_user.has_permission("system.administer"):
        return

    if permissions:
        granted = await load_org_permissions(db, current_user.user_id, organization_id)
        if any(code in granted for code in permissions):
            return
        raise _not_found(resource, resource_id)

    if await user_is_org_member(db, user_id=current_user.user_id, organization_id=organization_id):
        return
    raise _not_found(resource, resource_id)


def _permission_denied(
    *,
    required: tuple[str, ...],
    scope: str,
    **extras: str,
) -> HTTPException:
    detail: dict[str, object] = {
        "error": "permission_denied",
        "required": list(required),
        "scope": scope,
    }
    detail.update(extras)
    return HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=detail)


def _not_found(resource: str, resource_id: UUID) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail={"error": "not_found", "resource": resource, "id": str(resource_id)},
    )


def require_permission(perm_code: str) -> PermissionDependency:
    """Build a FastAPI dependency that enforces a single global permission code.

    Resolves the principal via :func:`get_current_user`, then loads the
    effective global permission set on demand and raises HTTP 403 if the code
    is missing.
    """

    async def dependency(
        current_user: Annotated[CurrentUser, Depends(get_current_user)],
        db: Annotated[AsyncSession, Depends(get_db)],
    ) -> CurrentUser:
        perms = await load_user_permissions(db, current_user.user_id)
        if perm_code not in perms:
            raise _permission_denied(required=(perm_code,), scope="global")
        return current_user.with_permissions(frozenset(perms))

    return dependency


def require_any_permission(*perm_codes: str) -> PermissionDependency:
    """Build a FastAPI dependency that passes if any of ``perm_codes`` is held.

    Empty ``perm_codes`` is a programming error: raises ``ValueError`` at
    factory build time, never at request time.
    """
    if not perm_codes:
        raise ValueError("require_any_permission requires at least one permission code")

    codes = tuple(perm_codes)

    async def dependency(
        current_user: Annotated[CurrentUser, Depends(get_current_user)],
        db: Annotated[AsyncSession, Depends(get_db)],
    ) -> CurrentUser:
        perms = await load_user_permissions(db, current_user.user_id)
        if not any(c in perms for c in codes):
            raise _permission_denied(required=codes, scope="global")
        return current_user.with_permissions(frozenset(perms))

    return dependency


def _resolve_path_uuid(request: Request, param_name: str, resource: str) -> UUID:
    raw = request.path_params.get(param_name)
    if raw is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "policy_misconfigured",
                "missing_path_param": param_name,
                "resource": resource,
            },
        )
    try:
        return raw if isinstance(raw, UUID) else UUID(str(raw))
    except (ValueError, AttributeError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail={"error": "invalid_uuid", "param": param_name, "value": str(raw)},
        ) from exc


def require_course_permission(
    course_id_param: str,
    *perm_codes: str,
    allow_owner: bool = True,
) -> PermissionDependency:
    """Build a FastAPI dependency that enforces course-scoped permissions.

    ``course_id_param`` is the name of the path parameter carrying the course
    UUID (typically ``"course_id"``). At request time the dependency:

    1. Loads the course's owner / organization / org_unit context.
    2. Returns immediately if the principal IS the course owner AND
       ``allow_owner`` is set -- ownership is an additional allow on top of
       explicit permissions (Reconciliation §A1). This saves a DB roundtrip on
       the hot path (teacher editing own course).
    3. Otherwise calls :func:`load_course_permissions` to resolve all four
       ``scope_kind`` values against the course context, and asserts that the
       intersection with ``perm_codes`` is non-empty.

    ``allow_owner=False`` disables the ownership short-circuit, so ONLY an
    explicit permission grant passes. Use it for operations that must stay with
    a role even on a course the caller owns -- e.g. learning-outcome authoring
    is manager-owned, so a teacher who owns the course still must NOT edit its
    LOs (they hold ``course.update`` for content but not ``learning_outcome
    .manage``). Without this flag the owner short-circuit would let the owning
    teacher bypass the LO gate entirely.

    Missing course -> HTTP 404. Missing permission -> HTTP 403 with the
    required codes and course id in the detail body.
    """
    if not perm_codes:
        raise ValueError("require_course_permission requires at least one permission code")

    codes = tuple(perm_codes)
    param_name = course_id_param

    async def dependency(
        request: Request,
        current_user: Annotated[CurrentUser, Depends(get_current_user)],
        db: Annotated[AsyncSession, Depends(get_db)],
    ) -> CurrentUser:
        course_id = _resolve_path_uuid(request, param_name, "course")
        result = await db.execute(_COURSE_OWNER_CTX_SQL, {"course_id": course_id})
        row = result.one_or_none()
        if row is None:
            raise _not_found("course", course_id)

        owner_user_id, _organization_id, _org_unit_id = row
        if allow_owner and owner_user_id == current_user.user_id:
            return current_user

        course_perms = await load_course_permissions(db, current_user.user_id, course_id)
        if not course_perms.intersection(codes):
            raise _permission_denied(
                required=codes,
                scope="course",
                course_id=str(course_id),
            )
        return current_user

    return dependency


def require_org_unit_permission(
    org_unit_id_param: str,
    *perm_codes: str,
) -> PermissionDependency:
    """Build a FastAPI dependency that enforces org_unit-scoped permissions.

    Org_units do not have an owner column, so there is no ownership short-
    circuit. The check loads the principal's effective permission set
    globally (catching ``scope_kind='global'`` and ``'organization'`` for the
    relevant org) and additionally walks the unit's ancestor chain to honour
    HOD-style assignments (``scope_kind='org_unit'``).

    Implementation note: this composes ``load_user_permissions`` (which
    already handles role assignments + direct grants + active window for the
    global view) with a Python-side scope filter. A dedicated SQL helper
    similar to :func:`load_course_permissions` would be cleaner; deferred
    until a feature actually needs the optimisation.
    """
    if not perm_codes:
        raise ValueError("require_org_unit_permission requires at least one permission code")

    codes = tuple(perm_codes)
    param_name = org_unit_id_param

    async def dependency(
        request: Request,
        current_user: Annotated[CurrentUser, Depends(get_current_user)],
        db: Annotated[AsyncSession, Depends(get_db)],
    ) -> CurrentUser:
        org_unit_id = _resolve_path_uuid(request, param_name, "org_unit")
        ancestor_rows = await db.execute(_ORG_UNIT_ANCESTORS_SQL, {"org_unit_id": org_unit_id})
        ancestors = ancestor_rows.all()
        if not ancestors:
            raise _not_found("org_unit", org_unit_id)

        ancestor_ids = {row[0] for row in ancestors}
        organization_id = ancestors[0][1]

        scoped_perms = await _load_org_unit_scoped_permissions(
            db,
            user_id=current_user.user_id,
            organization_id=organization_id,
            ancestor_ids=ancestor_ids,
        )
        if not scoped_perms.intersection(codes):
            raise _permission_denied(
                required=codes,
                scope="org_unit",
                org_unit_id=str(org_unit_id),
            )
        return current_user

    return dependency


_LOAD_ORG_UNIT_SCOPED_SQL = text(
    """
    SELECT DISTINCT p.code
    FROM permissions p
    JOIN role_permissions rp ON rp.permission_id = p.id
    JOIN user_role_assignments ura ON ura.role_id = rp.role_id
    WHERE ura.user_id = :user_id
      AND ura.deleted_at IS NULL
      AND ura.active_from <= NOW()
      AND (ura.active_until IS NULL OR ura.active_until > NOW())
      AND p.deleted_at IS NULL
      AND (
          ura.scope_kind = 'global'
          OR (ura.scope_kind = 'organization' AND ura.organization_id = :organization_id)
          OR (
              ura.scope_kind = 'org_unit'
              AND ura.org_unit_id = ANY(:ancestor_ids)
              AND EXISTS (
                  SELECT 1 FROM user_faculty_assignments ufa
                  WHERE ufa.user_id = ura.user_id
                    AND ufa.faculty_id = ura.org_unit_id
                    AND ufa.status = 'active'
                    AND ufa.deleted_at IS NULL
                    AND (ufa.active_until IS NULL OR ufa.active_until > :at)
              )
          )
      )

    UNION

    SELECT DISTINCT p.code
    FROM permissions p
    JOIN user_permission_grants upg ON upg.permission_id = p.id
    WHERE upg.user_id = :user_id
      AND upg.deleted_at IS NULL
      AND (upg.expires_at IS NULL OR upg.expires_at > NOW())
      AND p.deleted_at IS NULL
      AND (
          upg.scope_kind = 'global'
          OR (upg.scope_kind = 'organization' AND upg.organization_id = :organization_id)
          OR (upg.scope_kind = 'org_unit' AND upg.org_unit_id = ANY(:ancestor_ids))
      )
    """
)


async def _load_org_unit_scoped_permissions(
    db: AsyncSession,
    *,
    user_id: UUID,
    organization_id: UUID,
    ancestor_ids: set[UUID],
) -> set[str]:
    result = await db.execute(
        _LOAD_ORG_UNIT_SCOPED_SQL,
        {
            "user_id": user_id,
            "organization_id": organization_id,
            "ancestor_ids": list(ancestor_ids),
        },
    )
    return {row[0] for row in result.all()}


async def can_manage_course(
    db: AsyncSession,
    current_user_id: UUID,
    course_id: UUID,
    *,
    manage_perm: str = "course.update",
) -> bool:
    """Return ``True`` iff the user can manage ``course_id``.

    Mirrors :func:`require_course_permission` semantics (owner short-circuit
    + scope-aware permission resolution) but returns a ``bool`` instead of
    raising. Use from non-route contexts (cron jobs, bulk operations,
    internal queues) where HTTP exceptions are inappropriate.

    Returns ``False`` if the course does not exist (vs raising 404).
    """
    result = await db.execute(_COURSE_OWNER_CTX_SQL, {"course_id": course_id})
    row = result.one_or_none()
    if row is None:
        return False

    owner_user_id, _organization_id, _org_unit_id = row
    if owner_user_id == current_user_id:
        return True

    course_perms = await load_course_permissions(db, current_user_id, course_id)
    return manage_perm in course_perms


__all__ = [
    "PermissionDependency",
    "can_manage_course",
    "require_any_permission",
    "require_course_permission",
    "require_org_access",
    "require_org_unit_permission",
    "require_permission",
    "user_is_org_member",
]
