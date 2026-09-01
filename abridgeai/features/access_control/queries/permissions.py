"""Canonical scope-aware permission queries (FIX-CRIT-2).

Two functions produce the effective permission set for a user:

* :func:`load_user_permissions` — global view (any context).
* :func:`load_course_permissions` — restricted to a single course,
  resolving all four ``scope_kind`` values (global / organization /
  org_unit / course) against the course's organization and org_unit
  ancestor chain.

The legacy ``backend/app/queries/sql/permissions/effective_permissions.sql``
was incomplete: it joined only role-based assignments, ignored direct
``user_permission_grants``, and missed the ``active_from`` window. The
legacy ``_load_user_permissions`` function in ``backend/app/core/permissions.py``
was closer to correct but its sibling ``load_course_permissions`` only
covered ``scope_kind`` IN (``'global'``, ``'course'``) — silently dropping
HOD assignments at ``scope_kind='org_unit'`` and manager assignments at
``scope_kind='organization'`` (FIX-CRIT-2).

This module is the single canonical replacement: role assignments +
direct grants, active-window filtered, all 4 scope kinds resolved in one
query. Per Reconciliation §A1 / §A9 and plan §A9 the divergent
``effective_permissions.sql`` is deliberately NOT ported (T1.4b enforces).
"""

from __future__ import annotations

from datetime import UTC, datetime
from importlib import resources
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.core.ttl_cache import TTLCache

# Effective-permission resolution runs on EVERY authenticated request
# (``policies.py`` dependency chain): three joins over permissions /
# role_permissions / user_role_assignments plus grants, and the course/org
# variants additionally walk a recursive org-unit tree. The result only
# changes when an assignment / grant / role mapping is written, which is a
# rare admin action — so a short-TTL process-local cache removes the query
# from the hot path while bounding staleness to that window. Write paths in
# ``services/admin.py`` call :func:`invalidate_user_permissions` on every
# mutation, so a role change takes effect immediately even before expiry.
#
# ``at`` (explicit evaluation timestamp) always BYPASSES the cache: tests and
# time-travel callers pin determinism there, and caching would collapse two
# different ``at`` values into one answer.
_PERMISSIONS_TTL_SECONDS = 30.0
_USER_PERMS_CACHE = TTLCache(max_entries=2048, ttl_seconds=_PERMISSIONS_TTL_SECONDS)
_COURSE_PERMS_CACHE = TTLCache(max_entries=4096, ttl_seconds=_PERMISSIONS_TTL_SECONDS)
_ORG_PERMS_CACHE = TTLCache(max_entries=2048, ttl_seconds=_PERMISSIONS_TTL_SECONDS)


def invalidate_user_permissions(user_id: UUID) -> None:
    """Drop every cached permission set derived from ``user_id``'s roles/grants.

    Called by the access-control write paths (role assignment create/revoke,
    grant create/revoke). Course- and org-scoped caches use composite keys
    ``(user_id, scope_id)``, so a predicate sweep drops every entry for that
    user across all three caches.
    """
    _USER_PERMS_CACHE.invalidate(user_id)
    _COURSE_PERMS_CACHE.invalidate_where(lambda k: k[0] == user_id)
    _ORG_PERMS_CACHE.invalidate_where(lambda k: k[0] == user_id)


def clear_permissions_cache() -> None:
    """Drop everything. Test-support only."""
    _USER_PERMS_CACHE.clear()
    _COURSE_PERMS_CACHE.clear()
    _ORG_PERMS_CACHE.clear()


_ORG_UNIT_TREE_SQL = (
    resources.files("abridgeai.features.access_control.queries.sql")
    .joinpath("org_unit_tree.sql")
    .read_text(encoding="utf-8")
    .strip()
)


_LOAD_USER_PERMISSIONS_SQL = text(
    """
    SELECT DISTINCT p.code
    FROM permissions p
    JOIN role_permissions rp ON rp.permission_id = p.id
    JOIN user_role_assignments ura ON ura.role_id = rp.role_id
    WHERE ura.user_id = :user_id
      AND ura.deleted_at IS NULL
      AND ura.active_from <= :at
      AND (ura.active_until IS NULL OR ura.active_until > :at)
      AND p.deleted_at IS NULL

    UNION

    SELECT DISTINCT p.code
    FROM permissions p
    JOIN user_permission_grants upg ON upg.permission_id = p.id
    WHERE upg.user_id = :user_id
      AND upg.deleted_at IS NULL
      AND (upg.expires_at IS NULL OR upg.expires_at > :at)
      AND p.deleted_at IS NULL
    """
)


_LOAD_COURSE_PERMISSIONS_SQL = text(
    f"""
    WITH RECURSIVE course_ctx AS (
        SELECT id AS course_id, organization_id, faculty_id AS org_unit_id
        FROM courses
        WHERE id = :course_id
    ),
    {_ORG_UNIT_TREE_SQL}
    SELECT DISTINCT p.code
    FROM permissions p
    JOIN role_permissions rp ON rp.permission_id = p.id
    JOIN user_role_assignments ura ON ura.role_id = rp.role_id
    CROSS JOIN course_ctx cc
    WHERE ura.user_id = :user_id
      AND ura.deleted_at IS NULL
      AND ura.active_from <= :at
      AND (ura.active_until IS NULL OR ura.active_until > :at)
      AND p.deleted_at IS NULL
      AND (
          ura.scope_kind = 'global'
          OR (ura.scope_kind = 'course' AND ura.course_id = cc.course_id)
          OR (
              ura.scope_kind = 'org_unit'
              AND ura.org_unit_id IN (SELECT unit_id FROM org_unit_tree)
              AND EXISTS (
                  SELECT 1 FROM user_faculty_assignments ufa
                  WHERE ufa.user_id = ura.user_id
                    AND ufa.faculty_id = ura.org_unit_id
                    AND ufa.status = 'active'
                    AND ufa.deleted_at IS NULL
                    AND (ufa.active_until IS NULL OR ufa.active_until > :at)
              )
          )
          OR (
              ura.scope_kind = 'organization'
              AND ura.organization_id = cc.organization_id
          )
      )

    UNION

    SELECT DISTINCT p.code
    FROM permissions p
    JOIN user_permission_grants upg ON upg.permission_id = p.id
    CROSS JOIN course_ctx cc
    WHERE upg.user_id = :user_id
      AND upg.deleted_at IS NULL
      AND (upg.expires_at IS NULL OR upg.expires_at > :at)
      AND p.deleted_at IS NULL
      AND (
          upg.scope_kind = 'global'
          OR (upg.scope_kind = 'course' AND upg.course_id = cc.course_id)
          OR (
              upg.scope_kind = 'org_unit'
              AND upg.org_unit_id IN (SELECT unit_id FROM org_unit_tree)
          )
          OR (
              upg.scope_kind = 'organization'
              AND upg.organization_id = cc.organization_id
          )
      )
    """  # noqa: S608  -- interpolated value is a checked-in SQL fragment, not user input
)


_LOAD_ORG_PERMISSIONS_SQL = text(
    """
    SELECT DISTINCT p.code
    FROM permissions p
    JOIN role_permissions rp ON rp.permission_id = p.id
    JOIN user_role_assignments ura ON ura.role_id = rp.role_id
    WHERE ura.user_id = :user_id
      AND ura.deleted_at IS NULL
      AND ura.active_from <= :at
      AND (ura.active_until IS NULL OR ura.active_until > :at)
      AND p.deleted_at IS NULL
      AND (
          ura.scope_kind = 'global'
          OR (ura.scope_kind = 'organization' AND ura.organization_id = :organization_id)
          OR (
              ura.scope_kind = 'org_unit'
              AND ura.org_unit_id IN (
                  SELECT id FROM org_units
                  WHERE organization_id = :organization_id AND deleted_at IS NULL
              )
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
      AND (upg.expires_at IS NULL OR upg.expires_at > :at)
      AND p.deleted_at IS NULL
      AND (
          upg.scope_kind = 'global'
          OR (upg.scope_kind = 'organization' AND upg.organization_id = :organization_id)
          OR (
              upg.scope_kind = 'org_unit'
              AND upg.org_unit_id IN (
                  SELECT id FROM org_units
                  WHERE organization_id = :organization_id AND deleted_at IS NULL
              )
          )
      )
    """
)


def _now_at(at: datetime | None) -> datetime:
    if at is not None:
        return at
    return datetime.now(UTC)


async def load_user_permissions(
    db: AsyncSession, user_id: UUID, *, at: datetime | None = None
) -> set[str]:
    """Effective permission codes the user holds at ``at`` (defaults to now)."""
    if at is not None:
        # Explicit timestamp: never cached (determinism for tests/time-travel).
        result = await db.execute(_LOAD_USER_PERMISSIONS_SQL, {"user_id": user_id, "at": at})
        return {row[0] for row in result.all()}
    cached = _USER_PERMS_CACHE.get(user_id)
    if cached is not None:
        return set(cached)
    result = await db.execute(_LOAD_USER_PERMISSIONS_SQL, {"user_id": user_id, "at": _now_at(at)})
    perms = {row[0] for row in result.all()}
    _USER_PERMS_CACHE.put(user_id, frozenset(perms))
    return perms


async def load_course_permissions(
    db: AsyncSession,
    user_id: UUID,
    course_id: UUID,
    *,
    at: datetime | None = None,
) -> set[str]:
    """Effective permission codes the user holds for ``course_id``.

    Resolves all four ``scope_kind`` values (FIX-CRIT-2) against the
    course's ``organization_id`` and the recursive ancestor chain of its
    ``org_unit_id``. Includes role-based assignments and direct grants;
    both are filtered by the active window.
    """
    if at is not None:
        result = await db.execute(
            _LOAD_COURSE_PERMISSIONS_SQL,
            {"user_id": user_id, "course_id": course_id, "at": at},
        )
        return {row[0] for row in result.all()}
    cache_key = (user_id, course_id)
    cached = _COURSE_PERMS_CACHE.get(cache_key)
    if cached is not None:
        return set(cached)
    result = await db.execute(
        _LOAD_COURSE_PERMISSIONS_SQL,
        {"user_id": user_id, "course_id": course_id, "at": _now_at(at)},
    )
    perms = {row[0] for row in result.all()}
    _COURSE_PERMS_CACHE.put(cache_key, frozenset(perms))
    return perms


async def load_org_permissions(
    db: AsyncSession,
    user_id: UUID,
    organization_id: UUID,
    *,
    at: datetime | None = None,
) -> set[str]:
    """Permission codes the user holds **for** ``organization_id``.

    The organization-level counterpart of :func:`load_course_permissions`, and
    the answer that actually authorises a request against an org-owned
    resource. :func:`load_user_permissions` cannot: it flattens assignments
    without reading ``scope_kind``, so a role granted to a manager inside one
    tenant yields the same codes as a global grant, everywhere.

    Which scopes count, and why:

    * ``global`` — platform-wide by definition.
    * ``organization`` — granted for this org. The case the flat query loses.
    * ``org_unit`` — a unit belongs to exactly one organization, so a
      unit-scoped grant is authority *inside* this tenant. It is admitted here
      deliberately: the boundary being enforced is the tenant one, and
      narrowing a head-of-department's reach within their own organization is a
      separate question that :func:`require_org_unit_permission` already owns.
      Tightening it here would silently revoke access as a side effect of a
      tenancy fix.
    * ``course`` — deliberately absent. Authority over one course does not
      confer authority over the organization that owns it.
    """
    if at is not None:
        result = await db.execute(
            _LOAD_ORG_PERMISSIONS_SQL,
            {"user_id": user_id, "organization_id": organization_id, "at": at},
        )
        return {row[0] for row in result.all()}
    cache_key = (user_id, organization_id)
    cached = _ORG_PERMS_CACHE.get(cache_key)
    if cached is not None:
        return set(cached)
    result = await db.execute(
        _LOAD_ORG_PERMISSIONS_SQL,
        {"user_id": user_id, "organization_id": organization_id, "at": _now_at(at)},
    )
    perms = {row[0] for row in result.all()}
    _ORG_PERMS_CACHE.put(cache_key, frozenset(perms))
    return perms


__all__ = [
    "clear_permissions_cache",
    "invalidate_user_permissions",
    "load_course_permissions",
    "load_org_permissions",
    "load_user_permissions",
]
