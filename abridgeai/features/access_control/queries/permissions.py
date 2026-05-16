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
        SELECT id AS course_id, organization_id, org_unit_id
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


def _now_at(at: datetime | None) -> datetime:
    if at is not None:
        return at
    return datetime.now(UTC)


async def load_user_permissions(
    db: AsyncSession, user_id: UUID, *, at: datetime | None = None
) -> set[str]:
    """Effective permission codes the user holds at ``at`` (defaults to now)."""
    result = await db.execute(_LOAD_USER_PERMISSIONS_SQL, {"user_id": user_id, "at": _now_at(at)})
    return {row[0] for row in result.all()}


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
    result = await db.execute(
        _LOAD_COURSE_PERMISSIONS_SQL,
        {"user_id": user_id, "course_id": course_id, "at": _now_at(at)},
    )
    return {row[0] for row in result.all()}


__all__ = ["load_course_permissions", "load_user_permissions"]
