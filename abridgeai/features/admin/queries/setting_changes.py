"""Data access for ``system_setting_changes`` — the runtime-config audit trail.

Append and read only. There is deliberately no update or delete path: undoing a
change is another appended row (``action='rollback'``), because the record of
an incident is worth more than a tidy history.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

_INSERT_SQL = text(
    """
    INSERT INTO system_setting_changes (
        setting_key, organization_id, scope, action,
        before_value_json, after_value_json,
        reason, actor_id, source, reverted_change_id
    )
    VALUES (
        :setting_key, CAST(:organization_id AS uuid), :scope, :action,
        CAST(:before_value AS jsonb), CAST(:after_value AS jsonb),
        :reason, :actor_id, :source, CAST(:reverted_change_id AS uuid)
    )
    RETURNING id, created_at
    """
)

# Newest first. ``:setting_key`` / ``:organization_id`` are optional filters;
# ``:scope_filter`` distinguishes "global rows only" from "no scope filter",
# which a NULL organization_id alone cannot express.
_LIST_SQL = text(
    """
    SELECT c.id, c.setting_key, c.organization_id, c.scope, c.action,
           c.before_value_json, c.after_value_json, c.reason,
           c.actor_id, c.source, c.reverted_change_id, c.created_at,
           u.primary_email AS actor_email,
           o.name          AS organization_name
    FROM system_setting_changes c
    LEFT JOIN users u ON u.id = c.actor_id
    LEFT JOIN organizations o ON o.id = c.organization_id
    WHERE (CAST(:setting_key AS text) IS NULL OR c.setting_key = CAST(:setting_key AS text))
      AND (
            CAST(:scope_filter AS text) IS NULL
            OR (CAST(:scope_filter AS text) = 'global' AND c.scope = 'global')
            OR (
                 CAST(:scope_filter AS text) = 'organization'
                 AND c.organization_id = CAST(:organization_id AS uuid)
               )
          )
    ORDER BY c.created_at DESC, c.id DESC
    LIMIT :limit
    """
)

_GET_SQL = text(
    """
    SELECT id, setting_key, organization_id, scope, action,
           before_value_json, after_value_json, reason,
           actor_id, source, reverted_change_id, created_at
    FROM system_setting_changes
    WHERE id = CAST(:change_id AS uuid)
    """
)

# Organizations that would be affected by a global change to this key: every
# live organization WITHOUT its own override, since an org row wins over the
# global one. The count is the honest answer to "who does this touch".
_AFFECTED_ORGS_SQL = text(
    """
    SELECT COUNT(*) AS affected,
           (SELECT COUNT(*) FROM organizations WHERE deleted_at IS NULL) AS total
    FROM organizations o
    WHERE o.deleted_at IS NULL
      AND NOT EXISTS (
          SELECT 1
          FROM system_settings s
          WHERE s.organization_id = o.id
            AND s.setting_key = :setting_key
      )
    """
)


async def insert(
    db: AsyncSession,
    *,
    setting_key: str,
    organization_id: UUID | None,
    scope: str,
    action: str,
    before_value: str | None,
    after_value: str | None,
    reason: str,
    actor_id: UUID | None,
    source: str,
    reverted_change_id: UUID | None = None,
) -> dict[str, Any]:
    row = (
        await db.execute(
            _INSERT_SQL,
            {
                "setting_key": setting_key,
                "organization_id": str(organization_id) if organization_id else None,
                "scope": scope,
                "action": action,
                "before_value": before_value,
                "after_value": after_value,
                "reason": reason,
                "actor_id": actor_id,
                "source": source,
                "reverted_change_id": (
                    str(reverted_change_id) if reverted_change_id else None
                ),
            },
        )
    ).mappings().one()
    return dict(row)


async def list_changes(
    db: AsyncSession,
    *,
    setting_key: str | None,
    organization_id: UUID | None,
    scope_filter: str | None,
    limit: int,
) -> list[dict[str, Any]]:
    rows = (
        await db.execute(
            _LIST_SQL,
            {
                "setting_key": setting_key,
                "organization_id": str(organization_id) if organization_id else None,
                "scope_filter": scope_filter,
                "limit": limit,
            },
        )
    ).mappings()
    return [dict(row) for row in rows]


async def get_change(db: AsyncSession, *, change_id: UUID) -> dict[str, Any] | None:
    row = (
        await db.execute(_GET_SQL, {"change_id": str(change_id)})
    ).mappings().one_or_none()
    return dict(row) if row else None


async def affected_org_counts(db: AsyncSession, *, setting_key: str) -> tuple[int, int]:
    """``(organizations inheriting this key, total live organizations)``."""
    row = (
        await db.execute(_AFFECTED_ORGS_SQL, {"setting_key": setting_key})
    ).mappings().one()
    return int(row["affected"] or 0), int(row["total"] or 0)


__all__ = [
    "affected_org_counts",
    "get_change",
    "insert",
    "list_changes",
]
