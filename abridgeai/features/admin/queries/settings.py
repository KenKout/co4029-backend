"""Data access for ``system_settings``.

Rows are keyed on ``(organization_id, setting_key)`` with
``organization_id IS NULL`` meaning the deployment-wide default. Two partial
unique indexes enforce that (see migration ``0066_org_settings``) — a plain
composite UNIQUE would not, because Postgres treats NULLs as distinct and
would happily accept two global rows for one key.

The upsert therefore cannot use ``ON CONFLICT (organization_id, setting_key)``:
that constraint does not exist. It targets the matching partial index by
repeating its predicate.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

_SELECT_SQL = text(
    """
    SELECT setting_key, setting_value_json, organization_id, updated_by, updated_at
    FROM system_settings
    WHERE organization_id IS NULL
       OR organization_id = CAST(:organization_id AS uuid)
    ORDER BY setting_key
    """
)

_UPSERT_GLOBAL_SQL = text(
    """
    INSERT INTO system_settings (id, setting_key, setting_value_json, updated_by)
    VALUES (gen_random_uuid(), :setting_key, CAST(:value AS jsonb), :updated_by)
    ON CONFLICT (setting_key) WHERE organization_id IS NULL
    DO UPDATE SET setting_value_json = EXCLUDED.setting_value_json,
                  updated_by = EXCLUDED.updated_by,
                  updated_at = NOW()
    """
)

_UPSERT_ORG_SQL = text(
    """
    INSERT INTO system_settings
        (id, organization_id, setting_key, setting_value_json, updated_by)
    VALUES (gen_random_uuid(), CAST(:organization_id AS uuid), :setting_key,
            CAST(:value AS jsonb), :updated_by)
    ON CONFLICT (organization_id, setting_key) WHERE organization_id IS NOT NULL
    DO UPDATE SET setting_value_json = EXCLUDED.setting_value_json,
                  updated_by = EXCLUDED.updated_by,
                  updated_at = NOW()
    """
)

_DELETE_GLOBAL_SQL = text(
    "DELETE FROM system_settings WHERE setting_key = :setting_key AND organization_id IS NULL"
)

_DELETE_ORG_SQL = text(
    """
    DELETE FROM system_settings
    WHERE setting_key = :setting_key
      AND organization_id = CAST(:organization_id AS uuid)
    """
)


async def load_rows(
    db: AsyncSession, organization_id: UUID | None
) -> list[dict[str, Any]]:
    """Global rows plus this organization's rows, in one round trip."""
    result = await db.execute(
        _SELECT_SQL,
        {"organization_id": str(organization_id) if organization_id else None},
    )
    return [dict(row) for row in result.mappings()]


async def upsert(
    db: AsyncSession,
    *,
    setting_key: str,
    value_json: str,
    organization_id: UUID | None,
    updated_by: UUID | None,
) -> None:
    if organization_id is None:
        await db.execute(
            _UPSERT_GLOBAL_SQL,
            {
                "setting_key": setting_key,
                "value": value_json,
                "updated_by": updated_by,
            },
        )
        return
    await db.execute(
        _UPSERT_ORG_SQL,
        {
            "organization_id": str(organization_id),
            "setting_key": setting_key,
            "value": value_json,
            "updated_by": updated_by,
        },
    )


async def delete(
    db: AsyncSession, *, setting_key: str, organization_id: UUID | None
) -> int:
    """Remove one row. Returns the number deleted (0 when none was set)."""
    if organization_id is None:
        result = await db.execute(_DELETE_GLOBAL_SQL, {"setting_key": setting_key})
    else:
        result = await db.execute(
            _DELETE_ORG_SQL,
            {"setting_key": setting_key, "organization_id": str(organization_id)},
        )
    return int(result.rowcount or 0)


__all__ = ["delete", "load_rows", "upsert"]
