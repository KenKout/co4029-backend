"""Explicit opt-in scope for deleting audit rows (FR-6.7).

Migration 0105 puts a ``BEFORE UPDATE OR DELETE`` trigger on every append-only
audit store (``http_audit_log``, ``system_setting_changes``). ``UPDATE`` is
rejected outright; ``DELETE`` is rejected too, *unless* the session-local GUC
``app.audit_maintenance`` reads ``'on'``. This module is the only supported way
to set it.

The split exists because "immutable" and "prunable" are both requirements. An
audit store the application can delete from is not immutable; one nobody can
delete from grows until it takes the database with it. Gating deletion behind a
scope no ordinary code path enters keeps both: request handlers, services and
workers cannot remove audit rows even by accident, while a retention job says
so explicitly and is visible in the diff when it does.

``SET LOCAL`` is deliberate -- the setting reverts when the surrounding
transaction ends, so the permission cannot leak into later work on a pooled
connection the way ``SET`` would. That also means the scope MUST be entered
inside an open transaction, and every delete it authorises must run in that
same transaction.

Example
-------
    async with engine.begin() as conn:
        await audit_maintenance(conn)
        await conn.execute(
            text("DELETE FROM http_audit_log WHERE created_at < :cutoff"),
            {"cutoff": cutoff},
        )
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import text

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncConnection, AsyncSession

# Literal, not a bound parameter: SET LOCAL takes a value token, not a
# placeholder, so postgres rejects ``SET LOCAL x = :v``. The value is a fixed
# constant in this module and never comes from a caller, so nothing is
# interpolated into the statement.
_ENABLE_SQL = text("SET LOCAL app.audit_maintenance = 'on'")


async def audit_maintenance(executor: AsyncConnection | AsyncSession) -> None:
    """Authorise audit-row deletion for the rest of the current transaction.

    Call inside an already-open transaction, immediately before the retention
    ``DELETE``. The grant ends with the transaction; there is nothing to undo
    and no way to leave it enabled on a returned pool connection.

    Grants deletion only. ``UPDATE`` on an append-only audit table stays
    impossible -- the trigger has no bypass for it at all.
    """
    await executor.execute(_ENABLE_SQL)


__all__ = ["audit_maintenance"]
