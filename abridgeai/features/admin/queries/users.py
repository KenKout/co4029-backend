"""User management queries for the admin dashboard (T7.5).

Only reads here; writes (status flip + session revocation) live in the service
layer because they need transactional grouping with ``db.commit()``.
"""

from __future__ import annotations

from datetime import datetime
from importlib import resources
from typing import Any
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.elements import TextClause

_SQL_DIR = resources.files("abridgeai.features.admin.queries.sql")


def _load(name: str) -> TextClause:
    return text(_SQL_DIR.joinpath(name).read_text(encoding="utf-8"))


_LIST_USERS_SQL = _load("users/list_users.sql")
_ROLE_ASSIGNMENTS_SQL = _load("users/role_assignments.sql")
_ACTIVE_SESSIONS_SQL = _load("users/active_sessions.sql")
_ROLE_HISTORY_SQL = _load("users/role_history.sql")


async def list_users(
    db: AsyncSession,
    *,
    status_filter: str | None,
    role_code: str | None,
    organization_id: UUID | None,
    q: str | None,
    limit: int,
    after_created_at: datetime | None,
    after_id: UUID | None,
) -> list[dict[str, Any]]:
    rows = (
        await db.execute(
            _LIST_USERS_SQL,
            {
                "status_filter": status_filter,
                "role_code": role_code,
                "organization_id": organization_id,
                "q": q,
                "limit": limit,
                "after_created_at": after_created_at,
                "after_id": after_id,
            },
        )
    ).mappings()
    return [dict(r) for r in rows]


async def role_assignments(db: AsyncSession, *, user_id: UUID) -> list[dict[str, Any]]:
    rows = (await db.execute(_ROLE_ASSIGNMENTS_SQL, {"user_id": user_id})).mappings()
    return [dict(r) for r in rows]


async def active_sessions(db: AsyncSession, *, user_id: UUID) -> list[dict[str, Any]]:
    rows = (await db.execute(_ACTIVE_SESSIONS_SQL, {"user_id": user_id})).mappings()
    return [dict(r) for r in rows]


async def role_history(db: AsyncSession, *, user_id: UUID) -> list[dict[str, Any]]:
    rows = (await db.execute(_ROLE_HISTORY_SQL, {"user_id": user_id})).mappings()
    return [dict(r) for r in rows]


__all__ = ["active_sessions", "list_users", "role_assignments", "role_history"]
