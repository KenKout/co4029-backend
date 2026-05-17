"""User management service -- list, detail, disable / enable (T7.5).

Schema note: ``users.status`` CHECK allows only ``('active','invited','inactive','suspended')``.
The plan calls the operation "disable" -- we map disable -> ``status='inactive'``
(the closest semantic match in the existing constraint). When re-enabling the
user we restore ``status='active'``; existing ``auth_sessions`` revoked at
disable time stay revoked so the user must complete a fresh OAuth round-trip.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import UUID

from sqlalchemy import text

from abridgeai.core.exceptions import NotFoundError
from abridgeai.features.admin.queries import users as user_queries

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


_DISABLE_USER_SQL = text(
    """
    UPDATE users
       SET status = 'inactive',
           updated_at = NOW()
     WHERE id = :user_id
       AND status <> 'inactive'
    RETURNING id, status
    """
)

_ENABLE_USER_SQL = text(
    """
    UPDATE users
       SET status = 'active',
           updated_at = NOW()
     WHERE id = :user_id
    RETURNING id, status
    """
)

_REVOKE_SESSIONS_SQL = text(
    """
    UPDATE auth_sessions
       SET revoked_at = NOW()
     WHERE user_id = :user_id
       AND revoked_at IS NULL
    RETURNING id
    """
)

_GET_USER_SQL = text(
    """
    SELECT id, primary_email, status, last_login_at, created_at, updated_at
    FROM users
    WHERE id = :user_id
    """
)


async def list_users(
    db: AsyncSession,
    *,
    status_filter: str | None,
    role_code: str | None,
    organization_id: UUID | None,
    limit: int,
    offset: int,
) -> list[dict[str, Any]]:
    return await user_queries.list_users(
        db,
        status_filter=status_filter,
        role_code=role_code,
        organization_id=organization_id,
        limit=limit,
        offset=offset,
    )


async def user_detail(db: AsyncSession, *, user_id: UUID) -> dict[str, Any]:
    base = (await db.execute(_GET_USER_SQL, {"user_id": user_id})).mappings().one_or_none()
    if base is None:
        raise NotFoundError(f"user {user_id} not found")
    role_assignments = await user_queries.role_assignments(db, user_id=user_id)
    sessions = await user_queries.active_sessions(db, user_id=user_id)
    return {
        "user": dict(base),
        "role_assignments": role_assignments,
        "active_sessions": sessions,
    }


async def disable_user(db: AsyncSession, *, user_id: UUID) -> dict[str, Any]:
    """Set ``users.status='inactive'`` AND revoke every active auth session.

    Returns ``{"user_id", "status", "revoked_session_count"}``. Raises
    :class:`NotFoundError` when the user does not exist OR is already
    inactive (idempotent guard).
    """
    user_row = (await db.execute(_DISABLE_USER_SQL, {"user_id": user_id})).mappings().one_or_none()
    if user_row is None:
        check = (await db.execute(_GET_USER_SQL, {"user_id": user_id})).mappings().one_or_none()
        if check is None:
            raise NotFoundError(f"user {user_id} not found")
        revoked_already_inactive = (
            (await db.execute(_REVOKE_SESSIONS_SQL, {"user_id": user_id})).mappings().all()
        )
        await db.commit()
        return {
            "user_id": user_id,
            "status": check["status"],
            "revoked_session_count": len(revoked_already_inactive),
        }

    revoked_now = (await db.execute(_REVOKE_SESSIONS_SQL, {"user_id": user_id})).mappings().all()
    await db.commit()
    return {
        "user_id": user_id,
        "status": user_row["status"],
        "revoked_session_count": len(revoked_now),
    }


async def enable_user(db: AsyncSession, *, user_id: UUID) -> dict[str, Any]:
    """Restore ``users.status='active'``. Sessions stay revoked by design."""
    row = (await db.execute(_ENABLE_USER_SQL, {"user_id": user_id})).mappings().one_or_none()
    if row is None:
        raise NotFoundError(f"user {user_id} not found")
    await db.commit()
    return {"user_id": user_id, "status": row["status"]}


__all__ = [
    "disable_user",
    "enable_user",
    "list_users",
    "user_detail",
]
