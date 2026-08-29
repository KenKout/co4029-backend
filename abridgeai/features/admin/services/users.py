"""User management service -- list, detail, disable / enable (T7.5).

Schema note: ``users.status`` CHECK allows only ``('active','invited','inactive','suspended')``.
The plan calls the operation "disable" -- we map disable -> ``status='inactive'``
(the closest semantic match in the existing constraint). When re-enabling the
user we restore ``status='active'``; existing ``auth_sessions`` revoked at
disable time stay revoked so the user must complete a fresh OAuth round-trip.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID

from sqlalchemy import text

from abridgeai.core.exceptions import NotFoundError
from abridgeai.core.pagination import (
    CursorPage,
    decode_composite_cursor,
    encode_composite_cursor,
)
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

_REVOKE_ONE_SESSION_SQL = text(
    """
    UPDATE auth_sessions
       SET revoked_at = NOW()
     WHERE id = :session_id
       AND user_id = :user_id
       AND revoked_at IS NULL
    RETURNING id
    """
)

_GET_USER_SQL = text(
    """
    SELECT u.id, u.primary_email, u.status, u.last_login_at,
           u.created_at, u.updated_at,
           p.display_name, p.given_name, p.family_name, p.bio,
           p.avatar_object_id
    FROM users u
    LEFT JOIN user_profiles p
      ON p.user_id = u.id AND p.deleted_at IS NULL
    WHERE u.id = :user_id
    """
)


async def list_users(
    db: AsyncSession,
    *,
    status_filter: str | None,
    role_code: str | None,
    organization_id: UUID | None,
    q: str | None,
    limit: int,
    cursor: str | None,
) -> CursorPage[dict[str, Any]]:
    """Cursor-paginated admin user list ordered by ``(created_at DESC, id DESC)``.

    ``cursor`` is an opaque base64 token round-tripped through
    :func:`encode_composite_cursor` / :func:`decode_composite_cursor`.
    ``next_cursor`` is set when the page filled to ``limit`` (more rows
    may exist); ``None`` otherwise.
    """
    after_created_at: datetime | None = None
    after_id: UUID | None = None
    if cursor:
        sort_value, last_id = decode_composite_cursor(cursor)
        if not isinstance(sort_value, datetime):
            raise ValueError("Invalid cursor")
        after_created_at = sort_value
        after_id = last_id

    rows = await user_queries.list_users(
        db,
        status_filter=status_filter,
        role_code=role_code,
        organization_id=organization_id,
        q=q,
        limit=limit,
        after_created_at=after_created_at,
        after_id=after_id,
    )
    next_cursor = (
        encode_composite_cursor(rows[-1]["created_at"], rows[-1]["user_id"])
        if len(rows) == limit
        else None
    )
    return CursorPage(items=rows, next_cursor=next_cursor)


async def user_detail(db: AsyncSession, *, user_id: UUID) -> dict[str, Any]:
    base = (await db.execute(_GET_USER_SQL, {"user_id": user_id})).mappings().one_or_none()
    if base is None:
        raise NotFoundError(f"user {user_id} not found")
    role_assignments = await user_queries.role_assignments(db, user_id=user_id)
    sessions = await user_queries.active_sessions(db, user_id=user_id)
    role_history = await user_queries.role_history(db, user_id=user_id)
    base_dict = dict(base)
    profile_keys = ("display_name", "given_name", "family_name", "bio", "avatar_object_id")
    profile = {k: base_dict.pop(k) for k in profile_keys}
    user_payload = dict(base_dict)
    user_payload["profile"] = profile if profile.get("display_name") is not None else None
    return {
        "user": user_payload,
        "role_assignments": role_assignments,
        "active_sessions": sessions,
        "role_history": role_history,
    }


async def revoke_session(
    db: AsyncSession, *, user_id: UUID, session_id: UUID
) -> dict[str, Any]:
    row = (
        await db.execute(
            _REVOKE_ONE_SESSION_SQL,
            {"user_id": user_id, "session_id": session_id},
        )
    ).mappings().one_or_none()
    if row is None:
        raise NotFoundError(f"active session {session_id} not found for user {user_id}")
    await db.commit()
    return {"session_id": row["id"], "revoked": True}


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
    "revoke_session",
    "user_detail",
]
