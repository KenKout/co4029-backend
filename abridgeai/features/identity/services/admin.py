"""Admin-side identity reads (T1.9 ``GET /users/{id}`` + ``GET /users``).

Two helpers feed the admin lookup router:

* :func:`get_user_with_profile` -- single-user lookup by id.
* :func:`list_users` -- cursor-paginated list (base64 of last user id).

Routers must not import ``queries.*`` directly (import-linter contract #2),
so this service translates the raw row reads into ``UserRead`` instances.
The cursor format is a base64-encoded UUID of the last item on the previous
page; the implementation is intentionally minimal (sort by ``users.id``).
Future work may add timestamp-based opaque cursors per Reconciliation §A10.
"""

from __future__ import annotations

import base64
from typing import TYPE_CHECKING
from uuid import UUID

from abridgeai.features.identity.queries import users as user_queries
from abridgeai.features.identity.schemas import UserListPage, UserRead
from abridgeai.features.identity.services.profile import serialize_user

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


_DEFAULT_LIMIT = 50
_MAX_LIMIT = 100


def _encode_cursor(user_id: UUID) -> str:
    return base64.urlsafe_b64encode(str(user_id).encode()).decode().rstrip("=")


def _decode_cursor(cursor: str) -> UUID:
    padding = "=" * (-len(cursor) % 4)
    raw = base64.urlsafe_b64decode((cursor + padding).encode()).decode()
    return UUID(raw)


async def get_user_with_profile(db: AsyncSession, user_id: UUID) -> UserRead | None:
    """Return ``UserRead`` for ``user_id`` (with profile if present), or ``None``."""
    user = await user_queries.get_user(db, user_id)
    if user is None:
        return None
    profile = await user_queries.get_profile(db, user_id)
    return serialize_user(user, profile)


async def list_users(
    db: AsyncSession,
    *,
    cursor: str | None = None,
    limit: int = _DEFAULT_LIMIT,
) -> UserListPage:
    """Cursor-paginated user list ordered by ``users.id``.

    ``cursor`` is the base64-encoded UUID of the last item on the previous
    page. ``limit`` is clamped to ``_MAX_LIMIT``. ``next_cursor`` is set when
    the page was full (i.e. there may be more rows); ``None`` otherwise.
    """
    capped = min(max(limit, 1), _MAX_LIMIT)
    after: UUID | None = None
    if cursor:
        try:
            after = _decode_cursor(cursor)
        except (ValueError, UnicodeDecodeError) as exc:
            raise ValueError("Invalid cursor") from exc

    rows = await user_queries.list_users(db, after=after, limit=capped)
    items: list[UserRead] = []
    profiles = {p.user_id: p for p in await user_queries.list_profiles(db, [u.id for u in rows])}
    for user in rows:
        items.append(serialize_user(user, profiles.get(user.id)))

    next_cursor = _encode_cursor(rows[-1].id) if len(rows) == capped else None
    return UserListPage(items=items, next_cursor=next_cursor)


__all__ = ["get_user_with_profile", "list_users"]
