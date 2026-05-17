"""Learner inbox service -- thin wrappers around the query layer.

Exists so the router contract holds: routers depend on ``services``,
services depend on ``queries``. The query helpers already enforce
``user_id`` scoping; the service forwards arguments verbatim and adds
no business logic of its own. We keep it as a separate file (rather
than letting routers import queries directly) so the import-linter
"routers do not call queries directly" contract stays clean and so
future T7.6 callers (cross-feature dispatch) discover only the inbox
surface they're meant to use.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID

from abridgeai.features.notifications.queries import (
    bulk_mark_read as _bulk_mark_read_query,
)
from abridgeai.features.notifications.queries import (
    cursor_list_my_notifications as _cursor_list_query,
)
from abridgeai.features.notifications.queries import (
    dismiss_notification as _dismiss_query,
)
from abridgeai.features.notifications.queries import (
    get_my_notification as _get_my_notification_query,
)
from abridgeai.features.notifications.queries import (
    get_my_unread_count as _unread_count_query,
)
from abridgeai.features.notifications.queries import (
    list_my_preferences as _list_prefs_query,
)
from abridgeai.features.notifications.queries import (
    mark_as_read as _mark_as_read_query,
)
from abridgeai.features.notifications.queries import (
    upsert_preference as _upsert_pref_query,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from abridgeai.features.notifications.models import (
        Notification,
        NotificationPreference,
    )


async def list_my_notifications(
    db: AsyncSession,
    *,
    user_id: UUID,
    unread_only: bool,
    cursor: datetime | None,
    limit: int,
) -> tuple[list[Notification], datetime | None]:
    return await _cursor_list_query(
        db,
        user_id=user_id,
        unread_only=unread_only,
        cursor=cursor,
        limit=limit,
    )


async def get_my_unread_count(db: AsyncSession, *, user_id: UUID) -> int:
    return await _unread_count_query(db, user_id=user_id)


async def mark_notification_read(
    db: AsyncSession,
    *,
    user_id: UUID,
    notification_id: UUID,
) -> Notification | None:
    """Returns the persisted row, or None if not found / not owned."""
    affected = await _mark_as_read_query(db, user_id=user_id, notification_id=notification_id)
    _ = affected
    return await _get_my_notification_query(db, user_id=user_id, notification_id=notification_id)


async def bulk_mark_read(db: AsyncSession, *, user_id: UUID) -> int:
    return await _bulk_mark_read_query(db, user_id=user_id)


async def dismiss_notification(
    db: AsyncSession,
    *,
    user_id: UUID,
    notification_id: UUID,
) -> int:
    return await _dismiss_query(db, user_id=user_id, notification_id=notification_id)


async def list_my_preferences(db: AsyncSession, *, user_id: UUID) -> list[NotificationPreference]:
    return await _list_prefs_query(db, user_id=user_id)


async def upsert_preference(
    db: AsyncSession,
    *,
    user_id: UUID,
    category: str,
    channel: str,
    enabled: bool,
) -> NotificationPreference:
    return await _upsert_pref_query(
        db,
        user_id=user_id,
        category=category,
        channel=channel,
        enabled=enabled,
    )


__all__ = [
    "bulk_mark_read",
    "dismiss_notification",
    "get_my_unread_count",
    "list_my_notifications",
    "list_my_preferences",
    "mark_notification_read",
    "upsert_preference",
]
