"""Learner notification router (T7.4).

All endpoints are scoped to the authenticated user via ``user_id`` filters
in the underlying query helpers; cross-user reads are blocked by the
``WHERE user_id = :current.user_id`` clause baked into every query, so
attempts to read someone else's notification surface as a 404 (rather
than leaking existence via 403).
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.core.db import get_db
from abridgeai.core.security import CurrentUser, get_current_user
from abridgeai.features.notifications.models import (
    NOTIFICATION_CHANNELS,
    NOTIFICATION_PREFERENCE_CATEGORIES,
)
from abridgeai.features.notifications.schemas.public import (
    NotificationPreferenceRead,
    NotificationPreferenceUpdate,
    NotificationRead,
    UnreadCount,
)
from abridgeai.features.notifications.services import inbox

router = APIRouter(tags=["notifications-learner"])


def _not_found(resource: str, ident: object) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail={"error": "not_found", "resource": resource, "id": str(ident)},
    )


def _bad_request(message: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail={"error": "bad_request", "message": message},
    )


@router.get("/me/notifications", response_model=list[NotificationRead])
async def list_my_notifications(
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    unread_only: Annotated[bool, Query()] = False,
    cursor: Annotated[datetime | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
) -> list[NotificationRead]:
    rows, _next_cursor = await inbox.list_my_notifications(
        db,
        user_id=current_user.user_id,
        unread_only=unread_only,
        cursor=cursor,
        limit=limit,
    )
    return [NotificationRead.model_validate(r) for r in rows]


@router.get("/me/notifications/unread-count", response_model=UnreadCount)
async def get_my_unread_count(
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> UnreadCount:
    count = await inbox.get_my_unread_count(db, user_id=current_user.user_id)
    return UnreadCount(unread=count)


@router.patch(
    "/me/notifications/{notification_id}/read",
    response_model=NotificationRead,
)
async def mark_notification_read(
    notification_id: UUID,
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> NotificationRead:
    row = await inbox.mark_notification_read(
        db, user_id=current_user.user_id, notification_id=notification_id
    )
    if row is None:
        raise _not_found("notification", notification_id)
    await db.commit()
    return NotificationRead.model_validate(row)


@router.post(
    "/me/notifications/mark-all-read",
    response_model=UnreadCount,
)
async def mark_all_my_notifications_read(
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> UnreadCount:
    affected = await inbox.bulk_mark_read(db, user_id=current_user.user_id)
    await db.commit()
    return UnreadCount(unread=affected)


@router.delete(
    "/me/notifications/{notification_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def dismiss_my_notification(
    notification_id: UUID,
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> None:
    affected = await inbox.dismiss_notification(
        db, user_id=current_user.user_id, notification_id=notification_id
    )
    if affected == 0:
        raise _not_found("notification", notification_id)
    await db.commit()


@router.get(
    "/me/notification-preferences",
    response_model=list[NotificationPreferenceRead],
)
async def list_my_notification_preferences(
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[NotificationPreferenceRead]:
    rows = await inbox.list_my_preferences(db, user_id=current_user.user_id)
    return [NotificationPreferenceRead.model_validate(r) for r in rows]


@router.patch(
    "/me/notification-preferences/{category}/{channel}",
    response_model=NotificationPreferenceRead,
)
async def upsert_my_notification_preference(
    category: str,
    channel: str,
    payload: NotificationPreferenceUpdate,
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> NotificationPreferenceRead:
    if category not in NOTIFICATION_PREFERENCE_CATEGORIES:
        raise _bad_request(f"unknown category: {category}")
    if channel not in NOTIFICATION_CHANNELS:
        raise _bad_request(f"unknown channel: {channel}")

    row = await inbox.upsert_preference(
        db,
        user_id=current_user.user_id,
        category=category,
        channel=channel,
        enabled=payload.enabled,
    )
    await db.commit()
    return NotificationPreferenceRead.model_validate(row)


__all__ = ["router"]
