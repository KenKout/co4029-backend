"""SQL access for the ``notifications`` table.

All queries scope by ``user_id`` to prevent cross-user reads (the
``test_other_user_cannot_read_my_notifications`` integration check
exercises this). Soft-delete via SoftDeleteMixin is NOT used here -- §B5
forbids it -- so dismissal is implemented as
``delivery_status='cancelled'`` and the listing helpers explicitly filter
that value out.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import and_, func, select, update

from abridgeai.features.notifications.models import Notification

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


_PAGE_SIZE_DEFAULT = 20
_PAGE_SIZE_MAX = 100


def _utcnow() -> datetime:
    return datetime.now(tz=UTC)


async def insert_notification(
    db: AsyncSession,
    *,
    user_id: UUID,
    category: str,
    title: str,
    body: str,
    entity_type: str | None,
    entity_id: UUID | None,
    action_url: str | None = None,
    delivery_status: str = "pending",
) -> Notification:
    """Persist a Notification row (no commit -- caller controls transaction)."""
    row = Notification(
        user_id=user_id,
        category=category,
        title=title,
        body=body,
        entity_type=entity_type,
        entity_id=entity_id,
        action_url=action_url,
        delivery_status=delivery_status,
    )
    db.add(row)
    await db.flush()
    await db.refresh(row)
    return row


async def get_my_notification(
    db: AsyncSession,
    *,
    user_id: UUID,
    notification_id: UUID,
) -> Notification | None:
    """Fetch a notification owned by ``user_id``; ``None`` if missing or dismissed."""
    stmt = select(Notification).where(
        and_(
            Notification.id == notification_id,
            Notification.user_id == user_id,
            Notification.delivery_status != "cancelled",
        )
    )
    result = await db.execute(stmt)
    return result.scalar_one_or_none()


async def cursor_list_my_notifications(
    db: AsyncSession,
    *,
    user_id: UUID,
    unread_only: bool = False,
    cursor: datetime | None = None,
    limit: int = _PAGE_SIZE_DEFAULT,
) -> tuple[list[Notification], datetime | None]:
    """Cursor-paginated list ordered by ``created_at DESC, id DESC``.

    Cursor is the ``created_at`` of the last item from the previous page.
    Returns ``(rows, next_cursor)`` where ``next_cursor`` is ``None`` when
    the page is the last.
    """
    if limit <= 0 or limit > _PAGE_SIZE_MAX:
        limit = _PAGE_SIZE_DEFAULT

    conditions = [
        Notification.user_id == user_id,
        Notification.delivery_status != "cancelled",
    ]
    if unread_only:
        conditions.append(Notification.read_at.is_(None))
    if cursor is not None:
        conditions.append(Notification.created_at < cursor)

    stmt = (
        select(Notification)
        .where(and_(*conditions))
        .order_by(Notification.created_at.desc(), Notification.id.desc())
        .limit(limit + 1)
    )
    result = await db.execute(stmt)
    rows = list(result.scalars().all())

    next_cursor: datetime | None = None
    if len(rows) > limit:
        rows = rows[:limit]
        next_cursor = rows[-1].created_at
    return rows, next_cursor


async def get_my_unread_count(db: AsyncSession, *, user_id: UUID) -> int:
    """Count unread, non-dismissed notifications for ``user_id``."""
    stmt = select(func.count(Notification.id)).where(
        and_(
            Notification.user_id == user_id,
            Notification.read_at.is_(None),
            Notification.delivery_status != "cancelled",
        )
    )
    result = await db.execute(stmt)
    return int(result.scalar_one() or 0)


async def mark_as_read(
    db: AsyncSession,
    *,
    user_id: UUID,
    notification_id: UUID,
) -> int:
    """Stamp ``read_at`` if not yet read. Returns rows affected."""
    stmt = (
        update(Notification)
        .where(
            and_(
                Notification.id == notification_id,
                Notification.user_id == user_id,
                Notification.delivery_status != "cancelled",
                Notification.read_at.is_(None),
            )
        )
        .values(read_at=_utcnow())
    )
    result = await db.execute(stmt)
    return int(getattr(result, "rowcount", 0) or 0)


async def bulk_mark_read(db: AsyncSession, *, user_id: UUID) -> int:
    """Stamp ``read_at`` on every unread, non-dismissed row. Returns rows affected."""
    now = _utcnow()
    stmt = (
        update(Notification)
        .where(
            and_(
                Notification.user_id == user_id,
                Notification.delivery_status != "cancelled",
                Notification.read_at.is_(None),
            )
        )
        .values(read_at=now)
    )
    result = await db.execute(stmt)
    return int(getattr(result, "rowcount", 0) or 0)


async def dismiss_notification(
    db: AsyncSession,
    *,
    user_id: UUID,
    notification_id: UUID,
) -> int:
    """User-dismiss: set ``delivery_status='cancelled'`` (idempotent).

    Returns rows affected. §B5 forbids ``SoftDeleteMixin`` on this table;
    the ``cancelled`` enum value is the equivalent end-state and is
    filtered out of every list/get/count helper above.
    """
    stmt = (
        update(Notification)
        .where(
            and_(
                Notification.id == notification_id,
                Notification.user_id == user_id,
                Notification.delivery_status != "cancelled",
            )
        )
        .values(delivery_status="cancelled")
    )
    result = await db.execute(stmt)
    return int(getattr(result, "rowcount", 0) or 0)


__all__ = [
    "bulk_mark_read",
    "cursor_list_my_notifications",
    "dismiss_notification",
    "get_my_notification",
    "get_my_unread_count",
    "insert_notification",
    "mark_as_read",
]
