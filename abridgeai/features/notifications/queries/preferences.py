"""SQL access for ``notification_preferences``.

The dispatch service calls :func:`get_email_preference` to gate email
fan-out. ``in_app`` channel is always-on (per task contract), so the
listing/upsert helpers are scoped to user-controlled rows but the
service never *consults* an in_app preference.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import and_, select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from abridgeai.features.notifications.models import NotificationPreference

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


async def list_my_preferences(db: AsyncSession, *, user_id: UUID) -> list[NotificationPreference]:
    stmt = (
        select(NotificationPreference)
        .where(NotificationPreference.user_id == user_id)
        .order_by(NotificationPreference.category, NotificationPreference.channel)
    )
    result = await db.execute(stmt)
    return list(result.scalars().all())


async def upsert_preference(
    db: AsyncSession,
    *,
    user_id: UUID,
    category: str,
    channel: str,
    enabled: bool,
) -> NotificationPreference:
    """Create or toggle a preference. Idempotent on
    ``(user_id, category, channel)``.
    """
    stmt = (
        pg_insert(NotificationPreference)
        .values(
            user_id=user_id,
            category=category,
            channel=channel,
            enabled=enabled,
        )
        .on_conflict_do_update(
            index_elements=["user_id", "category", "channel"],
            set_={"enabled": enabled},
        )
        .returning(NotificationPreference)
    )
    result = await db.execute(stmt)
    return result.scalar_one()


async def get_email_preference(
    db: AsyncSession,
    *,
    user_id: UUID,
    category: str,
) -> bool:
    """Return ``True`` iff email fan-out is opted in for this category.

    Default policy is **opt-in via row presence**: if no preference row
    exists for ``(user_id, 'email', category)`` we treat it as
    *enabled* (matches the existing UX in ``backend/`` where users only
    see prefs after first toggle). Explicit ``enabled=False`` rows
    disable.
    """
    stmt = select(NotificationPreference.enabled).where(
        and_(
            NotificationPreference.user_id == user_id,
            NotificationPreference.channel == "email",
            NotificationPreference.category == category,
        )
    )
    result = await db.execute(stmt)
    value = result.scalar_one_or_none()
    if value is None:
        return True
    return bool(value)


__all__ = [
    "get_email_preference",
    "list_my_preferences",
    "upsert_preference",
]
