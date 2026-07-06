"""Email delivery service -- STUB for T7.4 (real SMTP wiring is Phase 9).

The worker module (``workers/email.py``) calls
:func:`deliver_email_for_notification` from inside an ``AsyncSession``
context. We log the outcome and stamp the Notification row's
``delivery_status`` so an admin viewing the row can tell whether the
email *would* have been sent.

Per the task body: "STUB the actual SMTP call -- log 'would send email'
and mark a metadata flag. Email backend wiring is Phase 9 cutover
concern."

DESCOPED (2026-06-10, gap-analysis phase-07): the requirements appendix
defines notifications as in-app only (FR-4.6/FR-6.3 — no email FR
exists), so this stub is intentionally NOT being wired to SMTP. Keep
the channel plumbing (preferences, delivery_status) so a future email
requirement only needs a transport implementation here.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import select, update

from abridgeai.core.observability import get_logger
from abridgeai.features.notifications.models import Notification

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


_logger = get_logger(__name__)


async def deliver_email_for_notification(
    db: AsyncSession,
    *,
    notification_id: UUID,
) -> bool:
    """Render + send an email for ``notification_id``.

    Returns ``True`` iff the email was logically dispatched (i.e. the
    notification row exists and is not already terminal). Real SMTP /
    SendGrid integration is out of scope for T7.4 -- we emit a
    structured log and mark ``delivery_status`` so downstream tests and
    observability can see the path was exercised.

    Caller controls commit; this helper only flushes.
    """
    stmt = select(Notification).where(Notification.id == notification_id)
    result = await db.execute(stmt)
    row = result.scalar_one_or_none()
    if row is None:
        _logger.warning(
            "email_notification_missing_row",
            notification_id=str(notification_id),
        )
        return False

    if row.delivery_status in {"sent", "cancelled", "failed"}:
        _logger.info(
            "email_notification_skipped_terminal_status",
            notification_id=str(notification_id),
            delivery_status=row.delivery_status,
        )
        return False

    _logger.info(
        "email_notification_would_send",
        notification_id=str(notification_id),
        recipient_user_id=str(row.user_id),
        category=row.category,
        title=row.title,
    )

    now = datetime.now(tz=UTC)
    await db.execute(
        update(Notification)
        .where(Notification.id == notification_id)
        .values(delivery_status="sent", delivered_at=now)
    )
    await db.flush()
    return True


__all__ = ["deliver_email_for_notification"]
