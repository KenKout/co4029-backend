"""ARQ task: send the email channel for a notification (T7.4).

Mirrors the T6.13 worker discipline (commit ``ec1628e``):

1. ``set_worker_actor(actor_id)`` first line -- T0.8 audit binding via
   ``current_actor_var`` so any audit columns updated downstream
   (e.g. ``Notification.updated_by``) inherit the right actor.
2. ``bind_request_context(...)`` for structured-log tracing.
3. Open an ``AsyncSession`` via the canonical ``get_sessionmaker()``.
4. Delegate to the email service. Re-raise on failure so ARQ's retry
   policy applies.
5. ``finally``: clear actor + log context so neighbouring jobs don't
   leak state.

Phase 0.8 convention: ``actor_id`` is the FIRST argument after ``ctx``.
The dispatcher (``services.dispatch.send_notification``) enqueues with
``(recipient_user_id, notification_id)`` -- recipient doubles as actor
because they are the row's "owner" from an audit perspective.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from abridgeai.core.audit import current_actor_var
from abridgeai.core.db import get_sessionmaker
from abridgeai.core.observability import (
    bind_request_context,
    clear_request_context,
    get_logger,
)
from abridgeai.features.notifications.services import email as email_service
from abridgeai.workers.actor import set_worker_actor

_logger = get_logger(__name__)


async def send_email_notification_task(
    ctx: dict[str, Any],
    actor_id: UUID,
    notification_id: UUID,
) -> None:
    """ARQ task: log + stamp the email-channel delivery for one notification."""
    _ = ctx
    set_worker_actor(actor_id)
    bind_request_context(
        notification_id=str(notification_id),
        actor_id=str(actor_id),
    )
    sessionmaker = get_sessionmaker()
    try:
        async with sessionmaker() as db:
            try:
                await email_service.deliver_email_for_notification(
                    db, notification_id=notification_id
                )
                await db.commit()
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception:
                await db.rollback()
                _logger.exception(
                    "email_notification_task_failed",
                    notification_id=str(notification_id),
                )
                raise
    finally:
        current_actor_var.set(None)
        clear_request_context()


__all__ = ["send_email_notification_task"]
