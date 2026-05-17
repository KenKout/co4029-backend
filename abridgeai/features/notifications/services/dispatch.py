"""Notification dispatch surface (T7.4).

Cross-feature entrypoint. Other features (quizzes, interviews, materials,
enrollments, admin announcements) call :func:`send_notification` from
their service layer when they want to notify a user.

Contract:

* In-app channel is **always-on** -- a row is created in ``notifications``
  for every call, regardless of preference state. The user controls
  visibility downstream via dismissal (delivery_status='cancelled') or
  read state.
* Email channel is gated by :func:`get_email_preference`. If enabled,
  the dispatcher enqueues ``send_email_notification_task`` on the
  provided ARQ pool. Email send is *never* awaited inline -- the API
  request thread does not block on SMTP.
* If ``arq_pool is None`` and email is enabled, the email send is
  skipped silently. This is the contract for unit tests / sync code
  paths that should never enqueue, and for emergency runtime configs
  where ARQ is intentionally disabled.

T7.6 will wire feature-side callers (quiz completion, interview
evaluation finished, manager enrolled student, material processed) to
this surface.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

from abridgeai.core.observability import get_logger
from abridgeai.features.notifications.queries.notifications import (
    insert_notification,
)
from abridgeai.features.notifications.queries.preferences import (
    get_email_preference,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from abridgeai.features.notifications.models import Notification


_logger = get_logger(__name__)

EMAIL_NOTIFICATION_TASK_NAME = "send_email_notification_task"
"""ARQ job-name for the email worker.

Locked here so the worker registration and the dispatcher always agree.
Mirrors the T6.11/T6.13 pattern (``run_interview_generation_task``,
``evaluate_interview_session_task``) -- canonical Python function name
used verbatim as the enqueue string to dodge the T5.14 reconcile-trap.
"""


async def send_notification(
    db: AsyncSession,
    *,
    recipient_user_id: UUID,
    notification_type: str,
    title: str,
    body: str,
    entity_type: str | None = None,
    entity_id: UUID | None = None,
    arq_pool: object | None = None,
) -> Notification:
    """Create the in-app row, then conditionally enqueue an email.

    Parameters
    ----------
    db
        Caller-owned ``AsyncSession``. We **do not** commit here -- the
        caller's unit-of-work decides when to commit so the notification
        write joins the surrounding transaction (e.g. the same tx that
        flips ``QuizGenerationRun.status='ready'``).
    recipient_user_id
        FK into ``users.id`` -- the user who should see this notification
        in their inbox.
    notification_type
        One of ``NOTIFICATION_CATEGORIES``. Validated by the DB CHECK
        constraint at flush time; we don't pre-validate so callers see
        the same error path whether they're typo'd at runtime or at
        schema-evolution time.
    title, body
        Inbox text shown to the user.
    entity_type, entity_id
        Optional pointer to the entity that caused this notification
        (e.g. ``('quiz', <quiz_id>)``). The legacy router exposes these
        via ``NotificationRead`` so the frontend can deep-link.
    arq_pool
        ARQ pool exposing ``enqueue_job(name, *args, **kwargs)``. Type
        is ``object`` to avoid importing ``arq.connections.ArqRedis``
        here and inflating the import graph; the import-linter
        ``services-do-not-touch-sqlalchemy`` contract is unaffected.

    Returns
    -------
    Notification
        The persisted row (after ``flush + refresh``).
    """
    notification = await insert_notification(
        db,
        user_id=recipient_user_id,
        category=notification_type,
        title=title,
        body=body,
        entity_type=entity_type,
        entity_id=entity_id,
        delivery_status="pending",
    )

    if arq_pool is None:
        _logger.debug(
            "notification_dispatched_no_pool",
            notification_id=str(notification.id),
            recipient_user_id=str(recipient_user_id),
            category=notification_type,
        )
        return notification

    email_enabled = await get_email_preference(
        db, user_id=recipient_user_id, category=notification_type
    )
    if not email_enabled:
        _logger.info(
            "notification_email_skipped_pref_disabled",
            notification_id=str(notification.id),
            recipient_user_id=str(recipient_user_id),
            category=notification_type,
        )
        return notification

    await arq_pool.enqueue_job(  # type: ignore[attr-defined]
        EMAIL_NOTIFICATION_TASK_NAME,
        recipient_user_id,
        notification.id,
    )
    _logger.info(
        "notification_email_enqueued",
        notification_id=str(notification.id),
        recipient_user_id=str(recipient_user_id),
        category=notification_type,
    )
    return notification


__all__ = ["EMAIL_NOTIFICATION_TASK_NAME", "send_notification"]
