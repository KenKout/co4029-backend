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
  the dispatcher stages ``send_email_notification_task`` on the caller's
  SQLAlchemy session and enqueues it from an ``after_commit`` hook. Email send
  is *never* awaited inline -- the API request thread does not block on SMTP.
* If ``arq_pool is None`` and email is enabled, the email send is
  skipped silently. This is the contract for unit tests / sync code
  paths that should never enqueue, and for emergency runtime configs
  where ARQ is intentionally disabled.

T7.6 will wire feature-side callers (quiz completion, interview
evaluation finished, manager enrolled student, material processed) to
this surface.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import event
from sqlalchemy.orm import Session

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
_PENDING_EMAIL_JOBS = "notifications.pending_email_jobs"
_PENDING_EMAIL_TASKS: set[asyncio.Task[None]] = set()


async def _enqueue_email_job(
    arq_pool: object, recipient_user_id: UUID, notification_id: UUID
) -> None:
    await arq_pool.enqueue_job(  # type: ignore[attr-defined]
        EMAIL_NOTIFICATION_TASK_NAME,
        recipient_user_id,
        notification_id,
    )


def _log_deferred_email_failure(task: asyncio.Task[None]) -> None:
    _PENDING_EMAIL_TASKS.discard(task)
    if task.cancelled():
        return
    error = task.exception()
    if error is not None:
        _logger.exception(
            "notification_email_enqueue_failed_after_commit",
            exc_info=(type(error), error, error.__traceback__),
        )


@event.listens_for(Session, "after_commit")
def _enqueue_staged_email_jobs(session: Session) -> None:
    jobs = session.info.pop(_PENDING_EMAIL_JOBS, ())
    if not jobs:
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        _logger.error("notification_email_enqueue_no_running_loop")
        return
    for arq_pool, recipient_user_id, notification_id in jobs:
        task = loop.create_task(
            _enqueue_email_job(arq_pool, recipient_user_id, notification_id)
        )
        _PENDING_EMAIL_TASKS.add(task)
        task.add_done_callback(_log_deferred_email_failure)


@event.listens_for(Session, "after_rollback")
def _clear_staged_email_jobs(session: Session) -> None:
    session.info.pop(_PENDING_EMAIL_JOBS, None)


# Email job-name for the worker.


async def send_notification(
    db: AsyncSession,
    *,
    recipient_user_id: UUID,
    notification_type: str,
    title: str,
    body: str,
    entity_type: str | None = None,
    entity_id: UUID | None = None,
    action_url: str | None = None,
    arq_pool: object | None = None,
) -> Notification:
    """Create the in-app row, then conditionally stage an email job.

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
        action_url=action_url,
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

    db.sync_session.info.setdefault(_PENDING_EMAIL_JOBS, []).append(
        (arq_pool, recipient_user_id, notification.id)
    )
    _logger.info(
        "notification_email_staged_until_commit",
        notification_id=str(notification.id),
        recipient_user_id=str(recipient_user_id),
        category=notification_type,
    )
    return notification


__all__ = ["EMAIL_NOTIFICATION_TASK_NAME", "send_notification"]
