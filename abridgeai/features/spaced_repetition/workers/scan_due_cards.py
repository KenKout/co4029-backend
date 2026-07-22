"""ARQ cron task: scan for due cards and dispatch in-app notifications (T7.5.8).

Runs hourly (registered in :mod:`abridgeai.workers.arq_app`). Finds students
whose card review window opens within the next hour and creates one
``Notification`` row per such student via the T7.4 dispatch surface.

Notification fatigue policy
---------------------------
The plan caps cron frequency at hourly (per §7.5.8) -- more frequent
scans would generate spam-grade notification volume because cards
naturally bunch into the same hour bucket. The hourly window aligns
with the SM-2 minimum-interval floor.

In-app channel only
-------------------
``send_notification`` is invoked with ``arq_pool=None``. Per
``services/dispatch.py`` contract this creates the in-app row and
silently skips email enqueue. Email-channel notifications for due
cards are intentionally out of scope; users get the inbox row only.

Cross-feature contract
----------------------
Importing :func:`abridgeai.features.notifications.services.dispatch.send_notification`
from inside this worker crosses the import-linter
``Features-are-independent`` contract. The import is therefore done
**inside** the dispatch helper so the static graph stays clean, and an
``ignore_imports`` entry is added in ``pyproject.toml`` to authorise
the runtime cross-feature call.

Phase 0.8 worker convention
---------------------------
``actor_id`` is the FIRST argument after ``ctx`` so the audit listener
stamps ``updated_by`` on dispatched ``Notification`` rows. Cron-initiated
runs use the canonical migration-seeded system user
(``00000000-0000-0000-0000-000000000001``) so audit FKs into ``users``
always resolve.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.core.audit import current_actor_var
from abridgeai.core.db import get_sessionmaker
from abridgeai.core.observability import (
    bind_request_context,
    clear_request_context,
    get_logger,
)
from abridgeai.features.spaced_repetition.models import StudentCardState
from abridgeai.workers.actor import set_worker_actor

SYSTEM_ACTOR_ID = UUID("00000000-0000-0000-0000-000000000001")
"""Migration-0004 system user. Stable across the suite (see ``conftest.py``)."""

_logger = get_logger(__name__)


async def scan_due_cards_task(
    ctx: dict[str, Any],
    actor_id: UUID = SYSTEM_ACTOR_ID,
) -> None:
    """ARQ cron task: notify students with cards becoming due in the next hour.

    Phase 0.8 worker convention: ``actor_id`` is the first arg after ``ctx``
    so the audit listener stamps ``updated_by`` on every dispatched
    ``Notification`` row.
    """
    _ = ctx
    set_worker_actor(actor_id)
    bind_request_context(task="sr.scan_due_cards")
    sessionmaker = get_sessionmaker()
    try:
        async with sessionmaker() as db:
            try:
                await _run(db)
                await db.commit()
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception:
                await db.rollback()
                _logger.exception("sr.scan_due_cards.failed")
                raise
    finally:
        current_actor_var.set(None)
        clear_request_context()


async def _run(db: AsyncSession) -> None:
    started_at = datetime.now(tz=UTC)
    window_end = started_at + timedelta(hours=1)
    due_count_expr = func.count().label("due_count")
    stmt = (
        select(StudentCardState.student_id, due_count_expr)
        .where(
            StudentCardState.due_at >= started_at,
            StudentCardState.due_at <= window_end,
        )
        .group_by(StudentCardState.student_id)
        .having(func.count() > 0)
    )
    rows = await db.execute(stmt)
    dispatched = 0
    for row in rows.all():
        student_id: UUID = row[0]
        due_count: int = int(row[1])
        await _dispatch_for_student(db, student_id=student_id, due_count=due_count)
        dispatched += 1

    duration_ms = int((datetime.now(tz=UTC) - started_at).total_seconds() * 1000)
    _logger.info(
        "sr.scan_due_cards.completed",
        students_notified=dispatched,
        duration_ms=duration_ms,
    )


async def _dispatch_for_student(
    db: AsyncSession,
    *,
    student_id: UUID,
    due_count: int,
) -> None:
    # Cross-feature: import inside function so the static import graph
    # stays within the ``Features-are-independent`` contract; the runtime
    # edge is authorised via ``ignore_imports`` entries in pyproject.
    from abridgeai.features.identity.api.public import get_user_locale
    from abridgeai.features.notifications import messages
    from abridgeai.features.notifications.services.dispatch import send_notification

    locale = await get_user_locale(db, student_id)
    await send_notification(
        db,
        recipient_user_id=student_id,
        notification_type="spaced_repetition",
        title=messages.due_cards_title(due_count=due_count, locale=locale),
        body=messages.due_cards_body(locale=locale),
        # Cross-course summary ("N cards due") — no single course context, so
        # point at the learner progress page where due cards surface. Option B:
        # the producer builds the navigable path at creation time.
        action_url="/progress",
    )


__all__ = ["SYSTEM_ACTOR_ID", "scan_due_cards_task"]
