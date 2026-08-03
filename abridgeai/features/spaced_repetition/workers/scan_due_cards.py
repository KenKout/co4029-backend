"""ARQ cron task: scan for due cards and dispatch in-app notifications (T7.5.8).

Runs hourly (registered in :mod:`abridgeai.workers.arq_app`). Finds students
whose card review window opens within the next hour and creates one
``Notification`` row per such student via the T7.4 dispatch surface.

Notification fatigue policy
---------------------------
The scan runs hourly (per §7.5.8) so newly-due cards surface within an hour.
On its own that would re-notify a standing backlog every hour — a student who
is behind would be pinged 24×/day. Two guards prevent that:

* Per-student cooldown. A student who already received a
  ``spaced_repetition`` notification within
  ``notifications.sr_reminder_cooldown_hours`` (admin-configurable, default
  24h) is skipped this run. The hourly scan stays responsive to *new*
  backlog while nobody is reminded more than once per window.
* ``Quiz.reminders_enabled`` trigger gate. A ping is sent only to students
  who have at least one due card from a quiz with reminders turned on; a
  student whose due backlog comes entirely from opted-out quizzes is never
  pinged (the flag was previously ignored). The gate decides *who* gets a
  ping — it does NOT shrink the number in the notification, which always
  mirrors the cards-due page (see the count contract in ``_run``).

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
    """ARQ cron task: notify students who have spaced-repetition cards due.

    "Due" means ``due_at`` has passed, backlog included — see the comment in
    :func:`_run`. An earlier version used a forward-looking one-hour window,
    which skipped overdue cards entirely and disagreed with the counts shown on
    every read surface.

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
    # Cross-feature: quizzes/courses/notifications models are imported INSIDE
    # the function to stay within the ``Features-are-independent`` import-linter
    # contract, matching the convention used by _dispatch_for_student below.
    from abridgeai.core.runtime_settings import resolve_setting
    from abridgeai.features.courses.models import Lesson
    from abridgeai.features.notifications.models import Notification
    from abridgeai.features.quizzes.models import Quiz, QuizQuestion, QuizSourceLesson

    started_at = datetime.now(tz=UTC)
    # Count cards that are DUE, i.e. due_at has passed — including everything
    # already overdue.
    #
    # COUNT CONTRACT: the number in the notification MUST equal the number the
    # student sees on the cards-due page (/me/cards-due → _CARDS_DUE_SQL and
    # get_due_card_count). The joins, the due predicate (due_at <= NOW()), the
    # reviewability filter (review_status='approved') and the soft-delete
    # filters mirror that read exactly — including the deliberate ABSENCE of a
    # reminders_enabled filter — so the two can never disagree.
    #
    # This previously used a forward window (due_at BETWEEN now AND now+1h),
    # which counted only cards *about to become* due and silently excluded the
    # backlog. Two consequences, both observed on the dev DB (68 cards all
    # overdue, 0 in the next hour):
    #   1. A student with a real backlog got NO reminder at all — the one case
    #      the reminder exists for.
    #   2. The number in the notification disagreed with every read surface,
    #      which count due_at <= NOW(). Same words, different query.
    #
    # reminders_enabled is applied as a TRIGGER, not a count filter: the
    # per-lesson rows carry a reminder-eligible count (cards from quizzes with
    # the flag on), and a student is only pinged if that count is > 0. A
    # student whose backlog comes entirely from opted-out quizzes gets no ping
    # (teachers who left reminders off never generate reminders), but a pinged
    # student is told the FULL backlog — the same number the page shows. Note a
    # quiz reaches a lesson via quiz_source_lessons — there is no lesson_id
    # column on quizzes.
    #
    # Grouping by lesson carries lesson identity through so the notification can
    # name what is due; the old flat count discarded the per-lesson structure
    # that SM-2 scheduling actually produces.
    due_count_expr = func.count().label("due_count")
    reminder_eligible_expr = (
        func.count().filter(Quiz.reminders_enabled.is_(True)).label("reminder_eligible")
    )
    stmt = (
        select(
            StudentCardState.student_id,
            QuizSourceLesson.lesson_id.label("lesson_id"),
            due_count_expr,
            reminder_eligible_expr,
        )
        .join(QuizQuestion, QuizQuestion.id == StudentCardState.question_id)
        .join(Quiz, Quiz.id == QuizQuestion.quiz_id)
        .join(QuizSourceLesson, QuizSourceLesson.quiz_id == Quiz.id)
        .join(Lesson, Lesson.id == QuizSourceLesson.lesson_id)
        .where(
            StudentCardState.due_at.is_not(None),
            StudentCardState.due_at <= started_at,
            QuizQuestion.deleted_at.is_(None),
            Quiz.deleted_at.is_(None),
            Lesson.deleted_at.is_(None),
            QuizQuestion.review_status == "approved",
        )
        .group_by(
            StudentCardState.student_id,
            QuizSourceLesson.lesson_id,
        )
    )
    rows = (await db.execute(stmt)).all()

    # Fold the per-lesson rows into one dispatch per student, so a student with
    # cards across three lessons gets a single notification that names them
    # rather than three separate pings. reminder_eligible (>0 anywhere) marks
    # the student as pingable — the reminders_enabled trigger gate.
    by_student: dict[UUID, list[tuple[UUID, int]]] = {}
    has_reminder_eligible: set[UUID] = set()
    for row in rows:
        student_id: UUID = row.student_id
        lesson_id: UUID = row.lesson_id
        count: int = int(row.due_count)
        by_student.setdefault(student_id, []).append((lesson_id, count))
        if int(row.reminder_eligible) > 0:
            has_reminder_eligible.add(student_id)

    if not by_student:
        _logger.info("sr.scan_due_cards.completed", students_notified=0, duration_ms=0)
        return

    # Notification-fatigue cooldown (admin-configurable). The scan runs hourly
    # so new backlog surfaces promptly, but a student who already received an
    # SR reminder within the cooldown window is skipped this run — otherwise a
    # standing backlog re-pings them every single hour, which is the spam this
    # guards against. Resolved at the global (deployment) scope: the worker is
    # not org-bound, and this is an operational fatigue knob rather than a
    # per-tenant policy.
    cooldown_hours = int(await resolve_setting(db, "notifications.sr_reminder_cooldown_hours"))
    cutoff = started_at - timedelta(hours=cooldown_hours)
    recent_stmt = select(Notification.user_id.distinct()).where(
        Notification.category == "spaced_repetition",
        Notification.user_id.in_(list(by_student.keys())),
        Notification.created_at >= cutoff,
    )
    recently_notified: set[UUID] = {
        row[0] for row in (await db.execute(recent_stmt)).all()
    }

    dispatched = 0
    skipped_cooldown = 0
    skipped_no_reminders = 0
    for student_id, lesson_counts in by_student.items():
        if student_id not in has_reminder_eligible:
            # reminders_enabled trigger gate: the student's backlog comes
            # entirely from opted-out quizzes — no ping.
            skipped_no_reminders += 1
            continue
        if student_id in recently_notified:
            skipped_cooldown += 1
            continue
        await _dispatch_for_student(db, student_id=student_id, lesson_counts=lesson_counts)
        dispatched += 1

    duration_ms = int((datetime.now(tz=UTC) - started_at).total_seconds() * 1000)
    _logger.info(
        "sr.scan_due_cards.completed",
        students_notified=dispatched,
        students_skipped_no_reminders=skipped_no_reminders,
        students_skipped_cooldown=skipped_cooldown,
        cooldown_hours=cooldown_hours,
        duration_ms=duration_ms,
    )


async def _dispatch_for_student(
    db: AsyncSession,
    *,
    student_id: UUID,
    lesson_counts: list[tuple[UUID, int]],
) -> None:
    # Cross-feature: import inside function so the static import graph
    # stays within the ``Features-are-independent`` contract; the runtime
    # edge is authorised via ``ignore_imports`` entries in pyproject.
    from abridgeai.features.courses.models import Lesson
    from abridgeai.features.identity.api.public import get_user_locale
    from abridgeai.features.notifications import messages
    from abridgeai.features.notifications.services.dispatch import send_notification

    due_count = sum(count for _, count in lesson_counts)
    if due_count == 0:
        return

    # Resolve lesson titles so the notification can say WHICH lesson needs
    # review. A student in three courses can't act on a bare total.
    lesson_ids = [lesson_id for lesson_id, _ in lesson_counts]
    title_rows = (
        await db.execute(select(Lesson.id, Lesson.title).where(Lesson.id.in_(lesson_ids)))
    ).all()
    titles: dict[UUID, str] = {row[0]: row[1] for row in title_rows}

    # Largest backlog first — that's the most useful lesson to name.
    ranked = sorted(
        (
            (titles.get(lesson_id) or "", count)
            for lesson_id, count in lesson_counts
            if titles.get(lesson_id)
        ),
        key=lambda pair: (-pair[1], pair[0]),
    )

    locale = await get_user_locale(db, student_id)
    # Single-lesson backlogs deep-link straight into that lesson's review;
    # multi-lesson ones fall back to the progress page, which lists them all.
    action_url = (
        f"/study/cards-due?lesson={lesson_ids[0]}" if len(ranked) == 1 else "/study/cards-due"
    )
    await send_notification(
        db,
        recipient_user_id=student_id,
        notification_type="spaced_repetition",
        title=messages.due_cards_title(due_count=due_count, locale=locale),
        body=messages.due_cards_body(
            lesson_counts=ranked,
            due_count=due_count,
            locale=locale,
        ),
        action_url=action_url,
    )


__all__ = ["SYSTEM_ACTOR_ID", "scan_due_cards_task"]
