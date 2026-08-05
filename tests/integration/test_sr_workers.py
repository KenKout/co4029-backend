"""Integration tests for SR due-card scanner worker (T7.5.8).

The worker is a thin scan + dispatch loop:

* Find students with cards due in the next hour via raw SQL.
* For each student, call the T7.4 ``send_notification`` surface so an
  in-app notification row is created.
* Honour the Phase 0.8 worker convention -- ``actor_id`` first arg
  after ``ctx``, audit context bound for the duration of the task.

Tests cover both **unit-shape** assertions (signature, registration,
JOBS export) and **integration-shape** assertions (worker run against
a seeded DB with ``send_notification`` patched).
"""

from __future__ import annotations

import inspect
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock
from uuid import UUID

import pytest
import pytest_asyncio
from alembic import command
from alembic.config import Config
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from abridgeai.core.audit import current_actor_var, register_audit_listener
from abridgeai.core.config import get_settings
from abridgeai.features.identity import models as _identity_models  # noqa: F401
from abridgeai.features.quizzes import models as _quiz_models  # noqa: F401
from abridgeai.features.spaced_repetition.workers import JOBS, scan_due_cards_task
from abridgeai.features.spaced_repetition.workers import scan_due_cards as worker_mod
from abridgeai.workers.arq_app import WorkerSettings

register_audit_listener()


def _async_url(database_url: str) -> str:
    if "+psycopg_async" in database_url:
        return database_url
    if database_url.startswith("postgresql+psycopg://"):
        return database_url.replace("postgresql+psycopg://", "postgresql+psycopg_async://", 1)
    if database_url.startswith("postgresql://"):
        return database_url.replace("postgresql://", "postgresql+psycopg_async://", 1)
    return database_url


def _ensure_head() -> None:
    cfg_path = Path(__file__).resolve().parents[2] / "alembic.ini"
    cfg = Config(str(cfg_path))
    cfg.set_main_option(
        "script_location",
        str(Path(__file__).resolve().parents[2] / "migrations"),
    )
    command.upgrade(cfg, "head")


@pytest_asyncio.fixture
async def engine() -> AsyncIterator[AsyncEngine]:
    _ensure_head()
    eng = create_async_engine(_async_url(get_settings().database_url), pool_pre_ping=True)
    try:
        yield eng
    finally:
        await eng.dispose()


@pytest_asyncio.fixture
async def session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False, autoflush=False)


@pytest_asyncio.fixture(autouse=True)
async def _purge_sr_state(engine: AsyncEngine) -> AsyncIterator[None]:
    """Wipe SR + quiz state so each test sees an empty due-card set.

    Tests share a single DB and the worker scans globally, so residual
    rows from earlier tests would otherwise leak into later assertions.
    """
    async with engine.begin() as conn:
        await conn.execute(text("DELETE FROM student_card_state"))
    yield
    async with engine.begin() as conn:
        await conn.execute(text("DELETE FROM student_card_state"))


async def _seed_card(
    engine: AsyncEngine,
    *,
    student_id: UUID,
    question_id: UUID,
    due_at: datetime,
    quiz_id: UUID,
    position: int,
    org_id: UUID,
    course_id: UUID,
    module_id: UUID,
    owner_id: UUID,
    create_user: bool = True,
    review_status: str = "approved",
) -> None:
    async with engine.begin() as conn:
        if create_user:
            await conn.execute(
                text("INSERT INTO users (id, primary_email) VALUES (:id, :email)"),
                {"id": student_id, "email": f"sr-due-{student_id.hex[:8]}@test.local"},
            )
        await conn.execute(
            text(
                "INSERT INTO quiz_questions ("
                "id, quiz_id, position, question_type, prompt_text, "
                "expected_response_time_ms, source_refs, review_status"
                ") VALUES ("
                ":id, :quiz, :pos, 'multiple_choice', 'Q?', "
                "30000, '[]'::jsonb, :review_status"
                ")"
            ),
            {
                "id": question_id,
                "quiz": quiz_id,
                "pos": position,
                "review_status": review_status,
            },
        )
        await conn.execute(
            text(
                "INSERT INTO student_card_state ("
                "student_id, question_id, ef, interval_days, repetition_count, "
                "due_at, total_reviews"
                ") VALUES ("
                ":sid, :qid, :ef, 1, 1, :due, 1"
                ") ON CONFLICT (student_id, question_id) DO UPDATE "
                "SET due_at = EXCLUDED.due_at"
            ),
            {"sid": student_id, "qid": question_id, "ef": Decimal("2.5"), "due": due_at},
        )


async def _seed_quiz_root(engine: AsyncEngine) -> tuple[UUID, UUID, UUID, UUID, UUID]:
    """Insert a fresh organization → course → module → quiz; return ids + owner."""
    org_id = uuid.uuid4()
    owner_id = uuid.uuid4()
    course_id = uuid.uuid4()
    module_id = uuid.uuid4()
    quiz_id = uuid.uuid4()
    lesson_id = uuid.uuid4()
    lesson_title = "Scheduling Basics"
    async with engine.begin() as conn:
        await conn.execute(
            text("INSERT INTO organizations (id, slug, name) VALUES (:id, :slug, :name)"),
            {"id": org_id, "slug": f"sr-due-{org_id.hex[:8]}", "name": "T7.5.8"},
        )
        await conn.execute(
            text("INSERT INTO users (id, primary_email) VALUES (:id, :email)"),
            {"id": owner_id, "email": f"owner-{owner_id.hex[:8]}@test.local"},
        )
        await conn.execute(
            text(
                "INSERT INTO courses (id, organization_id, owner_user_id, slug, title) "
                "VALUES (:id, :org, :owner, :slug, :title)"
            ),
            {
                "id": course_id,
                "org": org_id,
                "owner": owner_id,
                "slug": f"course-{course_id.hex[:8]}",
                "title": "T7.5.8 Course",
            },
        )
        await conn.execute(
            text(
                "INSERT INTO modules (id, course_id, title, position, status) "
                "VALUES (:id, :course, 'M', 1, 'published')"
            ),
            {"id": module_id, "course": course_id},
        )
        await conn.execute(
            text(
                "INSERT INTO quizzes (id, course_id, module_id, title, status, reminders_enabled) "
                "VALUES (:id, :course, :m, 'Q', 'published', TRUE)"
            ),
            {"id": quiz_id, "course": course_id, "m": module_id},
        )
        # A lesson linked via quiz_source_lessons. The due-cards scan reaches a
        # lesson through that join table (there is no lessons FK on quizzes), and
        # needs the title to name what is due in the notification body.
        await conn.execute(
            text(
                "INSERT INTO lessons (id, module_id, slug, title, status) "
                "VALUES (:id, :m, :slug, :title, 'published')"
            ),
            {
                "id": lesson_id,
                "m": module_id,
                "slug": f"lesson-{lesson_id.hex[:8]}",
                "title": lesson_title,
            },
        )
        await conn.execute(
            text(
                "INSERT INTO quiz_source_lessons (quiz_id, lesson_id) "
                "VALUES (:q, :l)"
            ),
            {"q": quiz_id, "l": lesson_id},
        )
    return org_id, course_id, module_id, quiz_id, owner_id


def _make_sessionmaker_factory(
    session_factory: async_sessionmaker[AsyncSession],
) -> Any:
    """Return a callable that the worker can invoke as ``get_sessionmaker()``.

    ``async_sessionmaker`` itself is callable and yields an ``AsyncSession``
    that doubles as an async context manager, so handing it back wholesale
    matches the worker's ``async with get_sessionmaker()() as db`` shape.
    """
    return lambda: session_factory


def test_jobs_export() -> None:
    assert scan_due_cards_task in JOBS


def test_arq_app_includes_sr_jobs() -> None:
    assert scan_due_cards_task in WorkerSettings.functions


def test_arq_app_includes_due_card_cron() -> None:
    cron_funcs = [job.coroutine for job in WorkerSettings.cron_jobs]
    assert scan_due_cards_task in cron_funcs


def test_scan_due_cards_signature() -> None:
    sig = inspect.signature(scan_due_cards_task)
    params = list(sig.parameters)
    assert params[0] == "ctx"
    assert params[1] == "actor_id"


@pytest.mark.asyncio
async def test_scan_due_cards_dispatches_for_due_students(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    org_id, course_id, module_id, quiz_id, owner_id = await _seed_quiz_root(engine)
    student_ids: list[UUID] = []
    # Overdue: the scan reminds on cards whose due_at has passed.
    due_at = datetime.now(tz=UTC) - timedelta(minutes=30)
    pos = 0
    for _ in range(5):
        student_id = uuid.uuid4()
        student_ids.append(student_id)
        for _q in range(3):
            pos += 1
            await _seed_card(
                engine,
                student_id=student_id,
                question_id=uuid.uuid4(),
                due_at=due_at,
                quiz_id=quiz_id,
                position=pos,
                org_id=org_id,
                course_id=course_id,
                module_id=module_id,
                owner_id=owner_id,
                create_user=(pos % 3 == 1),
            )

    send_mock = AsyncMock(return_value=None)
    monkeypatch.setattr(
        "abridgeai.features.notifications.services.dispatch.send_notification",
        send_mock,
    )
    monkeypatch.setattr(worker_mod, "get_sessionmaker", _make_sessionmaker_factory(session_factory))

    await scan_due_cards_task(ctx={}, actor_id=owner_id)

    assert send_mock.await_count == 5
    notified_ids = {call.kwargs["recipient_user_id"] for call in send_mock.await_args_list}
    assert notified_ids == set(student_ids)
    for call in send_mock.await_args_list:
        assert call.kwargs["notification_type"] == "spaced_repetition"
        assert "3 cards due" in call.kwargs["title"]


@pytest.mark.asyncio
async def test_scan_due_cards_singular_title_when_one_card(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    org_id, course_id, module_id, quiz_id, owner_id = await _seed_quiz_root(engine)
    student_id = uuid.uuid4()
    # Overdue: the scan reminds on cards whose due_at has passed.
    due_at = datetime.now(tz=UTC) - timedelta(minutes=15)
    await _seed_card(
        engine,
        student_id=student_id,
        question_id=uuid.uuid4(),
        due_at=due_at,
        quiz_id=quiz_id,
        position=1,
        org_id=org_id,
        course_id=course_id,
        module_id=module_id,
        owner_id=owner_id,
    )

    send_mock = AsyncMock(return_value=None)
    monkeypatch.setattr(
        "abridgeai.features.notifications.services.dispatch.send_notification",
        send_mock,
    )
    monkeypatch.setattr(worker_mod, "get_sessionmaker", _make_sessionmaker_factory(session_factory))

    await scan_due_cards_task(ctx={}, actor_id=owner_id)

    assert send_mock.await_count == 1
    title = send_mock.await_args_list[0].kwargs["title"]
    assert "1 card due" in title
    assert "cards" not in title


@pytest.mark.asyncio
async def test_scan_due_cards_skips_students_with_no_due_cards(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    org_id, course_id, module_id, quiz_id, owner_id = await _seed_quiz_root(engine)
    far_future = datetime.now(tz=UTC) + timedelta(days=7)
    for idx in range(3):
        await _seed_card(
            engine,
            student_id=uuid.uuid4(),
            question_id=uuid.uuid4(),
            due_at=far_future,
            quiz_id=quiz_id,
            position=idx + 1,
            org_id=org_id,
            course_id=course_id,
            module_id=module_id,
            owner_id=owner_id,
        )

    send_mock = AsyncMock(return_value=None)
    monkeypatch.setattr(
        "abridgeai.features.notifications.services.dispatch.send_notification",
        send_mock,
    )
    monkeypatch.setattr(worker_mod, "get_sessionmaker", _make_sessionmaker_factory(session_factory))

    await scan_due_cards_task(ctx={}, actor_id=owner_id)

    assert send_mock.await_count == 0


@pytest.mark.asyncio
async def test_scan_due_cards_notifies_overdue_not_future(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Overdue cards must notify; cards not yet due must not.

    This replaces an earlier test that asserted the opposite — it pinned a
    forward-looking one-hour window (due_at BETWEEN now AND now+1h), which meant
    a student with a real backlog was never reminded, and the count disagreed
    with every read surface (all of which use due_at <= NOW()).
    """
    org_id, course_id, module_id, quiz_id, owner_id = await _seed_quiz_root(engine)
    overdue_student = uuid.uuid4()
    future_student = uuid.uuid4()
    await _seed_card(
        engine,
        student_id=overdue_student,
        question_id=uuid.uuid4(),
        due_at=datetime.now(tz=UTC) - timedelta(days=3),
        quiz_id=quiz_id,
        position=1,
        org_id=org_id,
        course_id=course_id,
        module_id=module_id,
        owner_id=owner_id,
    )
    await _seed_card(
        engine,
        student_id=future_student,
        question_id=uuid.uuid4(),
        due_at=datetime.now(tz=UTC) + timedelta(hours=2),
        quiz_id=quiz_id,
        position=2,
        org_id=org_id,
        course_id=course_id,
        module_id=module_id,
        owner_id=owner_id,
    )

    send_mock = AsyncMock(return_value=None)
    monkeypatch.setattr(
        "abridgeai.features.notifications.services.dispatch.send_notification",
        send_mock,
    )
    monkeypatch.setattr(worker_mod, "get_sessionmaker", _make_sessionmaker_factory(session_factory))

    await scan_due_cards_task(ctx={}, actor_id=owner_id)

    notified = {c.kwargs["recipient_user_id"] for c in send_mock.await_args_list}
    assert overdue_student in notified, "a student with overdue cards must be reminded"
    assert future_student not in notified, "cards not yet due must not trigger a reminder"


@pytest.mark.asyncio
async def test_scan_due_cards_body_names_the_lesson(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The body must name the lesson, not just give a cross-course total.

    A flat "N cards due" discards the per-lesson structure SM-2 scheduling
    produces; a student in several courses can't tell which one needs work.
    """
    org_id, course_id, module_id, quiz_id, owner_id = await _seed_quiz_root(engine)
    student_id = uuid.uuid4()
    for i in range(3):
        await _seed_card(
            engine,
            student_id=student_id,
            question_id=uuid.uuid4(),
            due_at=datetime.now(tz=UTC) - timedelta(hours=i + 1),
            quiz_id=quiz_id,
            position=i + 1,
            org_id=org_id,
            course_id=course_id,
            module_id=module_id,
            owner_id=owner_id,
            # The user row is created by the first card only; _seed_card would
            # otherwise violate users_pkey on the repeat inserts.
            create_user=(i == 0),
        )

    send_mock = AsyncMock(return_value=None)
    monkeypatch.setattr(
        "abridgeai.features.notifications.services.dispatch.send_notification",
        send_mock,
    )
    monkeypatch.setattr(worker_mod, "get_sessionmaker", _make_sessionmaker_factory(session_factory))

    await scan_due_cards_task(ctx={}, actor_id=owner_id)

    call = next(
        c for c in send_mock.await_args_list if c.kwargs["recipient_user_id"] == student_id
    )
    assert "3" in call.kwargs["title"]
    # The seeded lesson title must appear, with its card count.
    assert "Scheduling Basics" in call.kwargs["body"]
    assert "3 cards" in call.kwargs["body"]
    # Single-lesson backlog deep-links straight to that lesson's review queue.
    assert "/study/cards-due?lesson=" in call.kwargs["action_url"]


@pytest.mark.asyncio
async def test_scan_due_cards_actor_propagates(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    org_id, course_id, module_id, quiz_id, owner_id = await _seed_quiz_root(engine)
    student_id = uuid.uuid4()
    await _seed_card(
        engine,
        student_id=student_id,
        question_id=uuid.uuid4(),
        due_at=datetime.now(tz=UTC) - timedelta(minutes=20),
        quiz_id=quiz_id,
        position=1,
        org_id=org_id,
        course_id=course_id,
        module_id=module_id,
        owner_id=owner_id,
    )

    seen: dict[str, UUID | None] = {"actor": None}

    async def _capture(*_args: Any, **_kwargs: Any) -> None:
        seen["actor"] = current_actor_var.get()

    monkeypatch.setattr(
        "abridgeai.features.notifications.services.dispatch.send_notification",
        _capture,
    )
    monkeypatch.setattr(worker_mod, "get_sessionmaker", _make_sessionmaker_factory(session_factory))

    await scan_due_cards_task(ctx={}, actor_id=owner_id)

    assert seen["actor"] == owner_id
    assert current_actor_var.get() is None


@pytest.mark.asyncio
async def test_scan_due_cards_clears_actor_on_failure(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    org_id, course_id, module_id, quiz_id, owner_id = await _seed_quiz_root(engine)
    await _seed_card(
        engine,
        student_id=uuid.uuid4(),
        question_id=uuid.uuid4(),
        due_at=datetime.now(tz=UTC) - timedelta(minutes=20),
        quiz_id=quiz_id,
        position=1,
        org_id=org_id,
        course_id=course_id,
        module_id=module_id,
        owner_id=owner_id,
    )

    monkeypatch.setattr(
        "abridgeai.features.notifications.services.dispatch.send_notification",
        AsyncMock(side_effect=RuntimeError("dispatch blew up")),
    )
    monkeypatch.setattr(worker_mod, "get_sessionmaker", _make_sessionmaker_factory(session_factory))

    with pytest.raises(RuntimeError, match="dispatch blew up"):
        await scan_due_cards_task(ctx={}, actor_id=owner_id)

    assert current_actor_var.get() is None


@pytest.mark.asyncio
async def test_scan_due_cards_skips_quiz_with_reminders_disabled(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A student whose due cards ALL come from reminders-disabled quizzes gets
    no ping.

    reminders_enabled is a per-quiz teacher flag; when every due card belongs
    to an opted-out quiz the trigger gate suppresses the reminder entirely.
    (When the student ALSO has opted-in cards, the ping fires and reports the
    full backlog — see
    ``test_scan_due_cards_counts_full_backlog_across_mixed_reminders``.)
    """
    org_id, course_id, module_id, quiz_id, owner_id = await _seed_quiz_root(engine)
    # Turn reminders OFF on the seeded quiz.
    async with engine.begin() as conn:
        await conn.execute(
            text("UPDATE quizzes SET reminders_enabled = FALSE WHERE id = :id"),
            {"id": quiz_id},
        )
    await _seed_card(
        engine,
        student_id=uuid.uuid4(),
        question_id=uuid.uuid4(),
        due_at=datetime.now(tz=UTC) - timedelta(minutes=30),
        quiz_id=quiz_id,
        position=1,
        org_id=org_id,
        course_id=course_id,
        module_id=module_id,
        owner_id=owner_id,
    )

    send_mock = AsyncMock(return_value=None)
    monkeypatch.setattr(
        "abridgeai.features.notifications.services.dispatch.send_notification",
        send_mock,
    )
    monkeypatch.setattr(worker_mod, "get_sessionmaker", _make_sessionmaker_factory(session_factory))

    await scan_due_cards_task(ctx={}, actor_id=owner_id)

    assert send_mock.await_count == 0


@pytest.mark.asyncio
async def test_scan_due_cards_counts_full_backlog_across_mixed_reminders(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A ping triggered by one opted-in quiz reports the FULL due backlog.

    reminders_enabled gates WHO gets pinged, not what the notification says.
    A student with due cards from an opted-in quiz AND an opted-out quiz must
    be told the same total the cards-due page shows — the gate used to shrink
    the count to the opted-in subset, so the notification disagreed with the
    landing page (6 due in the ping, 12 on /study/cards-due).
    """
    org_id, course_id, module_id, quiz_on, owner_id = await _seed_quiz_root(engine)
    # Second quiz whose cards are due but reminders are OFF.
    *_, quiz_off, _ = await _seed_quiz_root(engine)
    async with engine.begin() as conn:
        await conn.execute(
            text("UPDATE quizzes SET reminders_enabled = FALSE WHERE id = :id"),
            {"id": quiz_off},
        )
        await conn.execute(
            text(
                "UPDATE lessons SET title = 'Opted-Out Lesson' WHERE id = "
                "(SELECT lesson_id FROM quiz_source_lessons WHERE quiz_id = :q)"
            ),
            {"q": quiz_off},
        )

    student_id = uuid.uuid4()
    due_at = datetime.now(tz=UTC) - timedelta(minutes=30)
    for i in range(3):
        await _seed_card(
            engine,
            student_id=student_id,
            question_id=uuid.uuid4(),
            due_at=due_at,
            quiz_id=quiz_on,
            position=i + 1,
            org_id=org_id,
            course_id=course_id,
            module_id=module_id,
            owner_id=owner_id,
            create_user=(i == 0),
        )
    for i in range(2):
        await _seed_card(
            engine,
            student_id=student_id,
            question_id=uuid.uuid4(),
            due_at=due_at,
            quiz_id=quiz_off,
            position=10 + i,
            org_id=org_id,
            course_id=course_id,
            module_id=module_id,
            owner_id=owner_id,
            create_user=False,
        )

    send_mock = AsyncMock(return_value=None)
    monkeypatch.setattr(
        "abridgeai.features.notifications.services.dispatch.send_notification",
        send_mock,
    )
    monkeypatch.setattr(worker_mod, "get_sessionmaker", _make_sessionmaker_factory(session_factory))

    await scan_due_cards_task(ctx={}, actor_id=owner_id)

    assert send_mock.await_count == 1
    call = send_mock.await_args_list[0]
    assert call.kwargs["recipient_user_id"] == student_id
    # Full backlog (3 + 2), NOT the opted-in subset (3).
    assert "5 cards due" in call.kwargs["title"]
    # The opted-out lesson is reported too — same lessons the page lists.
    assert "Scheduling Basics" in call.kwargs["body"]
    assert "Opted-Out Lesson" in call.kwargs["body"]
    # Multi-lesson backlog falls back to the unscoped page.
    assert call.kwargs["action_url"] == "/study/cards-due"


@pytest.mark.asyncio
async def test_scan_due_cards_excludes_pending_questions(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cards whose question is not 'approved' are not reviewable and must not
    be counted.

    The cards-due page only serves approved questions (``_CARDS_DUE_SQL``
    filters ``review_status = 'approved'``); the notification must agree with
    it, so a pending-question card can't inflate the pinged count.
    """
    org_id, course_id, module_id, quiz_id, owner_id = await _seed_quiz_root(engine)
    student_id = uuid.uuid4()
    due_at = datetime.now(tz=UTC) - timedelta(minutes=30)
    await _seed_card(
        engine,
        student_id=student_id,
        question_id=uuid.uuid4(),
        due_at=due_at,
        quiz_id=quiz_id,
        position=1,
        org_id=org_id,
        course_id=course_id,
        module_id=module_id,
        owner_id=owner_id,
    )
    await _seed_card(
        engine,
        student_id=student_id,
        question_id=uuid.uuid4(),
        due_at=due_at,
        quiz_id=quiz_id,
        position=2,
        org_id=org_id,
        course_id=course_id,
        module_id=module_id,
        owner_id=owner_id,
        create_user=False,
        review_status="pending",
    )

    send_mock = AsyncMock(return_value=None)
    monkeypatch.setattr(
        "abridgeai.features.notifications.services.dispatch.send_notification",
        send_mock,
    )
    monkeypatch.setattr(worker_mod, "get_sessionmaker", _make_sessionmaker_factory(session_factory))

    await scan_due_cards_task(ctx={}, actor_id=owner_id)

    assert send_mock.await_count == 1
    title = send_mock.await_args_list[0].kwargs["title"]
    assert "1 card due" in title
    assert "2 cards" not in title


@pytest.mark.asyncio
async def test_scan_due_cards_cooldown_skips_recently_notified(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A student reminded inside the cooldown window is skipped this run.

    Two students both have the same overdue backlog. One already got a
    ``spaced_repetition`` notification an hour ago (inside the default 24h
    cooldown); the other's last reminder was three days ago (outside it).
    Only the second is notified — this is the anti-spam guard.
    """
    org_id, course_id, module_id, quiz_id, owner_id = await _seed_quiz_root(engine)
    recent_student = uuid.uuid4()
    stale_student = uuid.uuid4()
    due_at = datetime.now(tz=UTC) - timedelta(hours=2)
    for pos, student_id in enumerate((recent_student, stale_student), start=1):
        await _seed_card(
            engine,
            student_id=student_id,
            question_id=uuid.uuid4(),
            due_at=due_at,
            quiz_id=quiz_id,
            position=pos,
            org_id=org_id,
            course_id=course_id,
            module_id=module_id,
            owner_id=owner_id,
        )

    # recent_student got a reminder 1h ago (inside cooldown); stale_student's
    # last reminder was 3 days ago (outside the default 24h window).
    now = datetime.now(tz=UTC)
    notif_ids = [uuid.uuid4(), uuid.uuid4()]
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO notifications "
                "(id, user_id, category, title, body, delivery_status, created_at) "
                "VALUES (:id, :uid, 'spaced_repetition', 'prev', 'prev', 'sent', :created)"
            ),
            {"id": notif_ids[0], "uid": recent_student, "created": now - timedelta(hours=1)},
        )
        await conn.execute(
            text(
                "INSERT INTO notifications "
                "(id, user_id, category, title, body, delivery_status, created_at) "
                "VALUES (:id, :uid, 'spaced_repetition', 'prev', 'prev', 'sent', :created)"
            ),
            {"id": notif_ids[1], "uid": stale_student, "created": now - timedelta(days=3)},
        )

    send_mock = AsyncMock(return_value=None)
    monkeypatch.setattr(
        "abridgeai.features.notifications.services.dispatch.send_notification",
        send_mock,
    )
    monkeypatch.setattr(worker_mod, "get_sessionmaker", _make_sessionmaker_factory(session_factory))

    try:
        await scan_due_cards_task(ctx={}, actor_id=owner_id)

        notified = {c.kwargs["recipient_user_id"] for c in send_mock.await_args_list}
        assert notified == {stale_student}
        assert recent_student not in notified
    finally:
        async with engine.begin() as conn:
            await conn.execute(
                text("DELETE FROM notifications WHERE id = ANY(:ids)"),
                {"ids": notif_ids},
            )
