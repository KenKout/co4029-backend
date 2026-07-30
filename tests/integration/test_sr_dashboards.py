"""Integration tests for SR dashboard routers (T7.5.12).

Covers the spec scenarios:

* ``test_cards_due_returns_paginated`` -- 25 due cards → first page 20 + cursor.
* ``test_cards_due_filtered_by_lesson_id`` -- query filter narrows to lesson.
* ``test_cards_due_other_user_isolation`` -- bearer-derived student id only.
* ``test_sr_summary_composes_metrics`` -- ``StudentLessonSummaryRead`` shape.
* ``test_sr_overview_classifies_status`` -- locked / learning / mature labels.
* ``test_cohort_kr_teacher_only`` -- student token → 403.
* ``test_difficult_cards_returns_top_n_lowest`` -- lowest mean EF first.
* ``test_at_risk_students_returns_composite_signal``.
* ``test_sr_detail_per_student_breakdown``.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch
from uuid import UUID

import httpx
import pytest_asyncio
from alembic import command
from alembic.config import Config
from conftest import SeededUsers
from fastapi import FastAPI
from sqlalchemy import Column, Table, text
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

import abridgeai.features.access_control.models  # noqa: F401  -- register FK targets
import abridgeai.features.identity.models  # noqa: F401  -- register users FK target
import abridgeai.features.interviews.models  # noqa: F401  -- T6.1 registers interview_* tables
from abridgeai.core.config import get_settings
from abridgeai.core.db import Base, get_db
from abridgeai.core.security import create_access_token, generate_token, hash_secret
from abridgeai.features.spaced_repetition.routers import learner_router, teacher_router
from tests.support.db_graph import hard_delete_graph

for _stub_name in ("interview_configs",):
    if _stub_name not in Base.metadata.tables:
        Table(
            _stub_name,
            Base.metadata,
            Column("id", PGUUID(as_uuid=True), primary_key=True),
        )


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
def _isolate_cache() -> AsyncIterator[None]:
    fake_client = AsyncMock()
    fake_client.get = AsyncMock(return_value=None)
    fake_client.set = AsyncMock(return_value=None)
    with (
        patch("abridgeai.core.cache.decorators.get_cache", return_value=fake_client),
        patch(
            "abridgeai.features.spaced_repetition.routers.learner.get_cache",
            return_value=fake_client,
        ),
    ):
        yield


@pytest_asyncio.fixture
async def app(
    session_factory: async_sessionmaker[AsyncSession],
) -> AsyncIterator[FastAPI]:
    async def _override_get_db() -> AsyncIterator[AsyncSession]:
        async with session_factory() as session:
            yield session

    fastapi_app = FastAPI()
    fastapi_app.include_router(learner_router, prefix="/api/v1")
    fastapi_app.include_router(teacher_router, prefix="/api/v1")
    fastapi_app.dependency_overrides[get_db] = _override_get_db
    yield fastapi_app
    fastapi_app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def client(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as ac:
        yield ac


async def _seed_session(engine: AsyncEngine, user_id: uuid.UUID) -> uuid.UUID:
    session_id = uuid.uuid4()
    expires_at = datetime.now(tz=UTC) + timedelta(hours=1)
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO auth_sessions (id, user_id, refresh_token_hash, expires_at) "
                "VALUES (:id, :uid, :h, :exp)"
            ),
            {
                "id": session_id,
                "uid": user_id,
                "h": hash_secret(generate_token()),
                "exp": expires_at,
            },
        )
    return session_id


@pytest_asyncio.fixture
async def teacher_bearer(engine: AsyncEngine, seeded_users: SeededUsers) -> AsyncIterator[str]:
    sid = await _seed_session(engine, seeded_users.teacher_id)
    yield create_access_token(user_id=seeded_users.teacher_id, session_id=sid)
    async with engine.begin() as conn:
        await conn.execute(text("DELETE FROM auth_sessions WHERE id = :id"), {"id": sid})


@pytest_asyncio.fixture
async def student_bearer(engine: AsyncEngine, seeded_users: SeededUsers) -> AsyncIterator[str]:
    sid = await _seed_session(engine, seeded_users.student_id)
    yield create_access_token(user_id=seeded_users.student_id, session_id=sid)
    async with engine.begin() as conn:
        await conn.execute(text("DELETE FROM auth_sessions WHERE id = :id"), {"id": sid})


@pytest_asyncio.fixture
async def admin_bearer(engine: AsyncEngine, seeded_users: SeededUsers) -> AsyncIterator[str]:
    sid = await _seed_session(engine, seeded_users.admin_id)
    yield create_access_token(user_id=seeded_users.admin_id, session_id=sid)
    async with engine.begin() as conn:
        await conn.execute(text("DELETE FROM auth_sessions WHERE id = :id"), {"id": sid})


async def _seed_lesson(
    engine: AsyncEngine,
    *,
    course_id: UUID,
    n_questions: int,
    module_position: int | None = None,
    lesson_title: str = "Lesson",
    ef_min_unlock: Decimal = Decimal("2.0"),
    tau_unlock: Decimal = Decimal("0.8"),
) -> tuple[UUID, UUID, UUID, list[UUID]]:
    module_id = uuid.uuid4()
    lesson_id = uuid.uuid4()
    quiz_id = uuid.uuid4()
    qids = [uuid.uuid4() for _ in range(n_questions)]
    async with engine.begin() as conn:
        if module_position is None:
            next_pos_row = (
                await conn.execute(
                    text(
                        "SELECT COALESCE(MAX(position), 0) + 1 FROM modules "
                        "WHERE course_id = :course"
                    ),
                    {"course": course_id},
                )
            ).one()
            position = int(next_pos_row[0])
        else:
            position = module_position
        await conn.execute(
            text(
                "INSERT INTO modules (id, course_id, title, position, status) "
                "VALUES (:id, :course, :title, :pos, 'published')"
            ),
            {
                "id": module_id,
                "course": course_id,
                "title": f"Mod-{module_id.hex[:6]}",
                "pos": position,
            },
        )
        await conn.execute(
            text(
                "INSERT INTO lessons "
                "(id, module_id, slug, title, status, ef_min_unlock, tau_unlock) "
                "VALUES (:id, :m, :slug, :title, 'published', :ef, :tau)"
            ),
            {
                "id": lesson_id,
                "m": module_id,
                "slug": f"l-{lesson_id.hex[:8]}",
                "title": lesson_title,
                "ef": ef_min_unlock,
                "tau": tau_unlock,
            },
        )
        await conn.execute(
            text(
                "INSERT INTO quizzes (id, course_id, module_id, title, status) "
                "VALUES (:id, :course, :m, :title, 'published')"
            ),
            {"id": quiz_id, "course": course_id, "m": module_id, "title": "Quiz"},
        )
        await conn.execute(
            text("INSERT INTO quiz_source_lessons (quiz_id, lesson_id) VALUES (:q, :l)"),
            {"q": quiz_id, "l": lesson_id},
        )
        for idx, qid in enumerate(qids, start=1):
            await conn.execute(
                text(
                    "INSERT INTO quiz_questions ("
                    "id, quiz_id, position, question_type, prompt_text, "
                    "expected_response_time_ms, source_refs, review_status) VALUES ("
                    ":id, :q, :pos, 'multiple_choice', 'Q?', 30000, '[]'::jsonb, 'approved')"
                ),
                {"id": qid, "q": quiz_id, "pos": idx},
            )
    return module_id, lesson_id, quiz_id, qids


async def _set_card_state(
    engine: AsyncEngine,
    *,
    student_id: UUID,
    question_id: UUID,
    ef: Decimal,
    due_at: datetime | None,
    last_q: int | None = None,
) -> None:
    effective_due_at = due_at if due_at is not None else datetime.now(tz=UTC) + timedelta(days=365)
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO student_card_state ("
                "student_id, question_id, ef, interval_days, repetition_count, "
                "due_at, total_reviews, last_q) VALUES ("
                ":sid, :qid, :ef, 1, 1, :due, 1, :lq) "
                "ON CONFLICT (student_id, question_id) DO UPDATE "
                "SET ef = EXCLUDED.ef, due_at = EXCLUDED.due_at, last_q = EXCLUDED.last_q"
            ),
            {
                "sid": student_id,
                "qid": question_id,
                "ef": ef,
                "due": effective_due_at,
                "lq": last_q,
            },
        )


async def _enroll(engine: AsyncEngine, *, course_id: UUID, student_id: UUID) -> None:
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO course_enrollments (id, course_id, student_id, status) "
                "VALUES (:id, :course, :student, 'active') ON CONFLICT DO NOTHING"
            ),
            {"id": uuid.uuid4(), "course": course_id, "student": student_id},
        )


async def _add_card_review(
    engine: AsyncEngine,
    *,
    student_id: UUID,
    question_id: UUID,
    created_at: datetime,
    correct: bool = True,
    q: int = 4,
    ef_after: Decimal = Decimal("2.5"),
) -> None:
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO card_reviews ("
                "id, student_id, question_id, created_at, "
                "t_actual_ms, t_exp_ms, rho, correct, hint_used, q_derived, "
                "ef_before, ef_after, interval_before, interval_after, "
                "n_before, n_after) VALUES ("
                ":id, :sid, :qid, :ts, "
                "20000, 30000, 0.6667, :ok, FALSE, :q, "
                "2.4, :ef, 1, 1, 0, 1)"
            ),
            {
                "id": uuid.uuid4(),
                "sid": student_id,
                "qid": question_id,
                "ts": created_at,
                "ok": correct,
                "q": q,
                "ef": ef_after,
            },
        )


@pytest_asyncio.fixture
async def cards_due_scenario(
    engine: AsyncEngine, seeded_users: SeededUsers
) -> AsyncIterator[dict[str, Any]]:
    """25 due cards across two lessons in the seeded test_course."""
    course_id = seeded_users.course_id
    student_id = seeded_users.student_id
    await _enroll(engine, course_id=course_id, student_id=student_id)
    _, lesson_a, _, qa = await _seed_lesson(
        engine, course_id=course_id, n_questions=15, lesson_title="Lesson A"
    )
    _, lesson_b, _, qb = await _seed_lesson(
        engine,
        course_id=course_id,
        n_questions=10,
        lesson_title="Lesson B",
    )
    past = datetime.now(tz=UTC) - timedelta(hours=1)
    for qid in qa + qb:
        await _set_card_state(
            engine,
            student_id=student_id,
            question_id=qid,
            ef=Decimal("2.5"),
            due_at=past,
            last_q=4,
        )
    try:
        yield {
            "course_id": course_id,
            "student_id": student_id,
            "lesson_a": lesson_a,
            "lesson_b": lesson_b,
            "qa": qa,
            "qb": qb,
        }
    finally:
        async with engine.begin() as conn:
            await conn.execute(
                text("DELETE FROM card_reviews WHERE student_id = :s"),
                {"s": student_id},
            )
            await conn.execute(
                text("DELETE FROM student_card_state WHERE student_id = :s"),
                {"s": student_id},
            )
            # Graph-driven purge: hand-rolled DELETE chains here kept missing
            # sub-trees other files hang off the shared course (quiz attempts,
            # grades) and errored every later test's setup with FK violations.
            quiz_ids = [
                str(v)
                for v in (
                    await conn.execute(
                        text("SELECT id FROM quizzes WHERE course_id = :c"),
                        {"c": course_id},
                    )
                ).scalars()
            ]
            if quiz_ids:
                await hard_delete_graph(conn, "quizzes", quiz_ids)
            module_ids = [
                str(v)
                for v in (
                    await conn.execute(
                        text("SELECT id FROM modules WHERE course_id = :c"),
                        {"c": course_id},
                    )
                ).scalars()
            ]
            if module_ids:
                await hard_delete_graph(conn, "modules", module_ids)
            await conn.execute(
                text("DELETE FROM course_enrollments WHERE course_id = :c"),
                {"c": course_id},
            )


def test_router_metadata() -> None:
    learner_paths = {(r.path, tuple(sorted(r.methods))) for r in learner_router.routes}  # type: ignore[attr-defined]
    assert ("/me/cards-due", ("GET",)) in learner_paths
    assert ("/me/lessons/{lesson_id}/sr-summary", ("GET",)) in learner_paths
    assert ("/me/courses/{course_id}/sr-overview", ("GET",)) in learner_paths
    teacher_paths = {(r.path, tuple(sorted(r.methods))) for r in teacher_router.routes}  # type: ignore[attr-defined]
    assert ("/teacher/courses/{course_id}/lessons/{lesson_id}/cohort-kr", ("GET",)) in teacher_paths
    assert (
        "/teacher/courses/{course_id}/lessons/{lesson_id}/difficult-cards",
        ("GET",),
    ) in teacher_paths
    assert ("/teacher/courses/{course_id}/at-risk", ("GET",)) in teacher_paths
    assert (
        "/teacher/courses/{course_id}/students/{student_id}/sr-detail",
        ("GET",),
    ) in teacher_paths


async def test_cards_due_returns_paginated(
    client: httpx.AsyncClient,
    student_bearer: str,
    cards_due_scenario: dict[str, Any],
) -> None:
    headers = {"Authorization": f"Bearer {student_bearer}"}
    first = await client.get("/api/v1/me/cards-due?limit=20", headers=headers)
    assert first.status_code == 200, first.text
    body1 = first.json()
    assert len(body1["items"]) == 20
    assert body1["next_cursor"] is not None

    second = await client.get(
        f"/api/v1/me/cards-due?limit=20&cursor={body1['next_cursor']}",
        headers=headers,
    )
    assert second.status_code == 200
    body2 = second.json()
    assert len(body2["items"]) == 5
    assert body2["next_cursor"] is None

    seen_qids = {item["question_id"] for item in body1["items"] + body2["items"]}
    expected_qids = {str(q) for q in cards_due_scenario["qa"] + cards_due_scenario["qb"]}
    assert seen_qids == expected_qids


async def test_dashboard_summary_agrees_with_cards_due(
    client: httpx.AsyncClient,
    student_bearer: str,
    cards_due_scenario: dict[str, Any],
) -> None:
    """The dashboard tile and the cards-due page must report the same number.

    These previously came from different queries with different predicates, which
    is how the notification ended up disagreeing with the page (a forward 1-hour
    window vs due_at <= NOW()). Asserting equality here pins them together.
    """
    del cards_due_scenario
    headers = {"Authorization": f"Bearer {student_bearer}"}

    summary = await client.get("/api/v1/me/sr-dashboard-summary", headers=headers)
    assert summary.status_code == 200, summary.text
    body = summary.json()

    # limit is capped at 100 by the endpoint; the scenario seeds 25 cards.
    page = await client.get("/api/v1/me/cards-due?limit=100", headers=headers)
    assert page.status_code == 200, page.text
    page_count = len(page.json()["items"])

    assert body["cards_due_now"] == page_count, (
        "dashboard cards_due_now must match the cards-due page exactly"
    )
    assert body["cards_due_now"] > 0, "scenario should seed due cards"


async def test_dashboard_summary_reports_retention_and_unlock(
    client: httpx.AsyncClient,
    student_bearer: str,
    cards_due_scenario: dict[str, Any],
) -> None:
    """Shape check on the thesis metrics the student dashboard surfaces."""
    del cards_due_scenario
    headers = {"Authorization": f"Bearer {student_bearer}"}
    response = await client.get("/api/v1/me/sr-dashboard-summary", headers=headers)
    assert response.status_code == 200, response.text
    body = response.json()

    # R-hat is a probability.
    assert 0.0 <= body["avg_kr_estimate"] <= 1.0
    # has_retention_data distinguishes "no cards tracked yet" from a real 0%, so
    # the UI can show an em-dash rather than alarming the student with 0%.
    assert isinstance(body["has_retention_data"], bool)
    assert body["has_retention_data"] is True, "scenario seeds cards, so there is data"

    # Lesson buckets are consistent with the total.
    assert (
        body["lessons_mature"] + body["lessons_learning"] + body["lessons_locked"]
        == body["lessons_total"]
    )
    # Unlock progress is a percentage.
    assert 0.0 <= body["next_unlock_progress_pct"] <= 100.0
    if body["next_unlock_lesson_title"] is None:
        assert body["next_unlock_progress_pct"] == 0.0


async def test_dashboard_summary_requires_auth(client: httpx.AsyncClient) -> None:
    response = await client.get("/api/v1/me/sr-dashboard-summary")
    assert response.status_code in (401, 403)


async def test_cards_due_filtered_by_lesson_id(
    client: httpx.AsyncClient,
    student_bearer: str,
    cards_due_scenario: dict[str, Any],
) -> None:
    headers = {"Authorization": f"Bearer {student_bearer}"}
    response = await client.get(
        f"/api/v1/me/cards-due?limit=50&lesson_id={cards_due_scenario['lesson_b']}",
        headers=headers,
    )
    assert response.status_code == 200, response.text
    body = response.json()
    returned_lessons = {item["lesson_id"] for item in body["items"]}
    assert returned_lessons == {str(cards_due_scenario["lesson_b"])}
    assert len(body["items"]) == 10


async def test_cards_due_other_user_isolation(
    client: httpx.AsyncClient,
    engine: AsyncEngine,
    seeded_users: SeededUsers,
    student_bearer: str,
    cards_due_scenario: dict[str, Any],
) -> None:
    """Manager seeds their own due cards; student bearer must NOT see them."""
    manager_id = seeded_users.manager_id
    course_id = cards_due_scenario["course_id"]
    await _enroll(engine, course_id=course_id, student_id=manager_id)
    past = datetime.now(tz=UTC) - timedelta(hours=2)
    for qid in cards_due_scenario["qa"][:3]:
        await _set_card_state(
            engine,
            student_id=manager_id,
            question_id=qid,
            ef=Decimal("2.5"),
            due_at=past,
        )

    headers = {"Authorization": f"Bearer {student_bearer}"}
    response = await client.get("/api/v1/me/cards-due?limit=100", headers=headers)
    assert response.status_code == 200
    body = response.json()
    assert len(body["items"]) == 25
    async with engine.begin() as conn:
        await conn.execute(
            text("DELETE FROM student_card_state WHERE student_id = :s"),
            {"s": manager_id},
        )
        await conn.execute(
            text("DELETE FROM course_enrollments WHERE student_id = :s"),
            {"s": manager_id},
        )


async def test_sr_summary_composes_metrics(
    client: httpx.AsyncClient,
    engine: AsyncEngine,
    seeded_users: SeededUsers,
    student_bearer: str,
) -> None:
    student_id = seeded_users.student_id
    course_id = seeded_users.course_id
    await _enroll(engine, course_id=course_id, student_id=student_id)
    _, lesson_id, _, qids = await _seed_lesson(engine, course_id=course_id, n_questions=4)
    for qid in qids:
        await _set_card_state(
            engine,
            student_id=student_id,
            question_id=qid,
            ef=Decimal("2.5"),
            due_at=datetime.now(tz=UTC) - timedelta(minutes=5),
        )
    headers = {"Authorization": f"Bearer {student_bearer}"}
    response = await client.get(f"/api/v1/me/lessons/{lesson_id}/sr-summary", headers=headers)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["kr_estimate"] == 1.0
    assert body["progression_ready"] is True
    assert body["cards_total"] == 4
    assert body["cards_due_now"] == 4
    async with engine.begin() as conn:
        await conn.execute(
            text("DELETE FROM student_card_state WHERE student_id = :s"),
            {"s": student_id},
        )
        await conn.execute(
            text(
                "DELETE FROM quiz_questions WHERE quiz_id IN ("
                "SELECT id FROM quizzes WHERE course_id = :c)"
            ),
            {"c": course_id},
        )
        await conn.execute(
            text(
                "DELETE FROM quiz_source_lessons WHERE quiz_id IN ("
                "SELECT id FROM quizzes WHERE course_id = :c)"
            ),
            {"c": course_id},
        )
        await conn.execute(text("DELETE FROM quizzes WHERE course_id = :c"), {"c": course_id})
        await conn.execute(
            text(
                "DELETE FROM lessons WHERE module_id IN ("
                "SELECT id FROM modules WHERE course_id = :c)"
            ),
            {"c": course_id},
        )
        await conn.execute(text("DELETE FROM modules WHERE course_id = :c"), {"c": course_id})
        await conn.execute(
            text("DELETE FROM course_enrollments WHERE course_id = :c AND student_id = :s"),
            {"c": course_id, "s": student_id},
        )


async def test_sr_overview_classifies_status(
    client: httpx.AsyncClient,
    engine: AsyncEngine,
    seeded_users: SeededUsers,
    student_bearer: str,
) -> None:
    student_id = seeded_users.student_id
    course_id = seeded_users.course_id
    await _enroll(engine, course_id=course_id, student_id=student_id)

    _, locked_lesson, _, locked_qids = await _seed_lesson(
        engine,
        course_id=course_id,
        n_questions=3,
        lesson_title="Locked",
    )
    for qid in locked_qids:
        await _set_card_state(
            engine,
            student_id=student_id,
            question_id=qid,
            ef=Decimal("1.3"),
            due_at=None,
        )

    _, learning_lesson, _, learning_qids = await _seed_lesson(
        engine,
        course_id=course_id,
        n_questions=4,
        lesson_title="Learning",
        ef_min_unlock=Decimal("1.3"),
        tau_unlock=Decimal("0.5"),
    )
    for qid in learning_qids[:2]:
        await _set_card_state(
            engine,
            student_id=student_id,
            question_id=qid,
            ef=Decimal("1.9"),
            due_at=None,
        )
    for qid in learning_qids[2:]:
        await _set_card_state(
            engine,
            student_id=student_id,
            question_id=qid,
            ef=Decimal("1.9"),
            due_at=None,
        )

    _, mature_lesson, _, mature_qids = await _seed_lesson(
        engine,
        course_id=course_id,
        n_questions=2,
        lesson_title="Mature",
        ef_min_unlock=Decimal("1.3"),
        tau_unlock=Decimal("0.5"),
    )
    for qid in mature_qids:
        await _set_card_state(
            engine,
            student_id=student_id,
            question_id=qid,
            ef=Decimal("2.5"),
            due_at=None,
        )

    headers = {"Authorization": f"Bearer {student_bearer}"}
    response = await client.get(f"/api/v1/me/courses/{course_id}/sr-overview", headers=headers)
    assert response.status_code == 200, response.text
    body = response.json()
    by_id = {item["lesson_id"]: item for item in body}
    assert by_id[str(locked_lesson)]["status"] == "locked"
    assert by_id[str(learning_lesson)]["status"] == "learning"
    assert by_id[str(mature_lesson)]["status"] == "mature"
    assert by_id[str(mature_lesson)]["kr_estimate"] == 1.0

    async with engine.begin() as conn:
        await conn.execute(
            text("DELETE FROM student_card_state WHERE student_id = :s"), {"s": student_id}
        )
        await conn.execute(
            text(
                "DELETE FROM quiz_questions WHERE quiz_id IN ("
                "SELECT id FROM quizzes WHERE course_id = :c)"
            ),
            {"c": course_id},
        )
        await conn.execute(
            text(
                "DELETE FROM quiz_source_lessons WHERE quiz_id IN ("
                "SELECT id FROM quizzes WHERE course_id = :c)"
            ),
            {"c": course_id},
        )
        await conn.execute(text("DELETE FROM quizzes WHERE course_id = :c"), {"c": course_id})
        await conn.execute(
            text(
                "DELETE FROM lessons WHERE module_id IN ("
                "SELECT id FROM modules WHERE course_id = :c)"
            ),
            {"c": course_id},
        )
        await conn.execute(text("DELETE FROM modules WHERE course_id = :c"), {"c": course_id})
        await conn.execute(
            text("DELETE FROM course_enrollments WHERE course_id = :c AND student_id = :s"),
            {"c": course_id, "s": student_id},
        )


async def test_cohort_kr_teacher_only(
    client: httpx.AsyncClient,
    engine: AsyncEngine,
    seeded_users: SeededUsers,
    student_bearer: str,
    teacher_bearer: str,
) -> None:
    course_id = seeded_users.course_id
    _, lesson_id, _, qids = await _seed_lesson(engine, course_id=course_id, n_questions=2)
    await _enroll(engine, course_id=course_id, student_id=seeded_users.student_id)
    for qid in qids:
        await _set_card_state(
            engine,
            student_id=seeded_users.student_id,
            question_id=qid,
            ef=Decimal("2.0"),
            due_at=None,
        )

    student_resp = await client.get(
        f"/api/v1/teacher/courses/{course_id}/lessons/{lesson_id}/cohort-kr",
        headers={"Authorization": f"Bearer {student_bearer}"},
    )
    assert student_resp.status_code == 403, student_resp.text

    teacher_resp = await client.get(
        f"/api/v1/teacher/courses/{course_id}/lessons/{lesson_id}/cohort-kr",
        headers={"Authorization": f"Bearer {teacher_bearer}"},
    )
    assert teacher_resp.status_code == 200, teacher_resp.text
    body = teacher_resp.json()
    assert body["lesson_id"] == str(lesson_id)
    assert body["student_count"] >= 1

    async with engine.begin() as conn:
        await conn.execute(
            text("DELETE FROM student_card_state WHERE student_id = :s"),
            {"s": seeded_users.student_id},
        )
        await conn.execute(
            text(
                "DELETE FROM quiz_questions WHERE quiz_id IN ("
                "SELECT id FROM quizzes WHERE course_id = :c)"
            ),
            {"c": course_id},
        )
        await conn.execute(
            text(
                "DELETE FROM quiz_source_lessons WHERE quiz_id IN ("
                "SELECT id FROM quizzes WHERE course_id = :c)"
            ),
            {"c": course_id},
        )
        await conn.execute(text("DELETE FROM quizzes WHERE course_id = :c"), {"c": course_id})
        await conn.execute(
            text(
                "DELETE FROM lessons WHERE module_id IN ("
                "SELECT id FROM modules WHERE course_id = :c)"
            ),
            {"c": course_id},
        )
        await conn.execute(text("DELETE FROM modules WHERE course_id = :c"), {"c": course_id})
        await conn.execute(
            text("DELETE FROM course_enrollments WHERE course_id = :c"),
            {"c": course_id},
        )


async def test_difficult_cards_returns_top_n_lowest(
    client: httpx.AsyncClient,
    engine: AsyncEngine,
    seeded_users: SeededUsers,
    teacher_bearer: str,
) -> None:
    course_id = seeded_users.course_id
    student_id = seeded_users.student_id
    await _enroll(engine, course_id=course_id, student_id=student_id)
    _, lesson_id, _, qids = await _seed_lesson(engine, course_id=course_id, n_questions=5)
    efs = [Decimal("1.4"), Decimal("1.6"), Decimal("1.8"), Decimal("2.0"), Decimal("2.4")]
    for qid, ef in zip(qids, efs, strict=True):
        await _set_card_state(engine, student_id=student_id, question_id=qid, ef=ef, due_at=None)

    response = await client.get(
        f"/api/v1/teacher/courses/{course_id}/lessons/{lesson_id}/difficult-cards?top_n=3",
        headers={"Authorization": f"Bearer {teacher_bearer}"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert len(body) == 3
    returned_efs = [item["mean_ef"] for item in body]
    assert returned_efs == sorted(returned_efs)
    assert returned_efs[0] == 1.4

    async with engine.begin() as conn:
        await conn.execute(
            text("DELETE FROM student_card_state WHERE student_id = :s"), {"s": student_id}
        )
        await conn.execute(
            text(
                "DELETE FROM quiz_questions WHERE quiz_id IN ("
                "SELECT id FROM quizzes WHERE course_id = :c)"
            ),
            {"c": course_id},
        )
        await conn.execute(
            text(
                "DELETE FROM quiz_source_lessons WHERE quiz_id IN ("
                "SELECT id FROM quizzes WHERE course_id = :c)"
            ),
            {"c": course_id},
        )
        await conn.execute(text("DELETE FROM quizzes WHERE course_id = :c"), {"c": course_id})
        await conn.execute(
            text(
                "DELETE FROM lessons WHERE module_id IN ("
                "SELECT id FROM modules WHERE course_id = :c)"
            ),
            {"c": course_id},
        )
        await conn.execute(text("DELETE FROM modules WHERE course_id = :c"), {"c": course_id})
        await conn.execute(
            text("DELETE FROM course_enrollments WHERE course_id = :c"),
            {"c": course_id},
        )


async def test_at_risk_students_returns_composite_signal(
    client: httpx.AsyncClient,
    engine: AsyncEngine,
    seeded_users: SeededUsers,
    teacher_bearer: str,
) -> None:
    course_id = seeded_users.course_id
    student_id = seeded_users.student_id
    await _enroll(engine, course_id=course_id, student_id=student_id)
    _, _lesson_id, _, qids = await _seed_lesson(engine, course_id=course_id, n_questions=4)
    far_past = datetime.now(tz=UTC) - timedelta(days=10)
    for qid in qids:
        await _set_card_state(
            engine,
            student_id=student_id,
            question_id=qid,
            ef=Decimal("2.0"),
            due_at=far_past,
        )

    response = await client.get(
        f"/api/v1/teacher/courses/{course_id}/at-risk",
        headers={"Authorization": f"Bearer {teacher_bearer}"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert any(row["student_id"] == str(student_id) for row in body)
    flagged = next(row for row in body if row["student_id"] == str(student_id))
    assert flagged["low_compliance"] is True or flagged["frozen_kr"] is True

    async with engine.begin() as conn:
        await conn.execute(
            text("DELETE FROM student_card_state WHERE student_id = :s"), {"s": student_id}
        )
        await conn.execute(
            text(
                "DELETE FROM quiz_questions WHERE quiz_id IN ("
                "SELECT id FROM quizzes WHERE course_id = :c)"
            ),
            {"c": course_id},
        )
        await conn.execute(
            text(
                "DELETE FROM quiz_source_lessons WHERE quiz_id IN ("
                "SELECT id FROM quizzes WHERE course_id = :c)"
            ),
            {"c": course_id},
        )
        await conn.execute(text("DELETE FROM quizzes WHERE course_id = :c"), {"c": course_id})
        await conn.execute(
            text(
                "DELETE FROM lessons WHERE module_id IN ("
                "SELECT id FROM modules WHERE course_id = :c)"
            ),
            {"c": course_id},
        )
        await conn.execute(text("DELETE FROM modules WHERE course_id = :c"), {"c": course_id})
        await conn.execute(
            text("DELETE FROM course_enrollments WHERE course_id = :c"),
            {"c": course_id},
        )


async def test_sr_detail_per_student_breakdown(
    client: httpx.AsyncClient,
    engine: AsyncEngine,
    seeded_users: SeededUsers,
    teacher_bearer: str,
) -> None:
    course_id = seeded_users.course_id
    student_id = seeded_users.student_id
    await _enroll(engine, course_id=course_id, student_id=student_id)
    _, lesson_id, _, qids = await _seed_lesson(engine, course_id=course_id, n_questions=3)
    for qid in qids:
        await _set_card_state(
            engine,
            student_id=student_id,
            question_id=qid,
            ef=Decimal("2.5"),
            due_at=datetime.now(tz=UTC) - timedelta(minutes=10),
        )
        await _add_card_review(
            engine,
            student_id=student_id,
            question_id=qid,
            created_at=datetime.now(tz=UTC) - timedelta(hours=1),
        )

    response = await client.get(
        f"/api/v1/teacher/courses/{course_id}/students/{student_id}/sr-detail",
        headers={"Authorization": f"Bearer {teacher_bearer}"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["student_id"] == str(student_id)
    lesson_ids = {lesson["lesson_id"] for lesson in body["lessons"]}
    assert str(lesson_id) in lesson_ids
    target_lesson = next(
        lesson for lesson in body["lessons"] if lesson["lesson_id"] == str(lesson_id)
    )
    assert target_lesson["cards_total"] == 3
    assert target_lesson["cards_due_now"] == 3
    assert target_lesson["status"] == "mature"
    assert len(body["recent_reviews"]) == 3

    async with engine.begin() as conn:
        await conn.execute(
            text("DELETE FROM card_reviews WHERE student_id = :s"), {"s": student_id}
        )
        await conn.execute(
            text("DELETE FROM student_card_state WHERE student_id = :s"), {"s": student_id}
        )
        await conn.execute(
            text(
                "DELETE FROM quiz_questions WHERE quiz_id IN ("
                "SELECT id FROM quizzes WHERE course_id = :c)"
            ),
            {"c": course_id},
        )
        await conn.execute(
            text(
                "DELETE FROM quiz_source_lessons WHERE quiz_id IN ("
                "SELECT id FROM quizzes WHERE course_id = :c)"
            ),
            {"c": course_id},
        )
        await conn.execute(text("DELETE FROM quizzes WHERE course_id = :c"), {"c": course_id})
        await conn.execute(
            text(
                "DELETE FROM lessons WHERE module_id IN ("
                "SELECT id FROM modules WHERE course_id = :c)"
            ),
            {"c": course_id},
        )
        await conn.execute(text("DELETE FROM modules WHERE course_id = :c"), {"c": course_id})
        await conn.execute(
            text("DELETE FROM course_enrollments WHERE course_id = :c"),
            {"c": course_id},
        )


async def test_unauthenticated_returns_401(client: httpx.AsyncClient) -> None:
    response = await client.get("/api/v1/me/cards-due")
    assert response.status_code == 401
    response = await client.get(
        f"/api/v1/teacher/courses/{uuid.uuid4()}/lessons/{uuid.uuid4()}/cohort-kr"
    )
    assert response.status_code == 401


# ---------------------------------------------------------------------------
# Review loop (resolve a due card without re-taking the whole quiz)
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def review_scenario(
    engine: AsyncEngine, seeded_users: SeededUsers
) -> AsyncIterator[dict[str, Any]]:
    """One overdue MCQ card with a correct option, ready to review."""
    course_id = seeded_users.course_id
    student_id = seeded_users.student_id
    await _enroll(engine, course_id=course_id, student_id=student_id)
    _, lesson_id, quiz_id, qids = await _seed_lesson(
        engine, course_id=course_id, n_questions=1, lesson_title="Review Lesson"
    )
    question_id = qids[0]
    correct_option_id = uuid.uuid4()
    wrong_option_id = uuid.uuid4()
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO quiz_question_options "
                "(id, question_id, option_key, option_text, is_correct, position) VALUES "
                "(:c, :q, 'A', 'Correct', TRUE, 1), (:w, :q, 'B', 'Wrong', FALSE, 2)"
            ),
            {"c": correct_option_id, "w": wrong_option_id, "q": question_id},
        )
    await _set_card_state(
        engine,
        student_id=student_id,
        question_id=question_id,
        ef=Decimal("2.5"),
        due_at=datetime.now(tz=UTC) - timedelta(hours=1),
        last_q=0,
    )
    try:
        yield {
            "student_id": student_id,
            "course_id": course_id,
            "lesson_id": lesson_id,
            "quiz_id": quiz_id,
            "question_id": question_id,
            "correct_option_id": correct_option_id,
            "wrong_option_id": wrong_option_id,
        }
    finally:
        async with engine.begin() as conn:
            await conn.execute(
                text("DELETE FROM card_reviews WHERE student_id = :s"), {"s": student_id}
            )
            await conn.execute(
                text("DELETE FROM student_card_state WHERE student_id = :s"), {"s": student_id}
            )
            quiz_ids = [
                str(v)
                for v in (
                    await conn.execute(
                        text("SELECT id FROM quizzes WHERE course_id = :c"), {"c": course_id}
                    )
                ).scalars()
            ]
            if quiz_ids:
                await hard_delete_graph(conn, "quizzes", quiz_ids)
            module_ids = [
                str(v)
                for v in (
                    await conn.execute(
                        text("SELECT id FROM modules WHERE course_id = :c"), {"c": course_id}
                    )
                ).scalars()
            ]
            if module_ids:
                await hard_delete_graph(conn, "modules", module_ids)
            await conn.execute(
                text("DELETE FROM course_enrollments WHERE course_id = :c"), {"c": course_id}
            )


async def test_review_queue_serves_no_leak_payload(
    client: httpx.AsyncClient,
    student_bearer: str,
    review_scenario: dict[str, Any],
) -> None:
    """The queue returns the card + its question payload without leaking answers."""
    headers = {"Authorization": f"Bearer {student_bearer}"}
    resp = await client.get("/api/v1/me/review/queue", headers=headers)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["total_due"] >= 1
    card = next(c for c in body["items"] if c["question_id"] == str(review_scenario["question_id"]))
    assert card["course_title"]  # course context present
    assert card["lesson_title"] == "Review Lesson"
    # Question payload embedded; options must NOT carry is_correct.
    options = card["question"]["options"]
    assert len(options) == 2
    for opt in options:
        assert "is_correct" not in opt


async def test_review_submit_correct_advances_and_resolves(
    client: httpx.AsyncClient,
    student_bearer: str,
    review_scenario: dict[str, Any],
) -> None:
    """Answering correctly grades the card, reschedules it out, and clears it."""
    headers = {"Authorization": f"Bearer {student_bearer}"}
    question_id = str(review_scenario["question_id"])

    # Fast correct answer → q>=3 → passing, card advances beyond "now".
    submit = await client.post(
        f"/api/v1/me/review/{question_id}",
        json={
            "selected_option_id": str(review_scenario["correct_option_id"]),
            "hint_used": False,
            "t_actual_ms": 3000,
        },
        headers=headers,
    )
    assert submit.status_code == 200, submit.text
    result = submit.json()
    assert result["correct"] is True
    assert result["passing"] is True
    assert result["q"] >= 3
    assert result["interval_days"] >= 1
    assert result["remaining_due"] == 0  # the only due card is now resolved
    # Feedback surfaces the correct option.
    assert str(review_scenario["correct_option_id"]) in result["correct_option_ids"]

    # The card no longer appears in the due queue.
    queue = await client.get("/api/v1/me/review/queue", headers=headers)
    assert queue.status_code == 200, queue.text
    qids = {c["question_id"] for c in queue.json()["items"]}
    assert question_id not in qids


async def test_review_submit_wrong_resets_but_stays_due(
    client: httpx.AsyncClient,
    student_bearer: str,
    review_scenario: dict[str, Any],
) -> None:
    """A wrong answer grades q=0, resets the card, and pushes it to cooldown."""
    headers = {"Authorization": f"Bearer {student_bearer}"}
    question_id = str(review_scenario["question_id"])
    submit = await client.post(
        f"/api/v1/me/review/{question_id}",
        json={
            "selected_option_id": str(review_scenario["wrong_option_id"]),
            "hint_used": False,
            "t_actual_ms": 5000,
        },
        headers=headers,
    )
    assert submit.status_code == 200, submit.text
    result = submit.json()
    assert result["correct"] is False
    assert result["passing"] is False
    assert result["q"] == 0
    # Card is pushed out by the failure cooldown, so it leaves the "due now" set.
    assert result["remaining_due"] == 0


async def test_review_submit_unknown_question_404(
    client: httpx.AsyncClient,
    student_bearer: str,
    review_scenario: dict[str, Any],
) -> None:
    del review_scenario
    headers = {"Authorization": f"Bearer {student_bearer}"}
    resp = await client.post(
        f"/api/v1/me/review/{uuid.uuid4()}",
        json={"selected_option_id": str(uuid.uuid4())},
        headers=headers,
    )
    assert resp.status_code == 404, resp.text
