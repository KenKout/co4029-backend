"""Integration tests for T7.5.11 — cooldown enforcement on quiz retry.

Per thesis UC-LEARN-01 Alt 1a: when a student fails a card, that card's
``student_card_state.due_at`` is pushed out by the configured failure
cooldown (default 24 h). :func:`taking_service.start_attempt` must
honor this:

* All questions in cooldown → :class:`AllCardsInCooldownError` →
  router returns HTTP 429 with ``Retry-After`` header.
* Some in cooldown → drop them from the take payload, surface the
  list on ``attempt.cards_in_cooldown``.
* None in cooldown (or expired) → existing happy path.
* Unattempted cards (no ``student_card_state`` row) are *never* in
  cooldown — they are new material.

This task only READS the ``due_at`` set by T7.5.5's
``record_card_review``; it does not compute the cooldown duration
itself.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from unittest.mock import AsyncMock

import httpx
import pytest
import pytest_asyncio
from alembic import command
from alembic.config import Config
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
import abridgeai.features.courses.models  # noqa: F401  -- register courses/modules/lessons FK targets
import abridgeai.features.identity.models  # noqa: F401  -- register users FK target
import abridgeai.features.interviews.models  # noqa: F401  -- T6.1 registers interview_* tables
import abridgeai.features.quizzes.models  # noqa: F401  -- register quiz tables
import abridgeai.features.spaced_repetition.models  # noqa: F401  -- register student_card_state
from abridgeai.core.config import get_settings
from abridgeai.core.db import Base, get_db
from abridgeai.core.security import (
    CurrentUser,
    create_access_token,
    generate_token,
    hash_secret,
)
from abridgeai.features.quizzes.routers import learner_router
from abridgeai.features.quizzes.services import taking as taking_service
from abridgeai.features.quizzes.services.taking import AllCardsInCooldownError

for _stub_name in ("interview_configs", "learning_materials", "learning_material_versions"):
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
    yield eng
    await eng.dispose()


@pytest_asyncio.fixture
async def session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False, autoflush=False)


@pytest_asyncio.fixture
async def quiz_with_questions(engine: AsyncEngine) -> AsyncIterator[dict]:
    """Seed an org/course/module/quiz with five questions; clean up after."""
    org_id = uuid.uuid4()
    owner_id = uuid.uuid4()
    student_id = uuid.uuid4()
    course_id = uuid.uuid4()
    module_id = uuid.uuid4()
    quiz_id = uuid.uuid4()
    question_ids = [uuid.uuid4() for _ in range(5)]
    suffix = org_id.hex[:8]

    async with engine.begin() as conn:
        await conn.execute(
            text("INSERT INTO organizations (id, slug, name) VALUES (:id, :slug, :name)"),
            {"id": org_id, "slug": f"qc-{suffix}", "name": "Cooldown Test Org"},
        )
        await conn.execute(
            text("INSERT INTO users (id, primary_email) VALUES (:id, :email)"),
            {"id": owner_id, "email": f"qc-owner-{suffix}@test.local"},
        )
        await conn.execute(
            text("INSERT INTO users (id, primary_email) VALUES (:id, :email)"),
            {"id": student_id, "email": f"qc-student-{suffix}@test.local"},
        )
        await conn.execute(
            text(
                "INSERT INTO courses "
                "(id, organization_id, owner_user_id, slug, title, status) "
                "VALUES (:id, :org, :owner, :slug, 'Cooldown Course', 'published')"
            ),
            {
                "id": course_id,
                "org": org_id,
                "owner": owner_id,
                "slug": f"course-{suffix}",
            },
        )
        await conn.execute(
            text(
                "INSERT INTO modules (id, course_id, title, position, status) "
                "VALUES (:id, :course, 'Module', 1, 'published')"
            ),
            {"id": module_id, "course": course_id},
        )
        await conn.execute(
            text(
                "INSERT INTO quizzes (id, course_id, module_id, title, status, "
                "passing_score_percent) VALUES (:id, :course, :module, "
                "'Cooldown Quiz', 'published', 70.00)"
            ),
            {"id": quiz_id, "course": course_id, "module": module_id},
        )
        for index, qid in enumerate(question_ids, start=1):
            await conn.execute(
                text(
                    "INSERT INTO quiz_questions "
                    "(id, quiz_id, position, question_type, prompt_text, review_status) "
                    "VALUES (:id, :quiz, :pos, 'multiple_choice', :prompt, 'approved')"
                ),
                {"id": qid, "quiz": quiz_id, "pos": index, "prompt": f"Q{index}?"},
            )
            await conn.execute(
                text(
                    "INSERT INTO quiz_question_options "
                    "(id, question_id, option_key, option_text, is_correct, position) "
                    "VALUES (uuid_generate_v4(), :q, 'A', 'right', TRUE, 1), "
                    "       (uuid_generate_v4(), :q, 'B', 'wrong', FALSE, 2)"
                ),
                {"q": qid},
            )

    yield {
        "owner_id": owner_id,
        "student_id": student_id,
        "org_id": org_id,
        "course_id": course_id,
        "module_id": module_id,
        "quiz_id": quiz_id,
        "question_ids": question_ids,
    }

    async with engine.begin() as conn:
        await conn.execute(
            text("DELETE FROM card_reviews WHERE question_id = ANY(CAST(:qids AS uuid[]))"),
            {"qids": [str(q) for q in question_ids]},
        )
        await conn.execute(
            text("DELETE FROM student_card_state WHERE question_id = ANY(CAST(:qids AS uuid[]))"),
            {"qids": [str(q) for q in question_ids]},
        )
        await conn.execute(
            text(
                "DELETE FROM quiz_attempt_answers WHERE attempt_id IN "
                "(SELECT id FROM quiz_attempts WHERE quiz_id = :q)"
            ),
            {"q": quiz_id},
        )
        await conn.execute(
            text("DELETE FROM quiz_attempts WHERE quiz_id = :q"),
            {"q": quiz_id},
        )
        await conn.execute(
            text(
                "DELETE FROM quiz_question_options WHERE question_id = ANY(CAST(:qids AS uuid[]))"
            ),
            {"qids": [str(q) for q in question_ids]},
        )
        await conn.execute(
            text("DELETE FROM quiz_questions WHERE quiz_id = :q"),
            {"q": quiz_id},
        )
        await conn.execute(text("DELETE FROM quizzes WHERE id = :id"), {"id": quiz_id})
        await conn.execute(text("DELETE FROM modules WHERE course_id = :c"), {"c": course_id})
        await conn.execute(text("DELETE FROM courses WHERE id = :id"), {"id": course_id})
        await conn.execute(
            text("DELETE FROM users WHERE id IN (:o, :s)"),
            {"o": owner_id, "s": student_id},
        )
        await conn.execute(text("DELETE FROM organizations WHERE id = :id"), {"id": org_id})


def _actor(user_id: uuid.UUID) -> CurrentUser:
    return CurrentUser(user_id=user_id, session_id=uuid.uuid4())


async def _seed_card_state(
    engine: AsyncEngine,
    student_id: uuid.UUID,
    question_id: uuid.UUID,
    due_at: datetime,
) -> None:
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO student_card_state ("
                "  student_id, question_id, ef, interval_days, repetition_count, "
                "  due_at, total_reviews"
                ") VALUES (:sid, :qid, :ef, 1, 1, :due, 1) "
                "ON CONFLICT (student_id, question_id) DO UPDATE "
                "SET due_at = EXCLUDED.due_at"
            ),
            {"sid": student_id, "qid": question_id, "ef": Decimal("2.5"), "due": due_at},
        )


@pytest.mark.asyncio
async def test_no_cooldown_all_questions_returned(
    session_factory: async_sessionmaker[AsyncSession],
    quiz_with_questions: dict,
) -> None:
    """Student with no card_state rows: every question is returned."""
    async with session_factory() as session, session.begin():
        attempt, payload = await taking_service.start_attempt(
            session,
            quiz_with_questions["quiz_id"],
            _actor(quiz_with_questions["student_id"]),
        )
        await session.flush()

    assert attempt.attempt_number == 1
    assert len(payload.questions) == 5
    assert getattr(attempt, "cards_in_cooldown", []) == []


@pytest.mark.asyncio
async def test_some_in_cooldown_filtered_out(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
    quiz_with_questions: dict,
) -> None:
    """Two of five questions in cooldown: payload has three, attempt
    surfaces the two in ``cards_in_cooldown``."""
    student_id = quiz_with_questions["student_id"]
    question_ids = quiz_with_questions["question_ids"]
    future = datetime.now(tz=UTC) + timedelta(hours=12)
    await _seed_card_state(engine, student_id, question_ids[0], future)
    await _seed_card_state(engine, student_id, question_ids[1], future)

    async with session_factory() as session, session.begin():
        attempt, payload = await taking_service.start_attempt(
            session, quiz_with_questions["quiz_id"], _actor(student_id)
        )
        await session.flush()

    returned_ids = {q.id for q in payload.questions}
    assert len(returned_ids) == 3
    assert question_ids[0] not in returned_ids
    assert question_ids[1] not in returned_ids

    cooldown = getattr(attempt, "cards_in_cooldown", [])
    assert len(cooldown) == 2
    cooldown_qids = {entry["question_id"] for entry in cooldown}
    assert cooldown_qids == {question_ids[0], question_ids[1]}


@pytest.mark.asyncio
async def test_all_in_cooldown_raises(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
    quiz_with_questions: dict,
) -> None:
    """Every question in cooldown: service raises
    :class:`AllCardsInCooldownError` carrying the earliest due_at."""
    student_id = quiz_with_questions["student_id"]
    question_ids = quiz_with_questions["question_ids"]
    base = datetime.now(tz=UTC) + timedelta(hours=1)
    for index, qid in enumerate(question_ids):
        await _seed_card_state(engine, student_id, qid, base + timedelta(hours=index))

    with pytest.raises(AllCardsInCooldownError) as excinfo:
        async with session_factory() as session, session.begin():
            await taking_service.start_attempt(
                session, quiz_with_questions["quiz_id"], _actor(student_id)
            )

    err = excinfo.value
    assert len(err.cards_due_at) == 5
    earliest = min(due for _, due in err.cards_due_at)
    assert err.retry_available_at == earliest


@pytest.mark.asyncio
async def test_expired_cooldown_card_returns(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
    quiz_with_questions: dict,
) -> None:
    """A card whose ``due_at`` is in the past is no longer in cooldown."""
    student_id = quiz_with_questions["student_id"]
    question_ids = quiz_with_questions["question_ids"]
    past = datetime.now(tz=UTC) - timedelta(hours=1)
    await _seed_card_state(engine, student_id, question_ids[0], past)

    async with session_factory() as session, session.begin():
        _, payload = await taking_service.start_attempt(
            session, quiz_with_questions["quiz_id"], _actor(student_id)
        )
        await session.flush()

    returned_ids = {q.id for q in payload.questions}
    assert returned_ids == set(question_ids)


@pytest.mark.asyncio
async def test_unattempted_cards_no_cooldown(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
    quiz_with_questions: dict,
) -> None:
    """Cards the student never attempted (no card_state row) are
    treated as new material — never in cooldown."""
    student_id = quiz_with_questions["student_id"]
    question_ids = quiz_with_questions["question_ids"]
    future = datetime.now(tz=UTC) + timedelta(hours=12)
    await _seed_card_state(engine, student_id, question_ids[0], future)

    async with session_factory() as session, session.begin():
        _, payload = await taking_service.start_attempt(
            session, quiz_with_questions["quiz_id"], _actor(student_id)
        )
        await session.flush()

    returned_ids = {q.id for q in payload.questions}
    assert question_ids[0] not in returned_ids
    for unattempted_qid in question_ids[1:]:
        assert unattempted_qid in returned_ids


@pytest_asyncio.fixture
async def app(
    session_factory: async_sessionmaker[AsyncSession],
) -> AsyncIterator[FastAPI]:
    arq_pool = AsyncMock()
    arq_pool.enqueue_job = AsyncMock()

    async def _override_get_db() -> AsyncIterator[AsyncSession]:
        async with session_factory() as session:
            yield session

    fastapi_app = FastAPI()
    fastapi_app.include_router(learner_router, prefix="/api/v1")
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


@pytest.mark.asyncio
async def test_all_in_cooldown_returns_429(
    engine: AsyncEngine,
    client: httpx.AsyncClient,
    quiz_with_questions: dict,
) -> None:
    """End-to-end: every card in cooldown → HTTP 429 with body +
    ``Retry-After`` header."""
    student_id = quiz_with_questions["student_id"]
    quiz_id = quiz_with_questions["quiz_id"]
    question_ids = quiz_with_questions["question_ids"]
    base = datetime.now(tz=UTC) + timedelta(hours=2)
    for index, qid in enumerate(question_ids):
        await _seed_card_state(engine, student_id, qid, base + timedelta(hours=index))

    sid = await _seed_session(engine, student_id)
    token = create_access_token(user_id=student_id, session_id=sid)

    try:
        resp = await client.post(
            f"/api/v1/quizzes/{quiz_id}/attempts",
            json={"quiz_id": str(quiz_id)},
            headers={"Authorization": f"Bearer {token}"},
        )
    finally:
        async with engine.begin() as conn:
            await conn.execute(text("DELETE FROM auth_sessions WHERE id = :id"), {"id": sid})

    assert resp.status_code == 429, resp.text
    assert "Retry-After" in resp.headers
    assert int(resp.headers["Retry-After"]) > 0
    body = json.loads(resp.text)
    detail = body["detail"]
    assert detail["reason"] == "all_cards_in_cooldown"
    assert detail["retry_available_at"]
    assert len(detail["cards_due_at"]) == 5
