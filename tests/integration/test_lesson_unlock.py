"""Integration tests for the lesson unlock gate (T7.5.6).

Covers plan §7.5.6 + thesis §5.x + §3.2:

* All-pass and partial-fail EF gate scenarios.
* Empty lesson (0 cards) bypasses the EF gate.
* Recursive prereq chain (A → B → C); a failed leaf blocks the root.
* Interview gate: required-but-missing vs required-and-passed.
* Cycle detection in the prereq graph (cycle → eligible + WARN log).
* Cache speed-up: warm call latency < 0.2 × cold call latency.
"""

from __future__ import annotations

import os
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Any
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

from abridgeai.core.cache.client import reset_cache_client_for_tests
from abridgeai.core.config import get_settings
from abridgeai.features.identity import models as _identity_models  # noqa: F401
from abridgeai.features.quizzes import models as _quiz_models  # noqa: F401
from abridgeai.features.spaced_repetition.sm2 import (
    LessonUnlockStatus,
    check_lesson_unlock,
)

TEST_REDIS_URL = os.environ.get("TEST_REDIS_URL", "redis://localhost:6380/15")


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


@pytest.fixture(autouse=True)
def _redis_url(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setenv("REDIS_URL", TEST_REDIS_URL)
    from abridgeai.core import config as config_mod

    config_mod.get_settings.cache_clear()
    reset_cache_client_for_tests()
    yield
    reset_cache_client_for_tests()
    config_mod.get_settings.cache_clear()


@pytest_asyncio.fixture
async def engine() -> AsyncIterator[AsyncEngine]:
    _ensure_head()
    eng = create_async_engine(_async_url(get_settings().database_url), pool_pre_ping=True)
    try:
        yield eng
    finally:
        await eng.dispose()


@pytest_asyncio.fixture
async def session_factory(
    engine: AsyncEngine,
) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False, autoflush=False)


@pytest_asyncio.fixture
async def redis_clean() -> AsyncIterator[Any]:
    from abridgeai.core.cache import get_cache

    client = get_cache()
    with suppress(Exception):
        await client.flushdb()
    yield client
    with suppress(Exception):
        await client.flushdb()


@pytest.fixture
def cache_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force Redis to be unreachable so the cache decorator falls through."""
    monkeypatch.setenv("REDIS_URL", "redis://127.0.0.1:1/0")
    from abridgeai.core import config as config_mod

    config_mod.get_settings.cache_clear()
    reset_cache_client_for_tests()


@asynccontextmanager
async def _conn(engine: AsyncEngine):
    async with engine.begin() as conn:
        yield conn


async def _seed_org_user_course_module(engine: AsyncEngine) -> dict[str, UUID]:
    org_id = uuid.uuid4()
    student_id = uuid.uuid4()
    course_id = uuid.uuid4()
    module_id = uuid.uuid4()

    async with _conn(engine) as conn:
        await conn.execute(
            text("INSERT INTO organizations (id, slug, name) VALUES (:id, :slug, :name)"),
            {
                "id": org_id,
                "slug": f"unlock-{org_id.hex[:8]}",
                "name": "T7.5.6 Org",
            },
        )
        await conn.execute(
            text("INSERT INTO users (id, primary_email) VALUES (:id, :email)"),
            {
                "id": student_id,
                "email": f"unlock-{student_id.hex[:8]}@test.local",
            },
        )
        await conn.execute(
            text(
                "INSERT INTO courses "
                "(id, organization_id, owner_user_id, slug, title) "
                "VALUES (:id, :org, :owner, :slug, 'T7.5.6 Course')"
            ),
            {
                "id": course_id,
                "org": org_id,
                "owner": student_id,
                "slug": f"course-{course_id.hex[:8]}",
            },
        )
        await conn.execute(
            text(
                "INSERT INTO modules (id, course_id, title, position) VALUES (:id, :course, 'M', 1)"
            ),
            {"id": module_id, "course": course_id},
        )
    return {
        "org_id": org_id,
        "student_id": student_id,
        "course_id": course_id,
        "module_id": module_id,
    }


async def _seed_lesson(
    engine: AsyncEngine,
    *,
    module_id: UUID,
    ef_min_unlock: float = 2.0,
    tau_unlock: float = 0.8,
    requires_interview_pass: bool = False,
    slug_suffix: str | None = None,
) -> UUID:
    lesson_id = uuid.uuid4()
    slug = f"lesson-{slug_suffix or lesson_id.hex[:8]}"
    async with _conn(engine) as conn:
        await conn.execute(
            text(
                "INSERT INTO lessons "
                "(id, module_id, slug, title, status, "
                "ef_min_unlock, tau_unlock, requires_interview_pass) "
                "VALUES (:id, :m, :slug, :title, 'published', "
                ":ef, :tau, :req)"
            ),
            {
                "id": lesson_id,
                "m": module_id,
                "slug": slug,
                "title": f"Lesson {slug}",
                "ef": ef_min_unlock,
                "tau": tau_unlock,
                "req": requires_interview_pass,
            },
        )
    return lesson_id


async def _attach_quiz_with_cards(
    engine: AsyncEngine,
    *,
    course_id: UUID,
    module_id: UUID,
    student_id: UUID,
    cards: list[float],
    position: int = 1,
) -> tuple[UUID, list[UUID]]:
    """Create a quiz attached to module via module_items + cards w/ EF state."""
    quiz_id = uuid.uuid4()
    item_id = uuid.uuid4()
    question_ids: list[UUID] = [uuid.uuid4() for _ in cards]

    async with _conn(engine) as conn:
        await conn.execute(
            text(
                "INSERT INTO quizzes "
                "(id, course_id, module_id, title, status) "
                "VALUES (:id, :course, :module, 'Quiz', 'published')"
            ),
            {"id": quiz_id, "course": course_id, "module": module_id},
        )
        await conn.execute(
            text(
                "INSERT INTO module_items "
                "(id, module_id, item_type, quiz_id, position) "
                "VALUES (:id, :module, 'quiz', :quiz, :pos)"
            ),
            {
                "id": item_id,
                "module": module_id,
                "quiz": quiz_id,
                "pos": position,
            },
        )
        for idx, ef_value in enumerate(cards, start=1):
            qid = question_ids[idx - 1]
            await conn.execute(
                text(
                    "INSERT INTO quiz_questions "
                    "(id, quiz_id, position, question_type, prompt_text, "
                    "expected_response_time_ms, source_refs, review_status) "
                    "VALUES (:id, :quiz, :pos, 'multiple_choice', "
                    ":prompt, 30000, '[]'::jsonb, 'approved')"
                ),
                {
                    "id": qid,
                    "quiz": quiz_id,
                    "pos": idx,
                    "prompt": f"Q{idx}",
                },
            )
            await conn.execute(
                text(
                    "INSERT INTO student_card_state "
                    "(student_id, question_id, ef, interval_days, "
                    "repetition_count, due_at, total_reviews) "
                    "VALUES (:s, :q, :ef, 1, 1, NOW(), 1)"
                ),
                {"s": student_id, "q": qid, "ef": ef_value},
            )
    return quiz_id, question_ids


async def _add_lesson_prereq(engine: AsyncEngine, *, lesson_id: UUID, prereq_id: UUID) -> None:
    async with _conn(engine) as conn:
        await conn.execute(
            text("INSERT INTO lesson_prerequisites (lesson_id, prereq_lesson_id) VALUES (:l, :p)"),
            {"l": lesson_id, "p": prereq_id},
        )


async def _seed_passing_interview(
    engine: AsyncEngine, *, course_id: UUID, module_id: UUID, student_id: UUID
) -> UUID:
    config_id = uuid.uuid4()
    session_id = uuid.uuid4()
    async with _conn(engine) as conn:
        await conn.execute(
            text(
                "INSERT INTO interview_configs "
                "(id, course_id, module_id, title, status, supported_modes) "
                "VALUES (:id, :c, :m, 'Module Interview', 'published', 'text')"
            ),
            {"id": config_id, "c": course_id, "m": module_id},
        )
        await conn.execute(
            text(
                "INSERT INTO interview_sessions "
                "(id, interview_config_id, student_id, attempt_number, "
                "status, input_mode, pass_verdict) "
                "VALUES (:id, :cfg, :s, 1, 'completed', 'text', TRUE)"
            ),
            {"id": session_id, "cfg": config_id, "s": student_id},
        )
    return config_id


async def _seed_only_interview_config(
    engine: AsyncEngine, *, course_id: UUID, module_id: UUID
) -> UUID:
    config_id = uuid.uuid4()
    async with _conn(engine) as conn:
        await conn.execute(
            text(
                "INSERT INTO interview_configs "
                "(id, course_id, module_id, title, status, supported_modes) "
                "VALUES (:id, :c, :m, 'No-Pass Interview', 'published', 'text')"
            ),
            {"id": config_id, "c": course_id, "m": module_id},
        )
    return config_id


@pytest.mark.asyncio
async def test_eligible_all_passing(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
    cache_disabled: None,
) -> None:
    base = await _seed_org_user_course_module(engine)
    lesson_id = await _seed_lesson(engine, module_id=base["module_id"])
    await _attach_quiz_with_cards(
        engine,
        course_id=base["course_id"],
        module_id=base["module_id"],
        student_id=base["student_id"],
        cards=[2.5, 2.4, 2.3, 2.2, 2.1],
    )

    async with session_factory() as session:
        status = await check_lesson_unlock(
            session,
            student_id=base["student_id"],
            lesson_id=lesson_id,
        )

    assert isinstance(status, LessonUnlockStatus)
    assert status.eligible is True
    assert status.total_cards == 5
    assert status.passing_cards == 5
    assert status.current_ratio == pytest.approx(1.0)
    assert status.required_ratio == pytest.approx(0.8)
    assert status.ef_min == pytest.approx(2.0)
    assert status.blocking_cards == []
    assert status.prereq_lesson_ids_unlocked is True
    assert status.interview_pass_required is False
    assert status.interview_passed is False
    assert status.next_unlock_estimate is None


@pytest.mark.asyncio
async def test_blocked_partial_fail(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
    cache_disabled: None,
) -> None:
    base = await _seed_org_user_course_module(engine)
    lesson_id = await _seed_lesson(engine, module_id=base["module_id"])
    _, question_ids = await _attach_quiz_with_cards(
        engine,
        course_id=base["course_id"],
        module_id=base["module_id"],
        student_id=base["student_id"],
        cards=[2.5, 2.5, 2.5, 1.5, 1.7],
    )

    async with session_factory() as session:
        status = await check_lesson_unlock(
            session,
            student_id=base["student_id"],
            lesson_id=lesson_id,
        )

    assert status.eligible is False
    assert status.total_cards == 5
    assert status.passing_cards == 3
    assert status.current_ratio == pytest.approx(0.6)
    assert len(status.blocking_cards) == 2
    blocking_ef_values = {round(card.current_ef, 1) for card in status.blocking_cards}
    assert blocking_ef_values == {1.5, 1.7}
    blocking_qids = {card.question_id for card in status.blocking_cards}
    assert blocking_qids.issubset(set(question_ids))
    assert status.next_unlock_estimate is not None
    assert "card" in status.next_unlock_estimate


@pytest.mark.asyncio
async def test_empty_lesson_eligible_based_on_prereqs(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
    cache_disabled: None,
) -> None:
    base = await _seed_org_user_course_module(engine)
    lesson_id = await _seed_lesson(engine, module_id=base["module_id"])

    async with session_factory() as session:
        status = await check_lesson_unlock(
            session,
            student_id=base["student_id"],
            lesson_id=lesson_id,
        )

    assert status.eligible is True
    assert status.total_cards == 0
    assert status.passing_cards == 0
    assert status.current_ratio == pytest.approx(0.0)
    assert status.blocking_cards == []
    assert status.prereq_lesson_ids_unlocked is True


@pytest.mark.asyncio
async def test_prereq_blocked_recursion(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
    cache_disabled: None,
) -> None:
    base = await _seed_org_user_course_module(engine)
    lesson_a = await _seed_lesson(engine, module_id=base["module_id"], slug_suffix="a")
    lesson_b = await _seed_lesson(engine, module_id=base["module_id"], slug_suffix="b")
    lesson_c = await _seed_lesson(engine, module_id=base["module_id"], slug_suffix="c")
    await _add_lesson_prereq(engine, lesson_id=lesson_a, prereq_id=lesson_b)
    await _add_lesson_prereq(engine, lesson_id=lesson_b, prereq_id=lesson_c)

    await _attach_quiz_with_cards(
        engine,
        course_id=base["course_id"],
        module_id=base["module_id"],
        student_id=base["student_id"],
        cards=[2.5, 2.5, 1.4, 1.4, 1.4],
        position=1,
    )

    async with session_factory() as session:
        status_a = await check_lesson_unlock(
            session,
            student_id=base["student_id"],
            lesson_id=lesson_a,
        )

    assert status_a.eligible is False
    assert status_a.prereq_lesson_ids_unlocked is False
    assert status_a.next_unlock_estimate is not None
    assert "prerequisite" in status_a.next_unlock_estimate.lower()


@pytest.mark.asyncio
async def test_interview_required_not_passed(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
    cache_disabled: None,
) -> None:
    base = await _seed_org_user_course_module(engine)
    lesson_id = await _seed_lesson(
        engine,
        module_id=base["module_id"],
        requires_interview_pass=True,
    )
    await _attach_quiz_with_cards(
        engine,
        course_id=base["course_id"],
        module_id=base["module_id"],
        student_id=base["student_id"],
        cards=[2.5, 2.5, 2.5, 2.5, 2.5],
    )
    await _seed_only_interview_config(
        engine, course_id=base["course_id"], module_id=base["module_id"]
    )

    async with session_factory() as session:
        status = await check_lesson_unlock(
            session,
            student_id=base["student_id"],
            lesson_id=lesson_id,
        )

    assert status.eligible is False
    assert status.interview_pass_required is True
    assert status.interview_passed is False
    assert status.passing_cards == 5
    assert status.next_unlock_estimate is not None
    assert "interview" in status.next_unlock_estimate.lower()


@pytest.mark.asyncio
async def test_interview_required_passed(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
    cache_disabled: None,
) -> None:
    base = await _seed_org_user_course_module(engine)
    lesson_id = await _seed_lesson(
        engine,
        module_id=base["module_id"],
        requires_interview_pass=True,
    )
    await _attach_quiz_with_cards(
        engine,
        course_id=base["course_id"],
        module_id=base["module_id"],
        student_id=base["student_id"],
        cards=[2.5, 2.5, 2.5, 2.5, 2.5],
    )
    await _seed_passing_interview(
        engine,
        course_id=base["course_id"],
        module_id=base["module_id"],
        student_id=base["student_id"],
    )

    async with session_factory() as session:
        status = await check_lesson_unlock(
            session,
            student_id=base["student_id"],
            lesson_id=lesson_id,
        )

    assert status.eligible is True
    assert status.interview_pass_required is True
    assert status.interview_passed is True


@pytest.mark.asyncio
async def test_cycle_detection(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
    monkeypatch: pytest.MonkeyPatch,
    cache_disabled: None,
) -> None:
    base = await _seed_org_user_course_module(engine)
    lesson_a = await _seed_lesson(engine, module_id=base["module_id"], slug_suffix="cycle-a")
    lesson_b = await _seed_lesson(engine, module_id=base["module_id"], slug_suffix="cycle-b")
    await _add_lesson_prereq(engine, lesson_id=lesson_a, prereq_id=lesson_b)
    await _add_lesson_prereq(engine, lesson_id=lesson_b, prereq_id=lesson_a)

    from abridgeai.features.spaced_repetition.sm2 import lesson_unlock as mod

    captured: list[tuple[str, dict[str, Any]]] = []
    original_warning = mod.logger.warning

    def _spy(msg: str, *args: Any, **kwargs: Any) -> None:
        captured.append((msg, kwargs.get("extra") or {}))
        original_warning(msg, *args, **kwargs)

    monkeypatch.setattr(mod.logger, "warning", _spy)

    async with session_factory() as session:
        status = await check_lesson_unlock(
            session,
            student_id=base["student_id"],
            lesson_id=lesson_a,
        )

    assert status.eligible is True
    assert status.prereq_lesson_ids_unlocked is True
    cycle_msgs = [m for m, _ in captured if "cycle_detected" in m]
    assert cycle_msgs, f"expected cycle_detected warn; got {captured!r}"


@pytest.mark.asyncio
async def test_cache_speedup(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
    redis_clean: Any,
) -> None:
    base = await _seed_org_user_course_module(engine)
    lesson_id = await _seed_lesson(engine, module_id=base["module_id"])
    await _attach_quiz_with_cards(
        engine,
        course_id=base["course_id"],
        module_id=base["module_id"],
        student_id=base["student_id"],
        cards=[2.5, 2.5, 2.5, 2.5, 2.5],
    )

    async with session_factory() as session:
        cold_start = time.perf_counter()
        cold_status = await check_lesson_unlock(
            session,
            student_id=base["student_id"],
            lesson_id=lesson_id,
        )
        cold = time.perf_counter() - cold_start

        warm_start = time.perf_counter()
        warm_status = await check_lesson_unlock(
            session,
            student_id=base["student_id"],
            lesson_id=lesson_id,
        )
        warm = time.perf_counter() - warm_start

    assert cold_status.eligible is True
    assert warm_status.eligible is True
    assert cold_status.total_cards == warm_status.total_cards
    assert cold_status.passing_cards == warm_status.passing_cards
    assert warm < 0.2 * cold, f"warm cache call ({warm:.4f}s) was not <20% of cold ({cold:.4f}s)"


@pytest.mark.asyncio
async def test_unknown_lesson_returns_ineligible(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
    cache_disabled: None,
) -> None:
    base = await _seed_org_user_course_module(engine)
    bogus = uuid.uuid4()
    async with session_factory() as session:
        status = await check_lesson_unlock(
            session,
            student_id=base["student_id"],
            lesson_id=bogus,
        )
    assert status.eligible is False
    assert status.total_cards == 0
    assert status.passing_cards == 0
