"""Integration tests for :func:`quiz_results_summary` (grading-method-aware).

Seeds a single quiz with two students and a mix of completed attempts
(plus one in-progress attempt that must be ignored), then asserts the
per-method headline reduction, the aggregate stats (mean / median /
percentiles / pass-rate / mean-time) and the 11-bucket score histogram.

The function reduces COMPLETED attempts (``status IN ('submitted','graded')``
AND ``score_percent IS NOT NULL``) to one headline row per student
according to ``grading_method`` (highest/average/first/last), then
aggregates over that headline set.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

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

from abridgeai.core.config import get_settings
from abridgeai.features.quizzes.queries.analytics import quiz_results_summary


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
async def scenario(engine: AsyncEngine) -> AsyncIterator[dict[str, uuid.UUID]]:
    """Seed one course/module/quiz with two students and mixed attempts.

    Completed attempts (passing threshold 70):
      student A: #1 score 40 (passed=False), #2 score 100 (passed=True)
      student B: #1 score 60 (passed=False), #2 score 80  (passed=True)
    Plus one in_progress attempt (score NULL) that MUST be ignored.

    A second empty quiz is seeded for the zero-attempts case.
    """
    org = uuid.uuid4()
    teacher = uuid.uuid4()
    student_a = uuid.uuid4()
    student_b = uuid.uuid4()
    course = uuid.uuid4()
    module = uuid.uuid4()
    quiz = uuid.uuid4()
    empty_quiz = uuid.uuid4()

    now = datetime.now(UTC)

    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO organizations (id, slug, name, status) "
                "VALUES (:id, :slug, 'Analytics Org', 'active')"
            ),
            {"id": org, "slug": f"an-org-{org.hex[:8]}"},
        )
        await conn.execute(
            text(
                "INSERT INTO users (id, primary_email, status) VALUES "
                "(:t, :te, 'active'), (:a, :ae, 'active'), (:b, :be, 'active')"
            ),
            {
                "t": teacher,
                "a": student_a,
                "b": student_b,
                "te": f"t-{teacher.hex[:8]}@test.local",
                "ae": f"a-{student_a.hex[:8]}@test.local",
                "be": f"b-{student_b.hex[:8]}@test.local",
            },
        )
        await conn.execute(
            text(
                "INSERT INTO courses "
                "(id, organization_id, owner_user_id, slug, title, status) "
                "VALUES (:id, :org, :u, :slug, 'Analytics Course', 'published')"
            ),
            {"id": course, "org": org, "u": teacher, "slug": f"an-course-{course.hex[:8]}"},
        )
        await conn.execute(
            text(
                "INSERT INTO modules (id, course_id, title, position, status) "
                "VALUES (:id, :c, 'Module', 1, 'published')"
            ),
            {"id": module, "c": course},
        )
        await conn.execute(
            text(
                "INSERT INTO quizzes (id, course_id, module_id, title, status) VALUES "
                "(:q, :c, :m, 'Results Quiz', 'published'), "
                "(:eq, :c, :m, 'Empty Quiz', 'published')"
            ),
            {"q": quiz, "eq": empty_quiz, "c": course, "m": module},
        )

        # student A: #1 = 40 (fail), #2 = 100 (pass)
        # student B: #1 = 60 (fail), #2 = 80  (pass)
        attempts = [
            (student_a, 1, "graded", 40, False, 300),
            (student_a, 2, "graded", 100, True, 200),
            (student_b, 1, "graded", 60, False, 400),
            (student_b, 2, "submitted", 80, True, 500),
        ]
        for sid, num, status, score, passed, secs in attempts:
            await conn.execute(
                text(
                    "INSERT INTO quiz_attempts "
                    "(id, quiz_id, student_id, attempt_number, status, "
                    " score_percent, passed, time_taken_seconds, started_at, submitted_at) "
                    "VALUES (:id, :q, :s, :n, :st, :sc, :p, :secs, :start, :submit)"
                ),
                {
                    "id": uuid.uuid4(),
                    "q": quiz,
                    "s": sid,
                    "n": num,
                    "st": status,
                    "sc": score,
                    "p": passed,
                    "secs": secs,
                    "start": now - timedelta(hours=2),
                    "submit": now - timedelta(hours=1),
                },
            )

        # in_progress attempt (score NULL) — MUST be ignored by the summary.
        await conn.execute(
            text(
                "INSERT INTO quiz_attempts "
                "(id, quiz_id, student_id, attempt_number, status, "
                " score_percent, passed, time_taken_seconds, started_at) "
                "VALUES (:id, :q, :s, :n, 'in_progress', NULL, NULL, NULL, :start)"
            ),
            {
                "id": uuid.uuid4(),
                "q": quiz,
                "s": student_a,
                "n": 3,
                "start": now,
            },
        )

    yield {
        "course_id": course,
        "quiz_id": quiz,
        "empty_quiz_id": empty_quiz,
        "student_a": student_a,
        "student_b": student_b,
        "org_id": org,
    }

    async with engine.begin() as conn:
        await conn.execute(
            text("DELETE FROM quiz_attempts WHERE quiz_id IN (:q, :eq)"),
            {"q": quiz, "eq": empty_quiz},
        )
        await conn.execute(
            text("DELETE FROM quizzes WHERE id IN (:q, :eq)"),
            {"q": quiz, "eq": empty_quiz},
        )
        await conn.execute(text("DELETE FROM modules WHERE id = :m"), {"m": module})
        await conn.execute(text("DELETE FROM courses WHERE id = :c"), {"c": course})
        await conn.execute(
            text("DELETE FROM users WHERE id IN (:t, :a, :b)"),
            {"t": teacher, "a": student_a, "b": student_b},
        )
        await conn.execute(text("DELETE FROM organizations WHERE id = :o"), {"o": org})


async def test_highest_headline_and_aggregates(
    session_factory: async_sessionmaker[AsyncSession],
    scenario: dict[str, uuid.UUID],
) -> None:
    """grading_method='highest' → headline A=100, B=80."""
    async with session_factory() as db:
        result = await quiz_results_summary(db, scenario["quiz_id"], "highest")

    assert result["total_attempts"] == 4  # 4 completed; in_progress excluded
    assert result["unique_students"] == 2
    assert result["mean_score"] == pytest.approx(90.0)
    assert result["median_score"] == pytest.approx(90.0)
    assert result["p25"] == pytest.approx(85.0)
    assert result["p75"] == pytest.approx(95.0)
    assert result["pass_rate"] == pytest.approx(1.0)  # both headline attempts passed
    assert result["mean_time_seconds"] == pytest.approx(350.0)  # A#2=200, B#2=500 → 350

    assert len(result["histogram"]) == 11
    assert sum(b["count"] for b in result["histogram"]) == 2


async def test_average_headline_and_aggregates(
    session_factory: async_sessionmaker[AsyncSession],
    scenario: dict[str, uuid.UUID],
) -> None:
    """grading_method='average' → headline A=(40+100)/2=70, B=(60+80)/2=70."""
    async with session_factory() as db:
        result = await quiz_results_summary(db, scenario["quiz_id"], "average")

    assert result["total_attempts"] == 4
    assert result["unique_students"] == 2
    assert result["mean_score"] == pytest.approx(70.0)
    assert result["median_score"] == pytest.approx(70.0)
    assert result["pass_rate"] == pytest.approx(1.0)  # bool_or → both students have a pass
    assert len(result["histogram"]) == 11
    assert sum(b["count"] for b in result["histogram"]) == 2


async def test_first_headline(
    session_factory: async_sessionmaker[AsyncSession],
    scenario: dict[str, uuid.UUID],
) -> None:
    """grading_method='first' → headline A=40, B=60 (attempt #1 each)."""
    async with session_factory() as db:
        result = await quiz_results_summary(db, scenario["quiz_id"], "first")

    assert result["total_attempts"] == 4
    assert result["unique_students"] == 2
    assert result["mean_score"] == pytest.approx(50.0)  # (40+60)/2
    assert result["pass_rate"] == pytest.approx(0.0)  # neither first attempt passed
    assert sum(b["count"] for b in result["histogram"]) == 2


async def test_last_headline(
    session_factory: async_sessionmaker[AsyncSession],
    scenario: dict[str, uuid.UUID],
) -> None:
    """grading_method='last' → headline A=100, B=80 (attempt #2 each)."""
    async with session_factory() as db:
        result = await quiz_results_summary(db, scenario["quiz_id"], "last")

    assert result["total_attempts"] == 4
    assert result["unique_students"] == 2
    assert result["mean_score"] == pytest.approx(90.0)  # (100+80)/2
    assert result["pass_rate"] == pytest.approx(1.0)
    assert sum(b["count"] for b in result["histogram"]) == 2


async def test_zero_attempts(
    session_factory: async_sessionmaker[AsyncSession],
    scenario: dict[str, uuid.UUID],
) -> None:
    """A quiz with no attempts → zeroed counts, None stats, empty histogram."""
    async with session_factory() as db:
        result = await quiz_results_summary(db, scenario["empty_quiz_id"], "highest")

    assert result["total_attempts"] == 0
    assert result["unique_students"] == 0
    assert result["mean_score"] is None
    assert result["median_score"] is None
    assert result["p25"] is None
    assert result["p75"] is None
    assert result["pass_rate"] is None
    assert result["mean_time_seconds"] is None
    assert len(result["histogram"]) == 11
    assert sum(b["count"] for b in result["histogram"]) == 0


async def test_invalid_grading_method_raises(
    session_factory: async_sessionmaker[AsyncSession],
    scenario: dict[str, uuid.UUID],
) -> None:
    async with session_factory() as db:
        with pytest.raises(ValueError):
            await quiz_results_summary(db, scenario["quiz_id"], "bogus")
