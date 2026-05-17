"""Integration tests for the quizzes queries layer (T5.3).

Plan §5491-5543. Asserts:

* :mod:`abridgeai.features.quizzes.queries.published` excludes drafts,
  raises :class:`CooldownActive` / :class:`MaxAttemptsReached` correctly,
  and returns published quizzes attached to a module.
* :mod:`abridgeai.features.quizzes.queries.authoring` returns all
  states and computes dedup keys.
* :mod:`abridgeai.features.quizzes.queries.analytics` aggregates
  attempts and ranks missed questions.
* The queries directory is single-namespace + sql/ co-located —
  no ``orm/`` / ``raw/`` mechanism split (locked decision).
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
from abridgeai.features.quizzes.queries import (
    CooldownActive,
    MaxAttemptsReached,
    get_published_quiz,
    get_quiz_for_authoring,
    get_quiz_for_taking,
    list_existing_module_question_keys,
    list_published_quizzes_for_module,
    list_questions_with_source_refs,
    list_quizzes_for_course,
    quiz_completion_rate,
    top_missed_questions,
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
async def session_factory(
    engine: AsyncEngine,
) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False, autoflush=False)


@pytest_asyncio.fixture
async def fixture_data(engine: AsyncEngine) -> AsyncIterator[dict]:
    """Seed a quizzes tree across two modules with mixed statuses + attempts."""

    org = uuid.uuid4()
    teacher = uuid.uuid4()
    student = uuid.uuid4()
    fresh_student = uuid.uuid4()
    course = uuid.uuid4()
    module_a = uuid.uuid4()
    module_b = uuid.uuid4()
    quiz_pub = uuid.uuid4()
    quiz_draft = uuid.uuid4()
    quiz_archived = uuid.uuid4()
    quiz_cooldown = uuid.uuid4()
    quiz_capped = uuid.uuid4()
    quiz_module_b = uuid.uuid4()
    q_easy = uuid.uuid4()
    q_hard = uuid.uuid4()
    q_dup_a = uuid.uuid4()
    q_dup_b = uuid.uuid4()

    now = datetime.now(UTC)

    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO organizations (id, slug, name, status) "
                "VALUES (:id, :slug, 'Quiz Org', 'active')"
            ),
            {"id": org, "slug": f"quiz-org-{org.hex[:8]}"},
        )
        await conn.execute(
            text(
                "INSERT INTO users (id, primary_email, status) VALUES "
                "(:t, :te, 'active'), (:s, :se, 'active'), (:f, :fe, 'active')"
            ),
            {
                "t": teacher,
                "s": student,
                "f": fresh_student,
                "te": f"t-{teacher.hex[:8]}@test.local",
                "se": f"s-{student.hex[:8]}@test.local",
                "fe": f"f-{fresh_student.hex[:8]}@test.local",
            },
        )
        await conn.execute(
            text(
                "INSERT INTO courses "
                "(id, organization_id, owner_user_id, slug, title, status) "
                "VALUES (:id, :org, :u, :slug, 'Quiz Course', 'published')"
            ),
            {
                "id": course,
                "org": org,
                "u": teacher,
                "slug": f"quiz-course-{course.hex[:8]}",
            },
        )
        await conn.execute(
            text(
                "INSERT INTO modules "
                "(id, course_id, title, position, status) VALUES "
                "(:a, :c, 'Module A', 1, 'published'), "
                "(:b, :c, 'Module B', 2, 'published')"
            ),
            {"a": module_a, "b": module_b, "c": course},
        )
        await conn.execute(
            text(
                "INSERT INTO quizzes "
                "(id, course_id, module_id, title, status, "
                " cooldown_hours, max_attempts) VALUES "
                "(:p, :c, :m, 'Pub Quiz', 'published', NULL, NULL), "
                "(:d, :c, :m, 'Draft Quiz', 'draft', NULL, NULL), "
                "(:a, :c, :m, 'Archived Quiz', 'archived', NULL, NULL), "
                "(:cd, :c, :m, 'Cooldown Quiz', 'published', 2, NULL), "
                "(:cp, :c, :m, 'Capped Quiz', 'published', NULL, 3), "
                "(:mb, :c, :mb_id, 'Module B Quiz', 'published', NULL, NULL)"
            ),
            {
                "p": quiz_pub,
                "d": quiz_draft,
                "a": quiz_archived,
                "cd": quiz_cooldown,
                "cp": quiz_capped,
                "mb": quiz_module_b,
                "c": course,
                "m": module_a,
                "mb_id": module_b,
            },
        )
        await conn.execute(
            text(
                "INSERT INTO quiz_questions ("
                "id, quiz_id, position, question_type, prompt_text, "
                "source_refs, review_status) VALUES "
                "(:e, :p, 1, 'multiple_choice', 'Easy Q?', "
                " '[{\"chunk_id\":\"c1\"}]'::jsonb, 'approved'), "
                "(:h, :p, 2, 'multiple_choice', 'Hard Q?', "
                " '[{\"chunk_id\":\"c2\"}]'::jsonb, 'approved'), "
                "(:da, :p, 3, 'multiple_choice', 'Dup Prompt', "
                " '[{\"chunk_id\":\"cx\"}]'::jsonb, 'approved'), "
                "(:db, :mb, 1, 'multiple_choice', 'Module B Q', "
                " '[]'::jsonb, 'approved')"
            ),
            {
                "e": q_easy,
                "h": q_hard,
                "da": q_dup_a,
                "db": q_dup_b,
                "p": quiz_pub,
                "mb": quiz_module_b,
            },
        )

        # cooldown attempt: submitted 1 hour ago (within 2-hour window)
        cooldown_attempt = uuid.uuid4()
        await conn.execute(
            text(
                "INSERT INTO quiz_attempts ("
                "id, quiz_id, student_id, attempt_number, status, "
                "started_at, submitted_at) VALUES "
                "(:id, :q, :s, 1, 'submitted', :start, :submit)"
            ),
            {
                "id": cooldown_attempt,
                "q": quiz_cooldown,
                "s": student,
                "start": now - timedelta(hours=2),
                "submit": now - timedelta(hours=1),
            },
        )

        # capped attempts: 3 attempts already used (== max_attempts)
        for n in range(1, 4):
            await conn.execute(
                text(
                    "INSERT INTO quiz_attempts ("
                    "id, quiz_id, student_id, attempt_number, status, "
                    "started_at, submitted_at, score_percent, passed) VALUES "
                    "(:id, :q, :s, :n, 'submitted', :start, :submit, :sp, :p)"
                ),
                {
                    "id": uuid.uuid4(),
                    "q": quiz_capped,
                    "s": student,
                    "n": n,
                    "start": now - timedelta(days=n + 1),
                    "submit": now - timedelta(days=n + 1, minutes=-30),
                    "sp": 80 if n != 2 else 40,
                    "p": n != 2,
                },
            )

        # analytics seeding: 6 attempts on quiz_pub, q_easy mostly correct,
        # q_hard mostly wrong — gives top_missed_questions stable ordering
        per_question_answers: dict[uuid.UUID, list[bool]] = {
            q_easy: [True, True, True, True, True, False],
            q_hard: [False, False, False, False, True, False],
        }
        attempt_counter = 100
        for qid, correctness_list in per_question_answers.items():
            for is_correct in correctness_list:
                attempt_counter += 1
                attempt_id = uuid.uuid4()
                await conn.execute(
                    text(
                        "INSERT INTO quiz_attempts ("
                        "id, quiz_id, student_id, attempt_number, status, "
                        "started_at, submitted_at, score_percent, passed) VALUES "
                        "(:id, :q, :s, :n, 'submitted', :start, :submit, "
                        ":sp, :p)"
                    ),
                    {
                        "id": attempt_id,
                        "q": quiz_pub,
                        "s": fresh_student,
                        "n": attempt_counter,
                        "start": now - timedelta(hours=4),
                        "submit": now - timedelta(hours=3),
                        "sp": 90 if is_correct else 50,
                        "p": is_correct,
                    },
                )
                await conn.execute(
                    text(
                        "INSERT INTO quiz_attempt_answers ("
                        "id, attempt_id, question_id, is_correct, points_awarded) "
                        "VALUES (:id, :a, :q, :c, 1)"
                    ),
                    {
                        "id": uuid.uuid4(),
                        "a": attempt_id,
                        "q": qid,
                        "c": is_correct,
                    },
                )

    data = {
        "org": org,
        "teacher": teacher,
        "student": student,
        "fresh_student": fresh_student,
        "course": course,
        "module_a": module_a,
        "module_b": module_b,
        "quiz_pub": quiz_pub,
        "quiz_draft": quiz_draft,
        "quiz_archived": quiz_archived,
        "quiz_cooldown": quiz_cooldown,
        "quiz_capped": quiz_capped,
        "quiz_module_b": quiz_module_b,
        "q_easy": q_easy,
        "q_hard": q_hard,
        "q_dup_a": q_dup_a,
        "q_dup_b": q_dup_b,
    }
    yield data

    async with engine.begin() as conn:
        await conn.execute(
            text(
                "DELETE FROM quiz_attempt_answers WHERE attempt_id IN ("
                "SELECT id FROM quiz_attempts WHERE quiz_id = ANY(:ids))"
            ),
            {
                "ids": [
                    quiz_pub,
                    quiz_cooldown,
                    quiz_capped,
                    quiz_module_b,
                ]
            },
        )
        await conn.execute(
            text("DELETE FROM quiz_attempts WHERE quiz_id = ANY(:ids)"),
            {
                "ids": [
                    quiz_pub,
                    quiz_cooldown,
                    quiz_capped,
                    quiz_module_b,
                ]
            },
        )
        await conn.execute(
            text("DELETE FROM quiz_questions WHERE quiz_id = ANY(:ids)"),
            {"ids": [quiz_pub, quiz_module_b]},
        )
        await conn.execute(
            text("DELETE FROM quizzes WHERE id = ANY(:ids)"),
            {
                "ids": [
                    quiz_pub,
                    quiz_draft,
                    quiz_archived,
                    quiz_cooldown,
                    quiz_capped,
                    quiz_module_b,
                ]
            },
        )
        await conn.execute(
            text("DELETE FROM modules WHERE id = ANY(:ids)"),
            {"ids": [module_a, module_b]},
        )
        await conn.execute(text("DELETE FROM courses WHERE id = :id"), {"id": course})
        await conn.execute(
            text("DELETE FROM users WHERE id = ANY(:ids)"),
            {"ids": [teacher, student, fresh_student]},
        )
        await conn.execute(text("DELETE FROM organizations WHERE id = :id"), {"id": org})


async def test_get_published_quiz_excludes_drafts(
    session_factory: async_sessionmaker[AsyncSession],
    fixture_data: dict,
) -> None:
    async with session_factory() as session:
        pub = await get_published_quiz(session, fixture_data["quiz_pub"])
        draft = await get_published_quiz(session, fixture_data["quiz_draft"])
        archived = await get_published_quiz(session, fixture_data["quiz_archived"])
    assert pub is not None
    assert pub.status == "published"
    assert draft is None
    assert archived is None


async def test_list_published_quizzes_for_module_filters_status(
    session_factory: async_sessionmaker[AsyncSession],
    fixture_data: dict,
) -> None:
    async with session_factory() as session:
        rows = await list_published_quizzes_for_module(session, fixture_data["module_a"])
    ids = {q.id for q in rows}
    assert fixture_data["quiz_pub"] in ids
    assert fixture_data["quiz_cooldown"] in ids
    assert fixture_data["quiz_capped"] in ids
    assert fixture_data["quiz_draft"] not in ids
    assert fixture_data["quiz_archived"] not in ids
    assert fixture_data["quiz_module_b"] not in ids


async def test_get_quiz_for_taking_cooldown_active_raises(
    session_factory: async_sessionmaker[AsyncSession],
    fixture_data: dict,
) -> None:
    async with session_factory() as session:
        with pytest.raises(CooldownActive) as excinfo:
            await get_quiz_for_taking(
                session,
                fixture_data["quiz_cooldown"],
                fixture_data["student"],
            )
    assert excinfo.value.quiz_id == fixture_data["quiz_cooldown"]
    assert excinfo.value.retry_after > datetime.now(UTC)


async def test_get_quiz_for_taking_max_attempts_reached_raises(
    session_factory: async_sessionmaker[AsyncSession],
    fixture_data: dict,
) -> None:
    async with session_factory() as session:
        with pytest.raises(MaxAttemptsReached) as excinfo:
            await get_quiz_for_taking(
                session,
                fixture_data["quiz_capped"],
                fixture_data["student"],
            )
    assert excinfo.value.quiz_id == fixture_data["quiz_capped"]
    assert excinfo.value.max_attempts == 3


async def test_get_quiz_for_taking_succeeds_when_eligible(
    session_factory: async_sessionmaker[AsyncSession],
    fixture_data: dict,
) -> None:
    async with session_factory() as session:
        quiz = await get_quiz_for_taking(
            session,
            fixture_data["quiz_pub"],
            fixture_data["fresh_student"],
        )
        cooldown_for_fresh = await get_quiz_for_taking(
            session,
            fixture_data["quiz_cooldown"],
            fixture_data["fresh_student"],
        )
    assert quiz is not None
    assert quiz.id == fixture_data["quiz_pub"]
    assert cooldown_for_fresh is not None


async def test_get_quiz_for_authoring_returns_all_states(
    session_factory: async_sessionmaker[AsyncSession],
    fixture_data: dict,
) -> None:
    async with session_factory() as session:
        draft = await get_quiz_for_authoring(session, fixture_data["quiz_draft"])
        archived = await get_quiz_for_authoring(session, fixture_data["quiz_archived"])
        course_quizzes = await list_quizzes_for_course(session, fixture_data["course"])
    assert draft is not None
    assert draft.status == "draft"
    assert archived is not None
    assert archived.status == "archived"
    statuses = {q.status for q in course_quizzes}
    assert {"draft", "published", "archived"}.issubset(statuses)


async def test_list_existing_module_question_keys(
    session_factory: async_sessionmaker[AsyncSession],
    fixture_data: dict,
) -> None:
    async with session_factory() as session:
        keys = await list_existing_module_question_keys(session, fixture_data["quiz_pub"])
        questions = await list_questions_with_source_refs(session, fixture_data["quiz_pub"])
    assert len(keys) == 3
    assert all(isinstance(k, str) and len(k) == 64 for k in keys)
    assert {q.id for q in questions} == {
        fixture_data["q_easy"],
        fixture_data["q_hard"],
        fixture_data["q_dup_a"],
    }


async def test_top_missed_questions_orders_by_correctness_asc(
    session_factory: async_sessionmaker[AsyncSession],
    fixture_data: dict,
) -> None:
    async with session_factory() as session:
        rows = await top_missed_questions(session, fixture_data["course"], limit=5)
    assert len(rows) >= 2
    assert rows[0]["question_id"] == fixture_data["q_hard"]
    assert rows[0]["correctness_rate"] < rows[-1]["correctness_rate"]
    for row in rows:
        assert row["attempt_count"] >= 5


async def test_quiz_completion_rate_aggregates(
    session_factory: async_sessionmaker[AsyncSession],
    fixture_data: dict,
) -> None:
    async with session_factory() as session:
        stats = await quiz_completion_rate(session, fixture_data["quiz_capped"])
    assert stats["total_attempts"] == 3
    assert stats["completed_count"] == 3
    assert stats["avg_score"] is not None
    assert 60 <= stats["avg_score"] <= 80
    assert stats["pass_rate"] is not None
    assert 0.5 < stats["pass_rate"] <= 1.0


def test_no_mechanism_split() -> None:
    queries_dir = (
        Path(__file__).resolve().parents[2] / "abridgeai" / "features" / "quizzes" / "queries"
    )
    subdirs = {p.name for p in queries_dir.iterdir() if p.is_dir() and not p.name.startswith("__")}
    assert subdirs == {"sql"}, (
        f"Locked decision: queries/ must contain only sql/ as subdir, got {subdirs}"
    )
