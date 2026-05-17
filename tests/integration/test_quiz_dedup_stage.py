"""Integration tests for the quiz dedup stage (T5.8).

Plan §5747-5787. Asserts:

* Exact-prompt + chunk-overlap duplicates are removed.
* Drops carry human-readable reason strings (audit / teacher review).
* Cross-quiz collision detection delegates to the T5.3 query so an
  existing question elsewhere in the same module is enough to drop
  a candidate.
* The candidate hash matches Postgres' ``module_question_keys.sql``
  hash (same SQL ⇒ symmetric deletion).

Tests run against the docker postgres bound at the
``DATABASE_URL`` test settings — same engine fixture pattern as
``test_quiz_queries.py`` (T5.3).
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from pathlib import Path

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
from abridgeai.features.quizzes.ai.stages.dedup import (
    REASON_BATCH_DUPLICATE,
    REASON_EMPTY_PROMPT,
    REASON_EXISTING_MODULE_DUPLICATE,
    discard_duplicates,
)
from abridgeai.features.quizzes.queries import get_quiz_for_authoring


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
    """Two quizzes under the same module (target + sibling) so dedup
    can prove cross-quiz collision detection without polluting the
    target quiz's own question set."""

    org = uuid.uuid4()
    teacher = uuid.uuid4()
    course = uuid.uuid4()
    module_a = uuid.uuid4()
    target_quiz = uuid.uuid4()
    sibling_quiz = uuid.uuid4()
    sibling_q = uuid.uuid4()

    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO organizations (id, slug, name, status) "
                "VALUES (:id, :slug, 'Dedup Org', 'active')"
            ),
            {"id": org, "slug": f"dedup-org-{org.hex[:8]}"},
        )
        await conn.execute(
            text("INSERT INTO users (id, primary_email, status) VALUES (:t, :te, 'active')"),
            {"t": teacher, "te": f"t-{teacher.hex[:8]}@test.local"},
        )
        await conn.execute(
            text(
                "INSERT INTO courses "
                "(id, organization_id, owner_user_id, slug, title, status) "
                "VALUES (:id, :org, :u, :slug, 'Dedup Course', 'published')"
            ),
            {
                "id": course,
                "org": org,
                "u": teacher,
                "slug": f"dedup-course-{course.hex[:8]}",
            },
        )
        await conn.execute(
            text(
                "INSERT INTO modules "
                "(id, course_id, title, position, status) "
                "VALUES (:m, :c, 'Dedup Module', 1, 'published')"
            ),
            {"m": module_a, "c": course},
        )
        await conn.execute(
            text(
                "INSERT INTO quizzes "
                "(id, course_id, module_id, title, status) VALUES "
                "(:t, :c, :m, 'Target Quiz', 'draft'), "
                "(:s, :c, :m, 'Sibling Quiz', 'published')"
            ),
            {
                "t": target_quiz,
                "s": sibling_quiz,
                "c": course,
                "m": module_a,
            },
        )

    data = {
        "org": org,
        "teacher": teacher,
        "course": course,
        "module_a": module_a,
        "target_quiz": target_quiz,
        "sibling_quiz": sibling_quiz,
        "sibling_q": sibling_q,
    }
    yield data

    async with engine.begin() as conn:
        await conn.execute(
            text("DELETE FROM quiz_questions WHERE quiz_id = ANY(:ids)"),
            {"ids": [target_quiz, sibling_quiz]},
        )
        await conn.execute(
            text("DELETE FROM quizzes WHERE id = ANY(:ids)"),
            {"ids": [target_quiz, sibling_quiz]},
        )
        await conn.execute(
            text("DELETE FROM modules WHERE id = :id"),
            {"id": module_a},
        )
        await conn.execute(text("DELETE FROM courses WHERE id = :id"), {"id": course})
        await conn.execute(text("DELETE FROM users WHERE id = :id"), {"id": teacher})
        await conn.execute(text("DELETE FROM organizations WHERE id = :id"), {"id": org})


def _candidate(prompt: str, chunk_ids: list[str]) -> dict:
    return {
        "prompt_text": prompt,
        "source_refs": [{"chunk_id": c} for c in chunk_ids],
        "question_type": "multiple_choice",
    }


async def test_dedup_removes_exact_duplicates(
    session_factory: async_sessionmaker[AsyncSession],
    fixture_data: dict,
) -> None:
    """5 candidates with 2 intra-batch dupes → 3 kept, 2 dropped."""

    async with session_factory() as session:
        quiz = await get_quiz_for_authoring(session, fixture_data["target_quiz"])
        assert quiz is not None
        candidates = [
            _candidate("What is X?", ["c1"]),
            _candidate("Define Y.", ["c2"]),
            _candidate("What is X?", ["c1"]),
            _candidate("Explain Z.", ["c3"]),
            _candidate("Define Y.", ["c2"]),
        ]
        kept, drops = await discard_duplicates(session, quiz, candidates)

    assert len(kept) == 3
    assert len(drops) == 2
    kept_prompts = {q["prompt_text"] for q in kept}
    assert kept_prompts == {"What is X?", "Define Y.", "Explain Z."}
    assert all(d.reason == REASON_BATCH_DUPLICATE for d in drops)
    assert {d.index for d in drops} == {3, 5}


async def test_dedup_removes_chunk_overlap_collisions(
    session_factory: async_sessionmaker[AsyncSession],
    fixture_data: dict,
) -> None:
    """Same prompt + same chunk_ids hash identical and collide;
    same prompt + DIFFERENT chunk_ids hash differently and survive
    (matches T5.3 SQL semantics: hash includes both)."""

    async with session_factory() as session:
        quiz = await get_quiz_for_authoring(session, fixture_data["target_quiz"])
        assert quiz is not None
        candidates = [
            _candidate("Same prompt", ["chunk-a"]),
            _candidate("Same prompt", ["chunk-a"]),
            _candidate("Same prompt", ["chunk-b"]),
        ]
        kept, drops = await discard_duplicates(session, quiz, candidates)

    assert len(kept) == 2
    assert {tuple(r["chunk_id"] for r in q["source_refs"]) for q in kept} == {
        ("chunk-a",),
        ("chunk-b",),
    }
    assert len(drops) == 1
    assert drops[0].reason == REASON_BATCH_DUPLICATE
    assert drops[0].index == 2


async def test_drops_capture_reasons(
    session_factory: async_sessionmaker[AsyncSession],
    fixture_data: dict,
) -> None:
    """Every drop must have a non-empty reason string and carry the
    original question payload (teacher-review surface needs it)."""

    async with session_factory() as session:
        quiz = await get_quiz_for_authoring(session, fixture_data["target_quiz"])
        assert quiz is not None
        candidates = [
            _candidate("Real question?", ["c1"]),
            _candidate("", ["c2"]),
            _candidate("Real question?", ["c1"]),
            {"prompt_text": "   ", "source_refs": []},
        ]
        kept, drops = await discard_duplicates(session, quiz, candidates)

    assert len(kept) == 1
    assert len(drops) == 3
    reasons = {d.reason for d in drops}
    assert reasons == {REASON_EMPTY_PROMPT, REASON_BATCH_DUPLICATE}
    for drop in drops:
        assert isinstance(drop.reason, str)
        assert drop.reason
        assert drop.question is candidates[drop.index - 1]


async def test_dedup_uses_existing_module_keys(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
    fixture_data: dict,
) -> None:
    """Pre-seed a question on the SIBLING quiz (same module). A
    candidate with identical prompt + source_refs must collide via
    the T5.3 cross-quiz path even though the target quiz itself is
    empty."""

    sibling_quiz = fixture_data["sibling_quiz"]
    sibling_q_id = fixture_data["sibling_q"]
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO quiz_questions ("
                "id, quiz_id, position, question_type, prompt_text, "
                "source_refs, review_status) VALUES "
                "(:id, :q, 1, 'multiple_choice', 'Pre-seeded prompt', "
                " CAST(:refs AS jsonb), 'approved')"
            ),
            {
                "id": sibling_q_id,
                "q": sibling_quiz,
                "refs": '[{"chunk_id": "shared-chunk"}]',
            },
        )

    async with session_factory() as session:
        quiz = await get_quiz_for_authoring(session, fixture_data["target_quiz"])
        assert quiz is not None
        candidates = [
            _candidate("Pre-seeded prompt", ["shared-chunk"]),
            _candidate("Brand new prompt", ["other-chunk"]),
        ]
        kept, drops = await discard_duplicates(session, quiz, candidates)

    assert len(kept) == 1
    assert kept[0]["prompt_text"] == "Brand new prompt"
    assert len(drops) == 1
    assert drops[0].reason == REASON_EXISTING_MODULE_DUPLICATE
    assert drops[0].index == 1


async def test_dedup_empty_input_returns_empty(
    session_factory: async_sessionmaker[AsyncSession],
    fixture_data: dict,
) -> None:
    """No DB round-trip path — guards against the inevitable empty-
    batch race during regenerate-one when all candidates were
    rejected upstream by validation."""

    async with session_factory() as session:
        quiz = await get_quiz_for_authoring(session, fixture_data["target_quiz"])
        assert quiz is not None
        kept, drops = await discard_duplicates(session, quiz, [])

    assert kept == []
    assert drops == []
