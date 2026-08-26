"""Integration tests for the quizzes aggregate ORM models (T5.1).

Per plan §5422-5428: ``QuizQuestion.expected_response_time_ms`` MUST
accept NULL while a quiz is in draft. The publish gate (T7.5.9)
enforces NOT NULL at publish time, NOT at the column. Migration 0007
flips the baseline ``NOT NULL`` to ``NULL`` for that purpose.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest
import pytest_asyncio
from alembic import command
from alembic.config import Config
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from abridgeai.core.config import get_settings

NEW_HEAD = "0007_quiz_renames_and_sr_fields"
PRIOR_HEAD = "0006_courses_unlock_config"

BACKEND_ROOT = Path(__file__).resolve().parents[2]


def _async_url(database_url: str) -> str:
    if "+psycopg_async" in database_url:
        return database_url
    if database_url.startswith("postgresql+psycopg://"):
        return database_url.replace("postgresql+psycopg://", "postgresql+psycopg_async://", 1)
    if database_url.startswith("postgresql://"):
        return database_url.replace("postgresql://", "postgresql+psycopg_async://", 1)
    return database_url


def _alembic_config() -> Config:
    cfg = Config(str(BACKEND_ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(BACKEND_ROOT / "migrations"))
    return cfg


@pytest_asyncio.fixture
async def engine() -> AsyncEngine:
    eng = create_async_engine(_async_url(get_settings().database_url), pool_pre_ping=True)
    yield eng
    await eng.dispose()


@pytest.fixture
def alembic_cfg() -> Config:
    return _alembic_config()


@pytest_asyncio.fixture
async def at_new_head(alembic_cfg: Config) -> None:
    command.upgrade(alembic_cfg, "head")  # never leave the shared DB below real head
    yield


async def _seed_quiz_scaffold(engine: AsyncEngine) -> tuple[uuid.UUID, uuid.UUID]:
    org_id = uuid.uuid4()
    user_id = uuid.uuid4()
    course_id = uuid.uuid4()
    module_id = uuid.uuid4()
    quiz_id = uuid.uuid4()

    async with engine.begin() as conn:
        await conn.execute(
            text("INSERT INTO organizations (id, slug, name) VALUES (:id, :slug, :name)"),
            {
                "id": org_id,
                "slug": f"t5-{org_id.hex[:8]}",
                "name": "T5.1 Org",
            },
        )
        await conn.execute(
            text("INSERT INTO users (id, primary_email) VALUES (:id, :email)"),
            {"id": user_id, "email": f"t5-{user_id.hex[:8]}@test.local"},
        )
        await conn.execute(
            text(
                "INSERT INTO courses (id, organization_id, owner_user_id, slug, title) "
                "VALUES (:id, :org, :owner, :slug, :title)"
            ),
            {
                "id": course_id,
                "org": org_id,
                "owner": user_id,
                "slug": f"course-{course_id.hex[:8]}",
                "title": "T5.1 Course",
            },
        )
        await conn.execute(
            text(
                "INSERT INTO modules (id, course_id, title, position) "
                "VALUES (:id, :course, :title, :pos)"
            ),
            {"id": module_id, "course": course_id, "title": "M", "pos": 1},
        )
        await conn.execute(
            text(
                "INSERT INTO quizzes (id, course_id, module_id, title, status, slug) VALUES (:id, :course, :module, :title, 'draft', 'slug-' || uuid_generate_v4()::text);"
            ),
            {
                "id": quiz_id,
                "course": course_id,
                "module": module_id,
                "title": "T5.1 Draft Quiz",
            },
        )
    return quiz_id, user_id


async def test_t_exp_nullable_in_draft(
    at_new_head: None,
    engine: AsyncEngine,
) -> None:
    quiz_id, _user_id = await _seed_quiz_scaffold(engine)
    question_id = uuid.uuid4()

    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO quiz_questions ("
                "id, quiz_id, position, question_type, prompt_text, "
                "expected_response_time_ms, source_refs, review_status"
                ") VALUES ("
                ":id, :quiz, 1, 'multiple_choice', 'Q?', "
                "NULL, '[]'::jsonb, 'pending'"
                ")"
            ),
            {"id": question_id, "quiz": quiz_id},
        )

    async with engine.connect() as conn:
        result = await conn.execute(
            text(
                "SELECT expected_response_time_ms, question_type, source_refs "
                "FROM quiz_questions WHERE id = :id"
            ),
            {"id": question_id},
        )
        row = result.one()
    assert row[0] is None, "T_exp must be NULL in draft (plan §5422-5428)"
    assert row[1] == "multiple_choice"
    assert list(row[2]) == []


async def test_mcq_value_rejected_after_migration(
    at_new_head: None,
    engine: AsyncEngine,
) -> None:
    quiz_id, _user_id = await _seed_quiz_scaffold(engine)
    question_id = uuid.uuid4()

    with pytest.raises(Exception) as excinfo:
        async with engine.begin() as conn:
            await conn.execute(
                text(
                    "INSERT INTO quiz_questions ("
                    "id, quiz_id, position, question_type, prompt_text, "
                    "expected_response_time_ms, source_refs"
                    ") VALUES ("
                    ":id, :quiz, 2, 'mcq', 'Q?', NULL, '[]'::jsonb"
                    ")"
                ),
                {"id": question_id, "quiz": quiz_id},
            )
    assert "ck_quiz_questions_question_type" in str(excinfo.value) or "question_type" in str(
        excinfo.value
    )


async def test_source_refs_no_json_suffix_in_db(
    at_new_head: None,
    engine: AsyncEngine,
) -> None:
    async with engine.connect() as conn:
        result = await conn.execute(
            text(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'quiz_questions' "
                "AND column_name IN ('source_refs', 'source_refs_json')"
            )
        )
        cols = {row[0] for row in result}
    assert "source_refs" in cols
    assert "source_refs_json" not in cols, "§C1: source_refs_json was renamed by migration 0007"


async def test_t_actual_ms_renamed_in_db(
    at_new_head: None,
    engine: AsyncEngine,
) -> None:
    async with engine.connect() as conn:
        result = await conn.execute(
            text(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'quiz_attempt_answers' "
                "AND column_name IN ('t_actual_ms', 'response_time_ms')"
            )
        )
        cols = {row[0] for row in result}
    assert "t_actual_ms" in cols
    assert "response_time_ms" not in cols, (
        "plan §5371: response_time_ms was renamed by migration 0007"
    )


async def test_migration_round_trip(
    alembic_cfg: Config,
    engine: AsyncEngine,
) -> None:
    command.upgrade(alembic_cfg, "head")  # never leave the shared DB below real head
    command.downgrade(alembic_cfg, PRIOR_HEAD)

    async with engine.connect() as conn:
        result = await conn.execute(
            text(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'quiz_questions' "
                "AND column_name IN "
                "('expected_response_ms', 'expected_response_time_ms', "
                "'source_refs', 'source_refs_json')"
            )
        )
        cols_at_prior = {row[0] for row in result}
    assert "expected_response_ms" in cols_at_prior
    assert "source_refs_json" in cols_at_prior
    assert "expected_response_time_ms" not in cols_at_prior
    assert "source_refs" not in cols_at_prior

    command.upgrade(alembic_cfg, "head")  # never leave the shared DB below real head

    async with engine.connect() as conn:
        result = await conn.execute(
            text(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'quiz_questions' "
                "AND column_name IN "
                "('expected_response_ms', 'expected_response_time_ms', "
                "'source_refs', 'source_refs_json')"
            )
        )
        cols_at_new = {row[0] for row in result}
    assert "expected_response_time_ms" in cols_at_new
    assert "source_refs" in cols_at_new
    assert "expected_response_ms" not in cols_at_new
    assert "source_refs_json" not in cols_at_new
