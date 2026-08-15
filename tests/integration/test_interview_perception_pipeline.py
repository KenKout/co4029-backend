"""Integration tests for the adaptive-interviewer perception pipeline (Phase 2+3).

Exercises ``orchestrator.pipeline.perceive_turn`` against a live Postgres
session row with a STUBBED LLM gateway (no real model calls), proving:

* a genuine answer → intent=answer + a persisted AnswerAnalysis, version bumped,
* a non-academic intent (repeat request) → NO analysis, still persisted,
* a deterministic rule short-circuits the gateway entirely for obvious cases,
* an idempotent replay (same turn key) → no re-run, no version bump,
* provisional outcome coverage accrues supporting turn ids from evidence.

Audio is NOT involved anywhere here; nothing in this slice touches TTS/STT.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

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

import abridgeai.ai.models  # noqa: F401
import abridgeai.features.access_control.models  # noqa: F401
import abridgeai.features.courses.models  # noqa: F401
import abridgeai.features.identity.models  # noqa: F401
import abridgeai.features.interviews.models  # noqa: F401
import abridgeai.features.materials.models  # noqa: F401
import abridgeai.features.quizzes.models  # noqa: F401
from abridgeai.core.config import get_settings
from abridgeai.features.interviews.orchestrator import pipeline
from abridgeai.features.interviews.orchestrator import repository as repo
from abridgeai.features.interviews.orchestrator.intent import StudentIntent


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
async def live_session(engine: AsyncEngine) -> AsyncIterator[dict[str, Any]]:
    org_id = uuid.uuid4()
    teacher_id = uuid.uuid4()
    student_id = uuid.uuid4()
    course_id = uuid.uuid4()
    module_id = uuid.uuid4()
    config_id = uuid.uuid4()
    session_id = uuid.uuid4()
    suffix = org_id.hex[:8]

    async with engine.begin() as conn:
        await conn.execute(
            text("INSERT INTO organizations (id, slug, name) VALUES (:id, :slug, :name)"),
            {"id": org_id, "slug": f"pp-{suffix}", "name": "Perception Org"},
        )
        await conn.execute(
            text("INSERT INTO users (id, primary_email) VALUES (:id, :email)"),
            {"id": student_id, "email": f"pp-student-{suffix}@test.local"},
        )
        await conn.execute(
            text("INSERT INTO users (id, primary_email) VALUES (:id, :email)"),
            {"id": teacher_id, "email": f"pp-teacher-{suffix}@test.local"},
        )
        await conn.execute(
            text(
                "INSERT INTO courses (id, organization_id, owner_user_id, slug, title, status) "
                "VALUES (:id, :org, :owner, :slug, 'PP Course', 'published')"
            ),
            {"id": course_id, "org": org_id, "owner": teacher_id, "slug": f"pp-course-{suffix}"},
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
                "INSERT INTO interview_configs "
                "(id, course_id, module_id, title, status, created_by) "
                "VALUES (:id, :course, :module, 'PP Interview', 'published', :teacher)"
            ),
            {"id": config_id, "course": course_id, "module": module_id, "teacher": teacher_id},
        )
        await conn.execute(
            text(
                "INSERT INTO interview_sessions "
                "(id, interview_config_id, student_id, attempt_number, status, input_mode) "
                "VALUES (:id, :config, :student, 1, 'in_progress', 'text')"
            ),
            {"id": session_id, "config": config_id, "student": student_id},
        )

    yield {"session_id": session_id, "config_id": config_id, "student_id": student_id}

    async with engine.begin() as conn:
        await conn.execute(
            text("DELETE FROM interview_runtime_states WHERE session_id = :s"),
            {"s": session_id},
        )
        await conn.execute(text("DELETE FROM interview_sessions WHERE id = :s"), {"s": session_id})
        await conn.execute(text("DELETE FROM interview_configs WHERE id = :c"), {"c": config_id})
        await conn.execute(text("DELETE FROM modules WHERE id = :m"), {"m": module_id})
        await conn.execute(text("DELETE FROM courses WHERE id = :c"), {"c": course_id})
        await conn.execute(
            text("DELETE FROM users WHERE id IN (:s, :t)"),
            {"s": student_id, "t": teacher_id},
        )
        await conn.execute(text("DELETE FROM organizations WHERE id = :o"), {"o": org_id})


def _gateway(payloads: list[dict[str, object]]) -> SimpleNamespace:
    """A stub gateway whose generate_json returns queued payloads in order."""
    results = [SimpleNamespace(content_json=p) for p in payloads]
    return SimpleNamespace(generate_json=AsyncMock(side_effect=results))


@pytest.mark.asyncio
async def test_genuine_answer_persists_analysis_and_bumps_version(
    session_factory: async_sessionmaker[AsyncSession], live_session: dict[str, Any]
) -> None:
    session_id = live_session["session_id"]
    outcome_id = str(uuid.uuid4())
    turn_id = str(uuid.uuid4())

    # First payload = intent classification, second = answer analysis.
    gateway = _gateway(
        [
            {"intent": "answer", "confidence": 0.95, "rationale": "content response"},
            {
                "relevance": "relevant",
                "completeness": "partial",
                "correctness": "mostly_correct",
                "specificity": "specific",
                "has_concrete_example": True,
                "identified_concepts": ["fact table"],
                "evidence": [
                    {
                        "outcome_id": outcome_id,
                        "turn_id": turn_id,
                        "evidence_type": "partially_supports",
                        "summary": "mentions measurable business data",
                        "confidence": 0.7,
                    }
                ],
                "recommended_probe_type": "ask_for_example",
                "confidence": 0.7,
            },
        ]
    )

    async with session_factory() as db:
        result = await pipeline.perceive_turn(
            db,
            session_id=session_id,
            question_text="What is a fact table?",
            student_utterance="A fact table stores measurable business metrics like sales.",
            turn_id=turn_id,
            outcome_id=outcome_id,
            turn_idempotency_key="turn-1",
            gateway=gateway,
        )
        await db.commit()

    assert result.intent.intent is StudentIntent.ANSWER
    assert result.analysis is not None
    assert result.was_duplicate is False
    assert result.state_version == 1

    # Coverage accrued the supporting turn id from the evidence.
    async with session_factory() as db:
        loaded = await repo.load_or_init(db, session_id)
    cov = loaded.data.outcome_coverage.get(outcome_id)
    assert cov is not None
    assert turn_id in cov.supporting_turn_ids
    assert cov.evidence_count == 1


@pytest.mark.asyncio
async def test_repeat_request_is_not_analyzed(
    session_factory: async_sessionmaker[AsyncSession], live_session: dict[str, Any]
) -> None:
    session_id = live_session["session_id"]
    # Deterministic rule should fire for "can you repeat the question" and
    # short-circuit the gateway entirely — so an empty gateway is fine.
    gateway = _gateway([])

    async with session_factory() as db:
        result = await pipeline.perceive_turn(
            db,
            session_id=session_id,
            question_text="What is a fact table?",
            student_utterance="Sorry, could you repeat the question?",
            turn_id=str(uuid.uuid4()),
            turn_idempotency_key="turn-repeat",
            gateway=gateway,
        )
        await db.commit()

    assert result.intent.intent is StudentIntent.ASK_TO_REPEAT
    assert result.analysis is None
    gateway.generate_json.assert_not_awaited()


@pytest.mark.asyncio
async def test_duplicate_turn_is_noop(
    session_factory: async_sessionmaker[AsyncSession], live_session: dict[str, Any]
) -> None:
    session_id = live_session["session_id"]
    turn_id = str(uuid.uuid4())
    gateway = _gateway(
        [
            {"intent": "answer", "confidence": 0.9, "rationale": "content"},
            {"relevance": "relevant", "confidence": 0.4},
        ]
    )

    async with session_factory() as db:
        first = await pipeline.perceive_turn(
            db,
            session_id=session_id,
            question_text="Q",
            student_utterance="A real answer about indexing strategies.",
            turn_id=turn_id,
            turn_idempotency_key="dup-key",
            gateway=gateway,
        )
        await db.commit()

    assert first.was_duplicate is False
    assert first.state_version == 1

    # Replay with the SAME idempotency key — must not re-run the gateway or bump.
    gateway.generate_json.reset_mock()
    async with session_factory() as db:
        replay = await pipeline.perceive_turn(
            db,
            session_id=session_id,
            question_text="Q",
            student_utterance="A real answer about indexing strategies.",
            turn_id=turn_id,
            turn_idempotency_key="dup-key",
            gateway=gateway,
        )
        await db.commit()

    assert replay.was_duplicate is True
    assert replay.state_version == 1
    gateway.generate_json.assert_not_awaited()
