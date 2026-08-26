"""Integration tests for ``features.interviews.services`` (T6.11).

Covers the 4-capability split:

* :func:`authoring.create_interview_config` happy path.
* :func:`authoring.start_generation_run` ARQ enqueue contract
  (``run_interview_generation_task``).
* :func:`taking.start_session` idempotency + first-question seeding.
* :func:`taking.take_session_step` persistence, follow-up branch, 403
  boundary.
* :func:`taking.submit_session` ARQ enqueue contract
  (``evaluate_interview_session_task``).
* :func:`evaluation.evaluate_and_generate_report` persistence of
  outcome evaluations + gap-report row.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
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

import abridgeai.ai.models  # noqa: F401  -- register processing_jobs / generation_runs (audit + FK targets)
import abridgeai.features.access_control.models  # noqa: F401  -- register users/orgs FK targets
import abridgeai.features.courses.models  # noqa: F401  -- register courses/modules/lessons FK targets
import abridgeai.features.identity.models  # noqa: F401  -- register users FK target
import abridgeai.features.interviews.models  # noqa: F401  -- T6.1 registers interview_* tables
import abridgeai.features.materials.models  # noqa: F401  -- register learning_* tables
import abridgeai.features.quizzes.models  # noqa: F401  -- register quiz_attempts
from abridgeai.core.config import get_settings
from abridgeai.core.exceptions import ForbiddenError
from abridgeai.core.security import CurrentUser
from abridgeai.features.interviews.ai.stages.evaluation.outcome_verdicts import (
    OutcomeVerdict,
    build_outcome_verdicts,
)
from abridgeai.features.interviews.ai.stages.evaluation.rubric import (
    CriterionScore,
    ResponseEvaluation,
    RubricScores,
)
from abridgeai.features.interviews.ai.stages.gap_report.logic import GapReportDraft
from abridgeai.features.interviews.models import (
    InterviewConfig,
    InterviewSession,
)
from abridgeai.features.interviews.services import (
    authoring as authoring_service,
)
from abridgeai.features.interviews.services import (
    evaluation as evaluation_service,
)
from abridgeai.features.interviews.services import (
    taking as taking_service,
)
from abridgeai.features.interviews.queries import (
    authoring as authoring_queries,
)
from abridgeai.features.quizzes.api.public import GenerationRunDTO


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


def _actor(user_id: uuid.UUID) -> CurrentUser:
    return CurrentUser(user_id=user_id, session_id=uuid.uuid4())


def _force_legacy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the adaptive kill switch OFF so a test exercises the legacy path.

    Interviews now default to always-adaptive; tests that assert the legacy
    sequential mechanics (followup stage, plain question advance) must flip the
    emergency kill switch (``adaptive_interviewer_enabled=false``) to reach it.
    """
    legacy = get_settings().model_copy(update={"adaptive_interviewer_enabled": False})
    monkeypatch.setattr(taking_service, "get_settings", lambda: legacy)


class _CreatePayload:
    def __init__(self, **fields: Any) -> None:
        self._fields = fields

    def model_dump(self, exclude_unset: bool = False, mode: str | None = None) -> dict[str, Any]:
        del exclude_unset
        if mode == "json":
            return {k: str(v) if isinstance(v, uuid.UUID) else v for k, v in self._fields.items()}
        return dict(self._fields)


@pytest_asyncio.fixture
async def scenario(engine: AsyncEngine) -> AsyncIterator[dict[str, Any]]:
    org_id = uuid.uuid4()
    teacher_id = uuid.uuid4()
    student_id = uuid.uuid4()
    other_id = uuid.uuid4()
    course_id = uuid.uuid4()
    module_id = uuid.uuid4()
    suffix = org_id.hex[:8]

    async with engine.begin() as conn:
        await conn.execute(
            text("INSERT INTO organizations (id, slug, name) VALUES (:id, :slug, :name)"),
            {"id": org_id, "slug": f"is-{suffix}", "name": "Interview Svc Org"},
        )
        for uid, label in (
            (teacher_id, "teacher"),
            (student_id, "student"),
            (other_id, "other"),
        ):
            await conn.execute(
                text("INSERT INTO users (id, primary_email) VALUES (:id, :email)"),
                {"id": uid, "email": f"is-{label}-{suffix}@test.local"},
            )
        await conn.execute(
            text(
                "INSERT INTO courses "
                "(id, organization_id, owner_user_id, slug, title, status) "
                "VALUES (:id, :org, :owner, :slug, 'Interview Course', 'published')"
            ),
            {
                "id": course_id,
                "org": org_id,
                "owner": teacher_id,
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

    data = {
        "org_id": org_id,
        "teacher_id": teacher_id,
        "student_id": student_id,
        "other_id": other_id,
        "course_id": course_id,
        "module_id": module_id,
    }
    yield data

    async with engine.begin() as conn:
        await conn.execute(
            text(
                "DELETE FROM ai_model_calls WHERE generation_run_id IN "
                "(SELECT id FROM generation_runs WHERE module_id = :m)"
            ),
            {"m": module_id},
        )
        await conn.execute(
            text(
                "DELETE FROM gap_reports WHERE source_interview_session_id IN "
                "(SELECT id FROM interview_sessions WHERE interview_config_id IN "
                "(SELECT id FROM interview_configs WHERE module_id = :m))"
            ),
            {"m": module_id},
        )
        await conn.execute(
            text("DELETE FROM gap_reports WHERE student_id IN (:s, :o)"),
            {"s": student_id, "o": other_id},
        )
        await conn.execute(
            text(
                "DELETE FROM interview_outcome_evaluations WHERE session_id IN "
                "(SELECT id FROM interview_sessions WHERE interview_config_id IN "
                "(SELECT id FROM interview_configs WHERE module_id = :m))"
            ),
            {"m": module_id},
        )
        await conn.execute(
            text(
                "DELETE FROM interview_session_messages WHERE session_id IN "
                "(SELECT id FROM interview_sessions WHERE interview_config_id IN "
                "(SELECT id FROM interview_configs WHERE module_id = :m))"
            ),
            {"m": module_id},
        )
        await conn.execute(
            text(
                "DELETE FROM interview_session_questions WHERE session_id IN "
                "(SELECT id FROM interview_sessions WHERE interview_config_id IN "
                "(SELECT id FROM interview_configs WHERE module_id = :m))"
            ),
            {"m": module_id},
        )
        await conn.execute(
            text(
                "DELETE FROM interview_sessions WHERE interview_config_id IN "
                "(SELECT id FROM interview_configs WHERE module_id = :m)"
            ),
            {"m": module_id},
        )
        await conn.execute(
            text(
                "DELETE FROM interview_questions WHERE interview_config_id IN "
                "(SELECT id FROM interview_configs WHERE module_id = :m)"
            ),
            {"m": module_id},
        )
        await conn.execute(
            text(
                "DELETE FROM interview_outcomes WHERE interview_config_id IN "
                "(SELECT id FROM interview_configs WHERE module_id = :m)"
            ),
            {"m": module_id},
        )
        await conn.execute(
            text("DELETE FROM module_items WHERE module_id = :m"),
            {"m": module_id},
        )
        await conn.execute(
            text("DELETE FROM interview_configs WHERE module_id = :m"),
            {"m": module_id},
        )
        await conn.execute(
            text("DELETE FROM generation_runs WHERE module_id = :m"),
            {"m": module_id},
        )
        await conn.execute(text("DELETE FROM modules WHERE id = :m"), {"m": module_id})
        await conn.execute(
            text("DELETE FROM course_learning_outcomes WHERE course_id = :id"),
            {"id": course_id},
        )
        await conn.execute(text("DELETE FROM courses WHERE id = :id"), {"id": course_id})
        await conn.execute(
            text("DELETE FROM users WHERE id IN (:t, :s, :o)"),
            {"t": teacher_id, "s": student_id, "o": other_id},
        )
        await conn.execute(text("DELETE FROM organizations WHERE id = :id"), {"id": org_id})


async def _create_published_config(
    engine: AsyncEngine,
    *,
    course_id: uuid.UUID,
    module_id: uuid.UUID,
    teacher_id: uuid.UUID,
    questions: int = 2,
    outcomes: int = 1,
    max_attempts: int | None = None,
    cooldown_hours: int | None = None,
) -> dict[str, Any]:
    config_id = uuid.uuid4()
    question_ids = [uuid.uuid4() for _ in range(questions)]
    outcome_ids = [uuid.uuid4() for _ in range(outcomes)]
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO interview_configs (id, course_id, module_id, title, status, created_by, max_attempts, cooldown_hours, slug) VALUES (:id, :c, :m, 'Pub Interview', 'published', :t, :max_attempts, :cooldown_hours, 'slug-' || uuid_generate_v4()::text);"
            ),
            {
                "id": config_id,
                "c": course_id,
                "m": module_id,
                "t": teacher_id,
                "max_attempts": max_attempts,
                "cooldown_hours": cooldown_hours,
            },
        )
        for idx, qid in enumerate(question_ids, start=1):
            await conn.execute(
                text(
                    "INSERT INTO interview_questions "
                    "(id, interview_config_id, position, question_type, prompt_text, "
                    "review_status, ai_generated) "
                    "VALUES (:id, :cfg, :pos, 'technical', :prompt, 'approved', FALSE)"
                ),
                {
                    "id": qid,
                    "cfg": config_id,
                    "pos": idx,
                    "prompt": f"Question {idx}?",
                },
            )
        for idx, oid in enumerate(outcome_ids, start=1):
            await conn.execute(
                text(
                    "INSERT INTO interview_outcomes "
                    "(id, interview_config_id, position, outcome_text, outcome_type, "
                    "importance_weight) "
                    "VALUES (:id, :cfg, :pos, :txt, 'knowledge', 3)"
                ),
                {
                    "id": oid,
                    "cfg": config_id,
                    "pos": idx,
                    "txt": f"Outcome {idx}",
                },
            )
    return {
        "config_id": config_id,
        "question_ids": question_ids,
        "outcome_ids": outcome_ids,
    }


@pytest.mark.asyncio
async def test_create_interview_config_basic(
    session_factory: async_sessionmaker[AsyncSession],
    scenario: dict[str, Any],
) -> None:
    payload = _CreatePayload(
        title="Capstone Interview",
        module_id=scenario["module_id"],
    )
    async with session_factory() as session, session.begin():
        config = await authoring_service.create_interview_config(
            session, scenario["course_id"], payload, _actor(scenario["teacher_id"])
        )

    assert isinstance(config, InterviewConfig)
    assert config.title == "Capstone Interview"
    assert config.course_id == scenario["course_id"]
    assert config.module_id == scenario["module_id"]
    assert config.status == "draft"


@pytest.mark.asyncio
async def test_create_interview_config_seeds_course_outcomes(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
    scenario: dict[str, Any],
) -> None:
    # Seed the parent course with learning outcomes, then create a fresh
    # interview config: it should copy those course LOs in as its starting
    # rubric outcomes (no manual import needed).
    course_outcome_texts = ["Understand data warehouses", "Design a star schema"]
    async with engine.begin() as conn:
        for idx, txt in enumerate(course_outcome_texts, start=1):
            await conn.execute(
                text(
                    "INSERT INTO course_learning_outcomes "
                    "(id, course_id, position, outcome_text) "
                    "VALUES (:id, :course, :pos, :txt)"
                ),
                {
                    "id": uuid.uuid4(),
                    "course": scenario["course_id"],
                    "pos": idx,
                    "txt": txt,
                },
            )

    payload = _CreatePayload(
        title="Seeded Interview",
        module_id=scenario["module_id"],
    )
    async with session_factory() as session, session.begin():
        config = await authoring_service.create_interview_config(
            session, scenario["course_id"], payload, _actor(scenario["teacher_id"])
        )
        seeded = await authoring_queries.list_outcomes_for_config(session, config.id)

    assert [o.outcome_text for o in seeded] == course_outcome_texts
    assert [o.position for o in seeded] == [1, 2]
    assert all(o.outcome_type == "knowledge" and o.importance_weight == 3 for o in seeded)


@pytest.mark.asyncio
async def test_start_session_returns_existing_active_session(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
    scenario: dict[str, Any],
) -> None:
    seeded = await _create_published_config(
        engine,
        course_id=scenario["course_id"],
        module_id=scenario["module_id"],
        teacher_id=scenario["teacher_id"],
    )
    payload = _CreatePayload(input_mode="text")
    async with session_factory() as session, session.begin():
        first = await taking_service.start_session(
            session, seeded["config_id"], payload, _actor(scenario["student_id"])
        )
    async with session_factory() as session, session.begin():
        second = await taking_service.start_session(
            session, seeded["config_id"], payload, _actor(scenario["student_id"])
        )

    assert first.id == second.id
    assert first.attempt_number == 1


async def _finish_session(
    engine: AsyncEngine, session_id: uuid.UUID, *, status_: str, ended_at: Any
) -> None:
    """Mark a session terminal with a specific ``ended_at`` for gate tests."""
    async with engine.begin() as conn:
        await conn.execute(
            text("UPDATE interview_sessions SET status = :s, ended_at = :e WHERE id = :id"),
            {"s": status_, "e": ended_at, "id": session_id},
        )


async def _complete_onboarding(engine: AsyncEngine, session_id: uuid.UUID) -> None:
    """Fast-forward a session past onboarding so the answer path is reachable.

    The answer/step path gates on ``onboarding_stage == 'completed'`` (added in
    the onboarding-ceremony slice). These service tests exercise the *answering*
    mechanics, not onboarding, so they jump straight to the completed state —
    the same terminal state ``onboarding_service.respond`` reaches after the
    candidate confirms readiness (stage='completed' + assessment_started_at set).
    """
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "UPDATE interview_sessions "
                "SET onboarding_stage = 'completed', "
                "    assessment_started_at = COALESCE(assessment_started_at, NOW()) "
                "WHERE id = :id"
            ),
            {"id": session_id},
        )


@pytest.mark.asyncio
async def test_start_session_cooldown_blocks_new_attempt(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
    scenario: dict[str, Any],
) -> None:
    """FR-5.3 — a new attempt inside ``cooldown_hours`` is rejected."""
    seeded = await _create_published_config(
        engine,
        course_id=scenario["course_id"],
        module_id=scenario["module_id"],
        teacher_id=scenario["teacher_id"],
        cooldown_hours=24,
    )
    payload = _CreatePayload(input_mode="text")
    async with session_factory() as session, session.begin():
        first = await taking_service.start_session(
            session, seeded["config_id"], payload, _actor(scenario["student_id"])
        )
    # Finish it one hour ago — well inside the 24h cooldown.
    await _finish_session(
        engine,
        first.id,
        status_="completed",
        ended_at=datetime.now(UTC) - timedelta(hours=1),
    )

    with pytest.raises(taking_service.InterviewCooldownActive) as excinfo:
        async with session_factory() as session, session.begin():
            await taking_service.start_session(
                session, seeded["config_id"], payload, _actor(scenario["student_id"])
            )
    assert excinfo.value.retry_after > datetime.now(UTC)


@pytest.mark.asyncio
async def test_start_session_cooldown_elapsed_allows_new_attempt(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
    scenario: dict[str, Any],
) -> None:
    """FR-5.3 — once the cooldown has elapsed a new attempt is allowed."""
    seeded = await _create_published_config(
        engine,
        course_id=scenario["course_id"],
        module_id=scenario["module_id"],
        teacher_id=scenario["teacher_id"],
        cooldown_hours=2,
    )
    payload = _CreatePayload(input_mode="text")
    async with session_factory() as session, session.begin():
        first = await taking_service.start_session(
            session, seeded["config_id"], payload, _actor(scenario["student_id"])
        )
    # Finished 3h ago — the 2h cooldown has lapsed.
    await _finish_session(
        engine,
        first.id,
        status_="completed",
        ended_at=datetime.now(UTC) - timedelta(hours=3),
    )

    async with session_factory() as session, session.begin():
        second = await taking_service.start_session(
            session, seeded["config_id"], payload, _actor(scenario["student_id"])
        )
    assert second.id != first.id
    assert second.attempt_number == 2


@pytest.mark.asyncio
async def test_start_session_max_attempts_blocks_new_attempt(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
    scenario: dict[str, Any],
) -> None:
    """FR-5.3 — once ``max_attempts`` terminal sessions exist, block."""
    seeded = await _create_published_config(
        engine,
        course_id=scenario["course_id"],
        module_id=scenario["module_id"],
        teacher_id=scenario["teacher_id"],
        max_attempts=1,
    )
    payload = _CreatePayload(input_mode="text")
    async with session_factory() as session, session.begin():
        first = await taking_service.start_session(
            session, seeded["config_id"], payload, _actor(scenario["student_id"])
        )
    await _finish_session(
        engine,
        first.id,
        status_="completed",
        ended_at=datetime.now(UTC) - timedelta(days=30),
    )

    with pytest.raises(taking_service.InterviewMaxAttemptsReached) as excinfo:
        async with session_factory() as session, session.begin():
            await taking_service.start_session(
                session, seeded["config_id"], payload, _actor(scenario["student_id"])
            )
    assert excinfo.value.max_attempts == 1


@pytest.mark.asyncio
async def test_start_session_no_cooldown_when_unset(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
    scenario: dict[str, Any],
) -> None:
    """FR-5.3 — NULL cooldown/max_attempts leaves retakes unrestricted."""
    seeded = await _create_published_config(
        engine,
        course_id=scenario["course_id"],
        module_id=scenario["module_id"],
        teacher_id=scenario["teacher_id"],
    )
    payload = _CreatePayload(input_mode="text")
    async with session_factory() as session, session.begin():
        first = await taking_service.start_session(
            session, seeded["config_id"], payload, _actor(scenario["student_id"])
        )
    await _finish_session(
        engine,
        first.id,
        status_="completed",
        ended_at=datetime.now(UTC),
    )
    # Immediately retaking is fine — no gate configured.
    async with session_factory() as session, session.begin():
        second = await taking_service.start_session(
            session, seeded["config_id"], payload, _actor(scenario["student_id"])
        )
    assert second.id != first.id
    assert second.attempt_number == 2


@pytest.mark.asyncio
async def test_start_session_creates_first_session_question_row(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
    scenario: dict[str, Any],
) -> None:
    seeded = await _create_published_config(
        engine,
        course_id=scenario["course_id"],
        module_id=scenario["module_id"],
        teacher_id=scenario["teacher_id"],
    )
    payload = _CreatePayload(input_mode="text")
    async with session_factory() as session, session.begin():
        started = await taking_service.start_session(
            session, seeded["config_id"], payload, _actor(scenario["student_id"])
        )

    async with engine.begin() as conn:
        rows = (
            await conn.execute(
                text(
                    "SELECT sequence_no, interview_question_id "
                    "FROM interview_session_questions WHERE session_id = :s "
                    "ORDER BY sequence_no"
                ),
                {"s": started.id},
            )
        ).all()

    assert len(rows) == 1
    assert rows[0].sequence_no == 1
    assert rows[0].interview_question_id == seeded["question_ids"][0]


@pytest.mark.asyncio
async def test_take_step_persists_answer_and_advances(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
    scenario: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seeded = await _create_published_config(
        engine,
        course_id=scenario["course_id"],
        module_id=scenario["module_id"],
        teacher_id=scenario["teacher_id"],
    )
    monkeypatch.setattr(
        "abridgeai.features.interviews.services.taking.maybe_generate_followup",
        AsyncMock(return_value=None),
    )
    # Interviews now default to always-adaptive; this test exercises the LEGACY
    # sequential mechanics, which run only when the emergency kill switch is off.
    _force_legacy(monkeypatch)
    payload = _CreatePayload(input_mode="text")
    async with session_factory() as session, session.begin():
        started = await taking_service.start_session(
            session, seeded["config_id"], payload, _actor(scenario["student_id"])
        )
    # The answer path gates on completed onboarding; this test exercises the
    # answering mechanics, so fast-forward past setup.
    await _complete_onboarding(engine, started.id)

    async with session_factory() as session, session.begin():
        result = await taking_service.take_session_step(
            session,
            started.id,
            "A thorough answer about the topic.",
            _actor(scenario["student_id"]),
        )

    # On a plain advance the step now carries a persona-aware transition
    # signpost (Natural Interview Transitions) rather than None — the meaningful
    # assertion is that it's a next-question transition, not a probing follow-up.
    assert result["transition_target"] == "next_question"
    assert result["is_finished"] is False
    assert result["next_question"] is not None
    assert result["next_question"].id == seeded["question_ids"][1]

    async with engine.begin() as conn:
        msg_count = (
            await conn.execute(
                text(
                    "SELECT COUNT(*) AS c FROM interview_session_messages "
                    "WHERE session_id = :s AND role = 'user'"
                ),
                {"s": started.id},
            )
        ).scalar_one()
        seq_count = (
            await conn.execute(
                text("SELECT COUNT(*) AS c FROM interview_session_questions WHERE session_id = :s"),
                {"s": started.id},
            )
        ).scalar_one()
    assert msg_count == 1
    assert seq_count == 2


@pytest.mark.asyncio
async def test_take_step_returns_followup_when_shallow(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
    scenario: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seeded = await _create_published_config(
        engine,
        course_id=scenario["course_id"],
        module_id=scenario["module_id"],
        teacher_id=scenario["teacher_id"],
    )
    monkeypatch.setattr(
        "abridgeai.features.interviews.services.taking.maybe_generate_followup",
        AsyncMock(return_value="Can you elaborate?"),
    )
    # Legacy sequential followup mechanics — reachable only via the kill switch.
    _force_legacy(monkeypatch)
    payload = _CreatePayload(input_mode="text")
    async with session_factory() as session, session.begin():
        started = await taking_service.start_session(
            session, seeded["config_id"], payload, _actor(scenario["student_id"])
        )
    # The answer path gates on completed onboarding; this test exercises the
    # follow-up mechanics, so fast-forward past setup.
    await _complete_onboarding(engine, started.id)

    async with session_factory() as session, session.begin():
        result = await taking_service.take_session_step(
            session,
            started.id,
            "Yes.",
            _actor(scenario["student_id"]),
        )

    assert result["followup_text"] == "Can you elaborate?"
    assert result["is_finished"] is False
    assert result["next_question"] is None

    async with engine.begin() as conn:
        seq_count = (
            await conn.execute(
                text("SELECT COUNT(*) AS c FROM interview_session_questions WHERE session_id = :s"),
                {"s": started.id},
            )
        ).scalar_one()
        followup_count = (
            await conn.execute(
                text(
                    "SELECT COUNT(*) AS c FROM interview_session_messages "
                    "WHERE session_id = :s AND role = 'ai' "
                    "AND metadata_json->>'kind' = 'followup'"
                ),
                {"s": started.id},
            )
        ).scalar_one()
    assert seq_count == 1
    assert followup_count == 1


@pytest.mark.asyncio
async def test_take_step_403_for_other_user_session(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
    scenario: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seeded = await _create_published_config(
        engine,
        course_id=scenario["course_id"],
        module_id=scenario["module_id"],
        teacher_id=scenario["teacher_id"],
    )
    monkeypatch.setattr(
        "abridgeai.features.interviews.services.taking.maybe_generate_followup",
        AsyncMock(return_value=None),
    )
    payload = _CreatePayload(input_mode="text")
    async with session_factory() as session, session.begin():
        started = await taking_service.start_session(
            session, seeded["config_id"], payload, _actor(scenario["student_id"])
        )

    async with session_factory() as session, session.begin():
        with pytest.raises(ForbiddenError):
            await taking_service.take_session_step(
                session,
                started.id,
                "answer",
                _actor(scenario["other_id"]),
            )


@pytest.mark.asyncio
async def test_submit_session_marks_completed_and_enqueues_eval(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
    scenario: dict[str, Any],
) -> None:
    seeded = await _create_published_config(
        engine,
        course_id=scenario["course_id"],
        module_id=scenario["module_id"],
        teacher_id=scenario["teacher_id"],
    )
    payload = _CreatePayload(input_mode="text")
    async with session_factory() as session, session.begin():
        started = await taking_service.start_session(
            session, seeded["config_id"], payload, _actor(scenario["student_id"])
        )
    # Reaching the assessment is what makes a run gradeable; a session still in
    # onboarding is abandoned and never enqueued.
    await _complete_onboarding(engine, started.id)

    arq_pool = SimpleNamespace(enqueue_job=AsyncMock(return_value=None))
    async with session_factory() as session:
        finished = await taking_service.submit_session(
            session, started.id, _actor(scenario["student_id"]), arq_pool=arq_pool
        )

    assert finished.status == "completed"
    assert finished.ended_at is not None
    arq_pool.enqueue_job.assert_awaited_once()
    invocation = arq_pool.enqueue_job.await_args
    assert invocation.args[0] == "evaluate_interview_session_task"
    assert invocation.args[1] == scenario["student_id"]
    assert invocation.args[2] == started.id


@pytest.mark.asyncio
async def test_submit_enqueues_eval(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
    scenario: dict[str, Any],
) -> None:
    """Plan §6.11 explicit acceptance test."""
    seeded = await _create_published_config(
        engine,
        course_id=scenario["course_id"],
        module_id=scenario["module_id"],
        teacher_id=scenario["teacher_id"],
    )
    payload = _CreatePayload(input_mode="text")
    async with session_factory() as session, session.begin():
        started = await taking_service.start_session(
            session, seeded["config_id"], payload, _actor(scenario["student_id"])
        )
    await _complete_onboarding(engine, started.id)

    arq_pool = SimpleNamespace(enqueue_job=AsyncMock(return_value=None))
    async with session_factory() as session:
        await taking_service.submit_session(
            session, started.id, _actor(scenario["student_id"]), arq_pool=arq_pool
        )

    arq_pool.enqueue_job.assert_awaited_once()
    assert arq_pool.enqueue_job.await_args.args[0] == "evaluate_interview_session_task"


@pytest.mark.asyncio
async def test_start_generation_run_enqueues_arq(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
    scenario: dict[str, Any],
) -> None:
    seeded = await _create_published_config(
        engine,
        course_id=scenario["course_id"],
        module_id=scenario["module_id"],
        teacher_id=scenario["teacher_id"],
        questions=0,
        outcomes=0,
    )
    request = _CreatePayload(
        mode="topic",
        course_id=scenario["course_id"],
        module_id=scenario["module_id"],
        question_count=5,
    )
    arq_pool = SimpleNamespace(enqueue_job=AsyncMock(return_value=None))

    async with session_factory() as session:
        run = await authoring_service.start_generation_run(
            session,
            seeded["config_id"],
            request,
            _actor(scenario["teacher_id"]),
            arq_pool=arq_pool,
        )

    assert isinstance(run, GenerationRunDTO)
    assert run.status == "pending"
    assert run.module_id == scenario["module_id"]
    assert run.config_json["interview_config_id"] == str(seeded["config_id"])
    arq_pool.enqueue_job.assert_awaited_once()
    invocation = arq_pool.enqueue_job.await_args
    assert invocation.args[0] == "run_interview_generation_task"
    assert invocation.args[1] == scenario["teacher_id"]
    assert invocation.args[2] == run.id


@pytest.mark.asyncio
async def test_evaluate_and_generate_report_persists_outcome_evaluations_and_gap_report(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
    scenario: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seeded = await _create_published_config(
        engine,
        course_id=scenario["course_id"],
        module_id=scenario["module_id"],
        teacher_id=scenario["teacher_id"],
        outcomes=2,
    )

    payload = _CreatePayload(input_mode="text")
    async with session_factory() as session, session.begin():
        started = await taking_service.start_session(
            session, seeded["config_id"], payload, _actor(scenario["student_id"])
        )
    session_id = started.id
    # The evaluator refuses a run that never reached the assessment, so this
    # session has to be past onboarding for the persistence assertions below.
    await _complete_onboarding(engine, session_id)

    fake_question_id = seeded["question_ids"][0]
    rubric = RubricScores(
        response_evaluations=[
            ResponseEvaluation(
                session_question_id=fake_question_id,
                criterion_scores=[
                    CriterionScore(
                        criterion="technical_accuracy",
                        score=4.0,
                        justification="Solid.",
                    ),
                ],
            )
        ],
        aggregated={"technical_accuracy": 4.0},
        total_score=80.0,
    )
    draft = GapReportDraft(
        discrepancy_score=10.0,
        theory_score_avg=90.0,
        practice_score=80.0,
        strengths=["Communicates clearly"],
        weaknesses=["Skims complexity"],
        study_plan=[],
        student_summary="You did well; revisit complexity analysis.",
        teacher_summary="Strong overall; gap on complexity reasoning.",
        report_json={"discrepancy_score": 10.0},
    )

    monkeypatch.setattr(
        "abridgeai.features.interviews.services.evaluation.evaluate_session",
        AsyncMock(return_value=rubric),
    )
    # §4.3 gate: per-outcome verdicts decide pass/fail. Both outcomes met →
    # with NULL min_outcomes_to_pass (all-met rule) the session passes.
    outcome_verdicts = build_outcome_verdicts(
        [
            OutcomeVerdict(outcome_id=oid, met=True, reasoning="Demonstrated.", evidence="quote")
            for oid in seeded["outcome_ids"]
        ]
    )
    monkeypatch.setattr(
        "abridgeai.features.interviews.services.evaluation.evaluate_outcomes",
        AsyncMock(return_value=outcome_verdicts),
    )
    monkeypatch.setattr(
        "abridgeai.features.interviews.services.evaluation.generate_gap_report",
        AsyncMock(return_value=draft),
    )

    async with session_factory() as session:
        await evaluation_service.evaluate_and_generate_report(session, session_id)

    async with session_factory() as session:
        evals = (
            (
                await session.execute(
                    text(
                        "SELECT outcome_id, verdict_met FROM interview_outcome_evaluations "
                        "WHERE session_id = :s"
                    ),
                    {"s": session_id},
                )
            )
            .mappings()
            .all()
        )
        report = (
            (
                await session.execute(
                    text(
                        "SELECT student_summary, teacher_summary, report_json "
                        "FROM gap_reports WHERE source_interview_session_id = :s"
                    ),
                    {"s": session_id},
                )
            )
            .mappings()
            .one_or_none()
        )
        refreshed = await session.get(InterviewSession, session_id)

    assert len(evals) == 2
    assert all(row["verdict_met"] is True for row in evals)
    assert report is not None
    assert report["student_summary"] == "You did well; revisit complexity analysis."
    assert report["teacher_summary"] == "Strong overall; gap on complexity reasoning."
    assert refreshed is not None
    assert refreshed.internal_summary_json["total_score"] == 80.0
    assert refreshed.internal_summary_json["outcomes_met"] == 2
    assert refreshed.internal_summary_json["outcomes_total"] == 2
    assert refreshed.pass_verdict is True


@pytest.mark.asyncio
async def test_evaluate_and_generate_report_fails_when_outcomes_not_met(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
    scenario: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """§4.3: when not enough outcomes are met, the session fails — regardless of
    the rubric score (which no longer gates pass)."""
    seeded = await _create_published_config(
        engine,
        course_id=scenario["course_id"],
        module_id=scenario["module_id"],
        teacher_id=scenario["teacher_id"],
        outcomes=2,
    )
    payload = _CreatePayload(input_mode="text")
    async with session_factory() as session, session.begin():
        started = await taking_service.start_session(
            session, seeded["config_id"], payload, _actor(scenario["student_id"])
        )
    session_id = started.id
    await _complete_onboarding(engine, session_id)

    # High rubric score (would have passed under old >=60 logic)...
    rubric = RubricScores(
        response_evaluations=[], aggregated={"technical_accuracy": 5.0}, total_score=95.0
    )
    # ...but only 1 of 2 outcomes met, and NULL threshold requires ALL → FAIL.
    oids = seeded["outcome_ids"]
    verdicts = build_outcome_verdicts(
        [
            OutcomeVerdict(outcome_id=oids[0], met=True, reasoning="ok", evidence=None),
            OutcomeVerdict(outcome_id=oids[1], met=False, reasoning="missed", evidence=None),
        ]
    )
    draft = GapReportDraft(
        discrepancy_score=0.0,
        theory_score_avg=0.0,
        practice_score=95.0,
        strengths=[],
        weaknesses=[],
        study_plan=[],
        student_summary="s",
        teacher_summary="t",
        report_json={},
    )
    monkeypatch.setattr(
        "abridgeai.features.interviews.services.evaluation.evaluate_session",
        AsyncMock(return_value=rubric),
    )
    monkeypatch.setattr(
        "abridgeai.features.interviews.services.evaluation.evaluate_outcomes",
        AsyncMock(return_value=verdicts),
    )
    monkeypatch.setattr(
        "abridgeai.features.interviews.services.evaluation.generate_gap_report",
        AsyncMock(return_value=draft),
    )

    async with session_factory() as session:
        await evaluation_service.evaluate_and_generate_report(session, session_id)

    async with session_factory() as session:
        evals = (
            (
                await session.execute(
                    text(
                        "SELECT outcome_id, verdict_met FROM interview_outcome_evaluations "
                        "WHERE session_id = :s ORDER BY outcome_id"
                    ),
                    {"s": session_id},
                )
            )
            .mappings()
            .all()
        )
        refreshed = await session.get(InterviewSession, session_id)

    # Distinct per-outcome verdicts (NOT all-equal) — the core §4.3 fix.
    assert {row["verdict_met"] for row in evals} == {True, False}
    assert refreshed is not None
    assert refreshed.pass_verdict is False  # high rubric did NOT force a pass
