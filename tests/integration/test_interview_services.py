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

import hashlib
import json
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
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

import abridgeai.ai.models  # noqa: F401  -- register processing_jobs / generation_runs (audit + FK targets)
import abridgeai.features.access_control.models  # noqa: F401  -- register users/orgs FK targets
import abridgeai.features.courses.models  # noqa: F401  -- register courses/modules/lessons FK targets
import abridgeai.features.identity.models  # noqa: F401  -- register users FK target
import abridgeai.features.interviews.models  # noqa: F401  -- T6.1 registers interview_* tables
import abridgeai.features.materials.models  # noqa: F401  -- register learning_* tables
import abridgeai.features.quizzes.models  # noqa: F401  -- register quiz_attempts
from abridgeai.core.config import get_settings
from abridgeai.core.exceptions import AppError, ForbiddenError, NotFoundError
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
from abridgeai.features.interviews.ai.stages.validation.logic import _chunk_views_for_prompt
from abridgeai.features.interviews.models import (
    InterviewConfig,
    InterviewSession,
)
from abridgeai.features.interviews.queries import (
    authoring as authoring_queries,
)
from abridgeai.features.interviews.schemas import (
    InterviewQuestionBankItemCreate,
    InterviewQuestionBankItemUpdate,
    InterviewQuestionBankLogicalGroupCreate,
    InterviewQuestionBankSiblingCreate,
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
    status: str = "published",
) -> dict[str, Any]:
    config_id = uuid.uuid4()
    question_ids = [uuid.uuid4() for _ in range(questions)]
    outcome_ids = [uuid.uuid4() for _ in range(outcomes)]
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO interview_configs (id, course_id, module_id, title, status, created_by, max_attempts, cooldown_hours, slug) VALUES (:id, :c, :m, 'Pub Interview', :status, :t, :max_attempts, :cooldown_hours, 'slug-' || uuid_generate_v4()::text);"
            ),
            {
                "id": config_id,
                "c": course_id,
                "m": module_id,
                "t": teacher_id,
                "status": status,
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
async def test_runtime_group_collapse_selects_role_preferred_angle(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
    scenario: dict[str, Any],
) -> None:
    """A complete 4-angle group serves the role's preferred angle at runtime.

    generation_variant_strategy is generation-time metadata only: selection
    follows interviewer_role (tech_lead -> technical, hr -> behavioral),
    regardless of the flag value.
    """
    seeded = await _create_published_config(
        engine,
        course_id=scenario["course_id"],
        module_id=scenario["module_id"],
        teacher_id=scenario["teacher_id"],
        questions=4,
        outcomes=1,
        status="published",
    )
    q0, q1, q2, q3 = seeded["question_ids"]
    group_id = str(uuid.uuid4())
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "UPDATE interview_configs SET persona_profile_json=:p, "
                "generation_variant_strategy='role_only' WHERE id=:id"
            ),
            {"p": json.dumps({"interviewer_role": "backend_tech_lead"}), "id": seeded["config_id"]},
        )
        for qid, qtype in zip(
            (q0, q1, q2, q3),
            ("technical", "behavioral", "situational", "system_design"),
            strict=True,
        ):
            await conn.execute(
                text(
                    "UPDATE interview_questions SET question_type=:t, "
                    "variant_group_id=:g WHERE id=:id"
                ),
                {"t": qtype, "g": group_id, "id": qid},
            )

    payload = _CreatePayload(input_mode="text")

    async def _start_and_read() -> tuple[str, UUID]:
        async with session_factory() as session, session.begin():
            started = await taking_service.start_session(
                session, seeded["config_id"], payload, _actor(scenario["student_id"])
            )
        async with session_factory() as session:
            qtype = (
                await session.execute(
                    text(
                        "SELECT q.question_type FROM interview_session_questions s "
                        "JOIN interview_questions q ON q.id = s.interview_question_id "
                        "WHERE s.session_id=:sid"
                    ),
                    {"sid": started.id},
                )
            ).scalar_one()
        return qtype, started.id

    async def _finalize(session_id: UUID) -> None:
        # start_session resumes an in-progress row; terminalize it so the next
        # role check starts a genuinely fresh session.
        async with engine.begin() as conn:
            await conn.execute(
                text("UPDATE interview_sessions SET status='completed' WHERE id=:id"),
                {"id": session_id},
            )

    # Tech-lead role: only the technical angle of the group is served.
    qtype, sid = await _start_and_read()
    assert qtype == "technical"
    await _finalize(sid)

    # Same group, HR role: behavioral is served instead; the flag value
    # ("role_only") does not force anything.
    async with engine.begin() as conn:
        await conn.execute(
            text("UPDATE interview_configs SET persona_profile_json=:p WHERE id=:id"),
            {"p": json.dumps({"interviewer_role": "hr_screener"}), "id": seeded["config_id"]},
        )
    qtype, sid = await _start_and_read()
    assert qtype == "behavioral"
    await _finalize(sid)

    # Flag cleared to NULL: runtime outcome is unchanged (metadata only).
    async with engine.begin() as conn:
        await conn.execute(
            text("UPDATE interview_configs SET generation_variant_strategy=NULL WHERE id=:id"),
            {"id": seeded["config_id"]},
        )
    qtype, _sid = await _start_and_read()
    assert qtype == "behavioral"


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
        status="draft",
    )
    request = _CreatePayload(
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


async def _insert_foreign_scope(
    engine: AsyncEngine, org_id: uuid.UUID, teacher_id: uuid.UUID
) -> tuple[uuid.UUID, uuid.UUID, uuid.UUID]:
    """A second course + module + lesson under the same org.

    Used to prove cross-course scope is rejected: the teacher is authorized
    for ``scenario``'s course only, so any module/lesson belonging to the
    foreign course must be refused at the service boundary.
    """
    course_id, module_id, lesson_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    suffix = org_id.hex[:8]
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO courses "
                "(id, organization_id, owner_user_id, slug, title, status) "
                "VALUES (:id, :org, :owner, :slug, 'Foreign Course', 'published')"
            ),
            {
                "id": course_id,
                "org": org_id,
                "owner": teacher_id,
                "slug": f"foreign-course-{suffix}",
            },
        )
        await conn.execute(
            text(
                "INSERT INTO modules (id, course_id, title, position, status) "
                "VALUES (:id, :course, 'Foreign Module', 1, 'published')"
            ),
            {"id": module_id, "course": course_id},
        )
        await conn.execute(
            text(
                "INSERT INTO lessons (id, module_id, slug, title, status) "
                "VALUES (:id, :module, :slug, 'Foreign Lesson', 'published')"
            ),
            {"id": lesson_id, "module": module_id, "slug": f"foreign-lesson-{suffix}"},
        )
    return course_id, module_id, lesson_id


async def _cleanup_foreign_scope(
    engine: AsyncEngine,
    course_id: uuid.UUID,
    module_id: uuid.UUID,
    lesson_id: uuid.UUID | None,
) -> None:
    async with engine.begin() as conn:
        if lesson_id is not None:
            await conn.execute(text("DELETE FROM lessons WHERE id = :id"), {"id": lesson_id})
        await conn.execute(text("DELETE FROM module_items WHERE module_id = :m"), {"m": module_id})
        await conn.execute(text("DELETE FROM modules WHERE id = :id"), {"id": module_id})
        await conn.execute(text("DELETE FROM courses WHERE id = :id"), {"id": course_id})


async def test_create_config_rejects_module_from_another_course(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
    scenario: dict[str, Any],
) -> None:
    foreign_course, foreign_module, foreign_lesson = await _insert_foreign_scope(
        engine, scenario["org_id"], scenario["teacher_id"]
    )
    try:
        payload = _CreatePayload(
            title="Cross Course Interview",
            course_id=scenario["course_id"],
            module_id=foreign_module,
        )
        with pytest.raises(AppError, match="not part of course"):
            async with session_factory() as session:
                await authoring_service.create_interview_config(
                    session, scenario["course_id"], payload, _actor(scenario["teacher_id"])
                )
    finally:
        await _cleanup_foreign_scope(engine, foreign_course, foreign_module, foreign_lesson)


async def test_start_generation_run_rejects_foreign_scope(
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
        status="draft",
    )
    foreign_course, foreign_module, foreign_lesson = await _insert_foreign_scope(
        engine, scenario["org_id"], scenario["teacher_id"]
    )
    base = {"question_count": 5}
    try:
        async with session_factory() as session:
            # Foreign course_id in the body must not override the config's.
            with pytest.raises(AppError, match="course_id does not match"):
                await authoring_service.start_generation_run(
                    session,
                    seeded["config_id"],
                    _CreatePayload(
                        course_id=foreign_course,
                        module_id=scenario["module_id"],
                        **base,
                    ),
                    _actor(scenario["teacher_id"]),
                )
            # Foreign module_id must be refused even when course_id matches.
            with pytest.raises(AppError, match="module_id does not match"):
                await authoring_service.start_generation_run(
                    session,
                    seeded["config_id"],
                    _CreatePayload(
                        course_id=scenario["course_id"],
                        module_id=foreign_module,
                        **base,
                    ),
                    _actor(scenario["teacher_id"]),
                )
            # Foreign source_module_ids expand to foreign lessons.
            with pytest.raises(AppError, match="not part of course"):
                await authoring_service.start_generation_run(
                    session,
                    seeded["config_id"],
                    _CreatePayload(
                        course_id=scenario["course_id"],
                        module_id=scenario["module_id"],
                        source_module_ids=[foreign_module],
                        **base,
                    ),
                    _actor(scenario["teacher_id"]),
                )
            # Foreign source_lesson_ids must be refused outright.
            with pytest.raises(AppError, match="not part of course"):
                await authoring_service.start_generation_run(
                    session,
                    seeded["config_id"],
                    _CreatePayload(
                        course_id=scenario["course_id"],
                        module_id=scenario["module_id"],
                        source_lesson_ids=[foreign_lesson],
                        **base,
                    ),
                    _actor(scenario["teacher_id"]),
                )
            # Same-course scope still works and the config_json is canonicalized
            # to the config's own course/module.
            arq_pool = SimpleNamespace(enqueue_job=AsyncMock(return_value=None))
            run = await authoring_service.start_generation_run(
                session,
                seeded["config_id"],
                _CreatePayload(
                    course_id=scenario["course_id"],
                    module_id=scenario["module_id"],
                    source_module_ids=[scenario["module_id"]],
                    **base,
                ),
                _actor(scenario["teacher_id"]),
                arq_pool=arq_pool,
            )
            assert run.config_json["course_id"] == str(scenario["course_id"])
            assert run.config_json["module_id"] == str(scenario["module_id"])
            assert run.config_json["source_module_ids"] == [str(scenario["module_id"])]
            arq_pool.enqueue_job.assert_awaited_once()
    finally:
        await _cleanup_foreign_scope(engine, foreign_course, foreign_module, foreign_lesson)


async def test_start_generation_run_persists_type_weights(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
    scenario: dict[str, Any],
) -> None:
    """Validation must see the SAME type mix generation used.

    Generation resolves percentages from the config's supplementary
    rubric; validation compares fractions. The run snapshot normalizes
    once at enqueue so both stages agree (default 60/30/10 and custom).
    """
    seeded = await _create_published_config(
        engine,
        course_id=scenario["course_id"],
        module_id=scenario["module_id"],
        teacher_id=scenario["teacher_id"],
        questions=0,
        outcomes=0,
        status="draft",
    )
    arq_pool = SimpleNamespace(enqueue_job=AsyncMock(return_value=None))
    base = _CreatePayload(
        course_id=scenario["course_id"],
        module_id=scenario["module_id"],
        question_count=5,
    )
    async with session_factory() as session:
        run = await authoring_service.start_generation_run(
            session,
            seeded["config_id"],
            base,
            _actor(scenario["teacher_id"]),
            arq_pool=arq_pool,
        )
        assert run.config_json["type_weights"] == {
            "technical": 0.6,
            "behavioral": 0.3,
            "situational": 0.1,
        }

    # Release the active-run dedup (generation_runs partial unique index on
    # pending/running) so the second run below can be created for the SAME
    # config under its custom rubric.
    async with engine.begin() as conn:
        await conn.execute(
            text("UPDATE generation_runs SET status = 'completed' WHERE id = :id"),
            {"id": run.id},
        )

    # Custom rubric weights: generation + validation must agree on the mix.
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "UPDATE interview_configs SET supplementary_instructions = :json "
                "WHERE id = :id"
            ),
            {
                "json": json.dumps(
                    {"rubric_weights": {"technical": 80, "behavioral": 10, "situational": 10}}
                ),
                "id": seeded["config_id"],
            },
        )
    async with session_factory() as session:
        run = await authoring_service.start_generation_run(
            session,
            seeded["config_id"],
            base,
            _actor(scenario["teacher_id"]),
            arq_pool=arq_pool,
        )
        assert run.config_json["type_weights"] == {
            "technical": 0.8,
            "behavioral": 0.1,
            "situational": 0.1,
        }


async def test_start_generation_run_conflicts_while_active(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
    scenario: dict[str, Any],
) -> None:
    """Active-generation dedup: a second run for the same config → 409 Conflict."""
    from abridgeai.core.exceptions import ConflictError

    seeded = await _create_published_config(
        engine,
        course_id=scenario["course_id"],
        module_id=scenario["module_id"],
        teacher_id=scenario["teacher_id"],
        questions=0,
        outcomes=0,
        status="draft",
    )
    base = _CreatePayload(
        course_id=scenario["course_id"],
        module_id=scenario["module_id"],
        question_count=5,
    )
    arq_pool = SimpleNamespace(enqueue_job=AsyncMock(return_value=None))
    async with session_factory() as session:
        first = await authoring_service.start_generation_run(
            session,
            seeded["config_id"],
            base,
            _actor(scenario["teacher_id"]),
            arq_pool=arq_pool,
        )
        assert first is not None
        with pytest.raises(ConflictError, match="interview_generation_in_progress"):
            await authoring_service.start_generation_run(
                session,
                seeded["config_id"],
                base,
                _actor(scenario["teacher_id"]),
                arq_pool=arq_pool,
            )


async def test_add_question_rejects_outcome_not_in_this_config(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
    scenario: dict[str, Any],
) -> None:
    """Manual questions must link to an outcome of the SAME config.

    A cross-config outcome passes the FK (the row exists) but corrupts the
    rubric; a fake UUID would fail the FK with an IntegrityError. Both are
    rejected up front with NotFoundError, mirroring update_question.
    """
    seeded = await _create_published_config(
        engine,
        course_id=scenario["course_id"],
        module_id=scenario["module_id"],
        teacher_id=scenario["teacher_id"],
        questions=0,
        outcomes=1,
        status="draft",
    )
    own_outcome_id = seeded["outcome_ids"][0]
    foreign_config_id = uuid.uuid4()
    foreign_outcome_id = uuid.uuid4()
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO interview_configs "
                "(id, course_id, module_id, title, status, created_by, slug) "
                "VALUES (:id, :c, :m, 'Foreign Config', 'draft', :t, :slug)"
            ),
            {
                "id": foreign_config_id,
                "c": scenario["course_id"],
                "m": scenario["module_id"],
                "t": scenario["teacher_id"],
                "slug": f"foreign-cfg-{scenario['org_id'].hex[:8]}",
            },
        )
        await conn.execute(
            text(
                "INSERT INTO interview_outcomes "
                "(id, interview_config_id, position, outcome_text, outcome_type, "
                "importance_weight) "
                "VALUES (:id, :cfg, 1, 'Foreign Outcome', 'knowledge', 3)"
            ),
            {"id": foreign_outcome_id, "cfg": foreign_config_id},
        )
    base = {"prompt_text": "Linked to a foreign outcome?", "question_type": "technical"}
    try:
        async with session_factory() as session:
            with pytest.raises(NotFoundError, match="not found"):
                await authoring_service.add_question(
                    session,
                    seeded["config_id"],
                    _CreatePayload(linked_outcome_id=foreign_outcome_id, **base),
                    _actor(scenario["teacher_id"]),
                )
            with pytest.raises(NotFoundError, match="not found"):
                await authoring_service.add_question(
                    session,
                    seeded["config_id"],
                    _CreatePayload(linked_outcome_id=uuid.uuid4(), **base),
                    _actor(scenario["teacher_id"]),
                )
            # A link to the config's OWN outcome still works.
            question = await authoring_service.add_question(
                session,
                seeded["config_id"],
                _CreatePayload(linked_outcome_id=own_outcome_id, **base),
                _actor(scenario["teacher_id"]),
            )
            assert question.linked_outcome_id == own_outcome_id
    finally:
        async with engine.begin() as conn:
            await conn.execute(
                text("DELETE FROM interview_outcomes WHERE id = :id"), {"id": foreign_outcome_id}
            )
            await conn.execute(
                text("DELETE FROM interview_configs WHERE id = :id"), {"id": foreign_config_id}
            )


async def test_rejects_blank_question_prompt_and_outcome_text(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
    scenario: dict[str, Any],
) -> None:
    """Whitespace-only question prompts / outcome text are rejected; writes are trimmed."""
    from abridgeai.features.interviews.schemas.authoring import (
        InterviewOutcomeCreate,
        InterviewOutcomeUpdate,
        InterviewQuestionUpdate,
    )

    seeded = await _create_published_config(
        engine,
        course_id=scenario["course_id"],
        module_id=scenario["module_id"],
        teacher_id=scenario["teacher_id"],
        questions=0,
        outcomes=1,
        status="draft",
    )
    async with session_factory() as session:
        question = await authoring_service.add_question(
            session,
            seeded["config_id"],
            _CreatePayload(
                prompt_text="Needs a prompt.",
                question_type="technical",
                linked_outcome_id=seeded["outcome_ids"][0],
            ),
            _actor(scenario["teacher_id"]),
        )
        # PATCH question: whitespace-only prompt rejected.
        with pytest.raises(AppError, match="Question prompt is required"):
            await authoring_service.update_question(
                session,
                seeded["config_id"],
                question.id,
                InterviewQuestionUpdate(prompt_text=" \n\t "),
                _actor(scenario["teacher_id"]),
            )
        # PATCH question: non-blank prompt is trimmed before persistence.
        updated = await authoring_service.update_question(
            session,
            seeded["config_id"],
            question.id,
            InterviewQuestionUpdate(prompt_text="  trimmed prompt  "),
            _actor(scenario["teacher_id"]),
        )
        assert updated.prompt_text == "trimmed prompt"

        # Create outcome: whitespace-only text rejected.
        with pytest.raises(AppError, match="Outcome text is required"):
            await authoring_service.add_outcome(
                session,
                seeded["config_id"],
                InterviewOutcomeCreate(
                    position=2, outcome_text=" \n\t ", outcome_type="knowledge"
                ),
                _actor(scenario["teacher_id"]),
            )

        # PATCH outcome: whitespace-only text rejected; real text trimmed.
        outcome_id = seeded["outcome_ids"][0]
        with pytest.raises(AppError, match="Outcome text is required"):
            await authoring_service.update_outcome(
                session,
                seeded["config_id"],
                outcome_id,
                InterviewOutcomeUpdate(outcome_text="   "),
                _actor(scenario["teacher_id"]),
            )
        updated_outcome = await authoring_service.update_outcome(
            session,
            seeded["config_id"],
            outcome_id,
            InterviewOutcomeUpdate(outcome_text="  trimmed outcome  "),
            _actor(scenario["teacher_id"]),
        )
        assert updated_outcome.outcome_text == "trimmed outcome"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "patch",
    [
        {"prompt_text": "Updated coherence-bearing prompt."},
        {"question_type": "behavioral"},
        {"difficulty": "senior"},
        {"linked_outcome_id": None},
    ],
)
async def test_semantic_question_edit_clears_variant_group(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
    scenario: dict[str, Any],
    patch: dict[str, Any],
) -> None:
    seeded = await _create_published_config(
        engine,
        course_id=scenario["course_id"],
        module_id=scenario["module_id"],
        teacher_id=scenario["teacher_id"],
        questions=4,
        outcomes=1,
        status="draft",
    )
    group_id = uuid.uuid4()
    async with engine.begin() as conn:
        for question_id, question_type in zip(
            seeded["question_ids"],
            ("technical", "system_design", "situational", "behavioral"),
            strict=True,
        ):
            await conn.execute(
                text(
                    "UPDATE interview_questions SET variant_group_id=:group_id, "
                    "question_type=:question_type, linked_outcome_id=:outcome_id "
                    "WHERE id=:question_id"
                ),
                {
                    "group_id": group_id,
                    "question_type": question_type,
                    "outcome_id": seeded["outcome_ids"][0],
                    "question_id": question_id,
                },
            )

    async with session_factory() as session:
        await authoring_service.update_question(
            session,
            seeded["config_id"],
            seeded["question_ids"][0],
            _CreatePayload(**patch),
            _actor(scenario["teacher_id"]),
        )
        await session.commit()

    async with session_factory() as session:
        group_ids = (
            await session.execute(
                text(
                    "SELECT variant_group_id FROM interview_questions "
                    "WHERE interview_config_id=:config_id ORDER BY position"
                ),
                {"config_id": seeded["config_id"]},
            )
        ).scalars().all()
    assert group_ids == [None, None, None, None]


@pytest.mark.asyncio
async def test_nonsemantic_question_edit_keeps_variant_group(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
    scenario: dict[str, Any],
) -> None:
    seeded = await _create_published_config(
        engine,
        course_id=scenario["course_id"],
        module_id=scenario["module_id"],
        teacher_id=scenario["teacher_id"],
        questions=4,
        outcomes=1,
        status="draft",
    )
    group_id = uuid.uuid4()
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "UPDATE interview_questions SET variant_group_id=:group_id "
                "WHERE interview_config_id=:config_id"
            ),
            {"group_id": group_id, "config_id": seeded["config_id"]},
        )

    async with session_factory() as session:
        await authoring_service.update_question(
            session,
            seeded["config_id"],
            seeded["question_ids"][0],
            _CreatePayload(model_answer="Updated answer."),
            _actor(scenario["teacher_id"]),
        )
        await session.commit()

    async with session_factory() as session:
        group_ids = (
            await session.execute(
                text(
                    "SELECT variant_group_id FROM interview_questions "
                    "WHERE interview_config_id=:config_id ORDER BY position"
                ),
                {"config_id": seeded["config_id"]},
            )
        ).scalars().all()
    assert group_ids == [group_id] * 4


@pytest.mark.asyncio
async def test_approve_variants_stamps_whole_group(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
    scenario: dict[str, Any],
) -> None:
    seeded = await _create_published_config(
        engine,
        course_id=scenario["course_id"],
        module_id=scenario["module_id"],
        teacher_id=scenario["teacher_id"],
        questions=4,
        outcomes=1,
        status="draft",
    )
    group_id = uuid.uuid4()
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "UPDATE interview_questions SET variant_group_id=:group_id, "
                "linked_outcome_id=:outcome_id "
                "WHERE interview_config_id=:config_id"
            ),
            {
                "group_id": group_id,
                "outcome_id": seeded["outcome_ids"][0],
                "config_id": seeded["config_id"],
            },
        )
        await conn.execute(
            text(
                "UPDATE interview_questions SET review_status='pending' "
                "WHERE interview_config_id=:config_id"
            ),
            {"config_id": seeded["config_id"]},
        )

    async with session_factory() as session:
        approved = await authoring_service.approve_question_variants(
            session,
            seeded["config_id"],
            seeded["question_ids"][0],
            _actor(scenario["teacher_id"]),
        )
        await session.commit()
    assert approved == 4

    async with session_factory() as session:
        rows = (
            await session.execute(
                text(
                    "SELECT review_status, reviewed_by, reviewed_at "
                    "FROM interview_questions "
                    "WHERE interview_config_id=:config_id ORDER BY position"
                ),
                {"config_id": seeded["config_id"]},
            )
        ).all()
    assert all(row.review_status == "approved" for row in rows)
    assert all(row.reviewed_by == scenario["teacher_id"] for row in rows)
    assert all(row.reviewed_at is not None for row in rows)


@pytest.mark.asyncio
async def test_approve_variants_standalone_question_approves_only_itself(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
    scenario: dict[str, Any],
) -> None:
    seeded = await _create_published_config(
        engine,
        course_id=scenario["course_id"],
        module_id=scenario["module_id"],
        teacher_id=scenario["teacher_id"],
        questions=4,
        outcomes=1,
        status="draft",
    )
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "UPDATE interview_questions SET review_status='pending' "
                "WHERE interview_config_id=:config_id"
            ),
            {"config_id": seeded["config_id"]},
        )

    async with session_factory() as session:
        approved = await authoring_service.approve_question_variants(
            session,
            seeded["config_id"],
            seeded["question_ids"][0],
            _actor(scenario["teacher_id"]),
        )
        await session.commit()
    assert approved == 1

    async with session_factory() as session:
        statuses = (
            await session.execute(
                text(
                    "SELECT review_status FROM interview_questions "
                    "WHERE interview_config_id=:config_id ORDER BY position"
                ),
                {"config_id": seeded["config_id"]},
            )
        ).scalars().all()
    assert statuses == ["approved", "pending", "pending", "pending"]


@pytest.mark.asyncio
async def test_approve_variants_rejects_published_config(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
    scenario: dict[str, Any],
) -> None:
    from abridgeai.core.exceptions import ConflictError

    seeded = await _create_published_config(
        engine,
        course_id=scenario["course_id"],
        module_id=scenario["module_id"],
        teacher_id=scenario["teacher_id"],
        questions=4,
        outcomes=1,
        status="published",
    )
    async with session_factory() as session:
        with pytest.raises(ConflictError):
            await authoring_service.approve_question_variants(
                session,
                seeded["config_id"],
                seeded["question_ids"][0],
                _actor(scenario["teacher_id"]),
            )


async def test_start_generation_run_enqueue_failure_marks_run_failed(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
    scenario: dict[str, Any],
) -> None:
    """An ARQ/Redis enqueue failure after commit terminalizes the pending run.

    The run was already committed as pending; the failure handler must not
    leave it stuck pending forever, and must not clobber a run a worker
    already claimed. The original exception is re-raised.
    """
    seeded = await _create_published_config(
        engine,
        course_id=scenario["course_id"],
        module_id=scenario["module_id"],
        teacher_id=scenario["teacher_id"],
        questions=0,
        outcomes=0,
        status="draft",
    )
    base = _CreatePayload(
        course_id=scenario["course_id"],
        module_id=scenario["module_id"],
        question_count=5,
    )
    failing_pool = SimpleNamespace(
        enqueue_job=AsyncMock(side_effect=RuntimeError("redis connection lost"))
    )
    async with session_factory() as session:
        with pytest.raises(RuntimeError, match="redis connection lost"):
            await authoring_service.start_generation_run(
                session,
                seeded["config_id"],
                base,
                _actor(scenario["teacher_id"]),
                arq_pool=failing_pool,
            )

    async with session_factory() as session:
        row = (
            await session.execute(
                text(
                    "SELECT r.status, r.finished_at, r.config_json "
                    "FROM generation_runs r "
                    "JOIN interview_configs c ON c.generation_run_id = r.id "
                    "WHERE c.id = :cfg"
                ),
                {"cfg": seeded["config_id"]},
            )
        ).one()
    assert row.status == "failed"
    assert row.finished_at is not None
    assert row.config_json["failure"]["message"] == "Generation worker could not be queued"


async def _seed_lesson_and_chunk(
    engine: AsyncEngine, scenario: dict[str, Any]
) -> dict[str, uuid.UUID]:
    """A lesson + one indexed chunk under the scenario's course/module."""
    lesson_id, storage_id, material_id, version_id, chunk_id = (uuid.uuid4() for _ in range(5))
    suffix = scenario["org_id"].hex[:8]
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO lessons (id, module_id, slug, title, status) "
                "VALUES (:id, :m, :slug, 'Chunk Lesson', 'published')"
            ),
            {"id": lesson_id, "m": scenario["module_id"], "slug": f"chunk-lesson-{suffix}"},
        )
        await conn.execute(
            text(
                "INSERT INTO storage_objects (id, bucket, object_key) "
                "VALUES (:id, 'test', :key)"
            ),
            {"id": storage_id, "key": f"chunk/{suffix}.txt"},
        )
        await conn.execute(
            text(
                "INSERT INTO learning_materials (id, lesson_id, title, material_type) "
                "VALUES (:id, :l, 'Chunk Material', 'text')"
            ),
            {"id": material_id, "l": lesson_id},
        )
        await conn.execute(
            text(
                "INSERT INTO learning_material_versions "
                "(id, material_id, storage_object_id, version_no, processing_status) "
                "VALUES (:id, :m, :s, 1, 'ready')"
            ),
            {"id": version_id, "m": material_id, "s": storage_id},
        )
        await conn.execute(
            text(
                "INSERT INTO document_chunks "
                "(id, course_id, module_id, lesson_id, material_version_id, "
                "chunk_index, chunk_type, content, embedding, content_hash) "
                "VALUES (:id, :c, :m, :l, :v, 0, 'text', :content, NULL, :hash)"
            ),
            {
                "id": chunk_id,
                "c": scenario["course_id"],
                "m": scenario["module_id"],
                "l": lesson_id,
                "v": version_id,
                "content": "Chunk content for validation",
                "hash": hashlib.sha256(b"chunk").hexdigest(),
            },
        )
    return {
        "lesson_id": lesson_id,
        "storage_id": storage_id,
        "material_id": material_id,
        "version_id": version_id,
        "chunk_id": chunk_id,
    }


async def _cleanup_lesson_and_chunk(
    engine: AsyncEngine, seeded: dict[str, uuid.UUID]
) -> None:
    async with engine.begin() as conn:
        await conn.execute(
            text("DELETE FROM document_chunks WHERE id = :id"), {"id": seeded["chunk_id"]}
        )
        await conn.execute(
            text(
                "UPDATE learning_materials SET current_version_id = NULL "
                "WHERE id = :id"
            ),
            {"id": seeded["material_id"]},
        )
        await conn.execute(
            text("DELETE FROM learning_material_versions WHERE id = :id"),
            {"id": seeded["version_id"]},
        )
        await conn.execute(
            text("DELETE FROM learning_materials WHERE id = :id"), {"id": seeded["material_id"]}
        )
        await conn.execute(
            text("DELETE FROM storage_objects WHERE id = :id"), {"id": seeded["storage_id"]}
        )
        await conn.execute(
            text("DELETE FROM lessons WHERE id = :id"), {"id": seeded["lesson_id"]}
        )


class _Chunk:
    def __init__(self, content: str, *, chunk_id: UUID | None = None) -> None:
        self.chunk_id = chunk_id
        self.id = chunk_id or uuid.uuid4()
        self.content = content


class _ChunkContext:
    """Immutable-chunks stand-in for InterviewRetrievalContext."""

    def __init__(self, chunks: tuple[_Chunk, ...] = ()) -> None:
        self.chunks = chunks


def test_chunk_views_for_prompt_uses_retrieval_content() -> None:
    """Leading check gets chunk ids + CONTENT from the live retrieval context.

    No second DB lookup and no source text in run JSON: the views are built
    from the in-memory ``context.chunks`` (both ``chunk_id`` and plain ``id``
    chunk shapes are accepted, matching ``_collect_chunk_ids``).
    """
    chunk_a = uuid.uuid4()
    id_only = uuid.uuid4()
    context = _ChunkContext(
        (
            _Chunk("Chunk content for validation", chunk_id=chunk_a),
            _Chunk("", chunk_id=uuid.uuid4()),
            _Chunk("id-only chunk shape", chunk_id=None),
        )
    )
    # The id-only chunk's stub id is server-assigned via _Chunk.__init__.
    id_only = context.chunks[2].id
    views = _chunk_views_for_prompt(context)
    assert [v["id"] for v in views] == [str(chunk_a), str(id_only)]
    assert all(v["content"] for v in views)
    # Empty content chunks are skipped; a chunkless context degrades to [].
    assert _chunk_views_for_prompt(_ChunkContext()) == []


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


def _bank_payload(question_type: str, prompt: str, *, difficulty: str | None = None) -> dict[str, Any]:
    return {
        "prompt_text": prompt,
        "question_type": question_type,
        "difficulty": difficulty,
        "model_answer": f"Answer for {prompt}",
        "tags": [],
    }


@pytest.mark.asyncio
async def test_create_logical_group_makes_four_grouped_bank_items(
    session_factory: async_sessionmaker[AsyncSession],
    scenario: dict[str, Any],
) -> None:
    payload = InterviewQuestionBankLogicalGroupCreate(
        items=[
            InterviewQuestionBankItemCreate(
                **_bank_payload(t, f"Group {t}?")
            )
            for t in ("technical", "system_design", "situational", "behavioral")
        ]
    )

    async with session_factory() as session:
        items = await authoring_service.create_question_bank_logical_group(
            session,
            scenario["course_id"],
            payload,
            _actor(scenario["teacher_id"]),
        )
        await session.commit()

    assert len(items) == 4
    group_ids = {item.variant_group_id for item in items}
    assert len(group_ids) == 1
    assert group_ids != {None}
    types = {item.question_type for item in items}
    assert types == {"technical", "system_design", "situational", "behavioral"}


@pytest.mark.asyncio
async def test_create_logical_group_rejects_duplicate_angle(
    session_factory: async_sessionmaker[AsyncSession],
    scenario: dict[str, Any],
) -> None:
    from pydantic import ValidationError

    items = [
        InterviewQuestionBankItemCreate(**_bank_payload("technical", f"T{i}?"))
        for i in range(4)
    ]
    items[1].question_type = "technical"  # duplicate angle
    with pytest.raises(ValidationError):
        InterviewQuestionBankLogicalGroupCreate(items=items)


@pytest.mark.asyncio
async def test_add_sibling_expands_singleton_into_group(
    session_factory: async_sessionmaker[AsyncSession],
    scenario: dict[str, Any],
) -> None:
    async with session_factory() as session:
        singleton = await authoring_service.add_to_question_bank(
            session,
            scenario["course_id"],
            _CreatePayload(**_bank_payload("technical", "Solo technical")),
            _actor(scenario["teacher_id"]),
        )
        await session.commit()
        singleton_id = singleton.id
    assert singleton.variant_group_id is None

    async with session_factory() as session:
        items = await authoring_service.add_question_bank_sibling(
            session,
            scenario["course_id"],
            singleton_id,
            InterviewQuestionBankSiblingCreate(
                **_bank_payload("system_design", "Solo system design")
            ),
            _actor(scenario["teacher_id"]),
        )
        await session.commit()

    assert len(items) == 2
    group_ids = {item.variant_group_id for item in items}
    assert len(group_ids) == 1
    assert group_ids != {None}
    assert {item.question_type for item in items} == {"technical", "system_design"}


@pytest.mark.asyncio
async def test_add_sibling_rejects_duplicate_angle_in_group(
    session_factory: async_sessionmaker[AsyncSession],
    scenario: dict[str, Any],
) -> None:
    from abridgeai.core.exceptions import ConflictError

    async with session_factory() as session:
        singleton = await authoring_service.add_to_question_bank(
            session,
            scenario["course_id"],
            _CreatePayload(**_bank_payload("technical", "Dup technical")),
            _actor(scenario["teacher_id"]),
        )
        await session.commit()
        singleton_id = singleton.id

    async with session_factory() as session:
        with pytest.raises(ConflictError):
            await authoring_service.add_question_bank_sibling(
                session,
                scenario["course_id"],
                singleton_id,
                InterviewQuestionBankSiblingCreate(
                    **_bank_payload("technical", "Dup technical 2")
                ),
                _actor(scenario["teacher_id"]),
            )


@pytest.mark.asyncio
async def test_import_bank_expands_group_and_fresh_group_id(
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
        status="draft",
    )
    payload = InterviewQuestionBankLogicalGroupCreate(
        items=[
            InterviewQuestionBankItemCreate(
                **_bank_payload(t, f"Group {t}?")
            )
            for t in ("technical", "system_design", "situational", "behavioral")
        ]
    )
    async with session_factory() as session:
        group = await authoring_service.create_question_bank_logical_group(
            session,
            scenario["course_id"],
            payload,
            _actor(scenario["teacher_id"]),
        )
        await session.commit()
    bank_group_id = {i.variant_group_id for i in group}.pop()

    async with session_factory() as session:
        # Selecting a NON-first child (behavioral) still pulls in the whole
        # group, and the persisted order is the canonical interviewer-angle
        # sequence — never UUID / SQL order.
        late_child = next(
            item for item in group if item.question_type == "behavioral"
        )
        created = await authoring_service.import_question_bank_items(
            session,
            seeded["config_id"],
            [late_child.id],
            _actor(scenario["teacher_id"]),
        )
        await session.commit()

    assert len(created) == 4
    config_group_ids = {q.variant_group_id for q in created}
    assert len(config_group_ids) == 1
    assert config_group_ids != {bank_group_id}  # fresh group id, never the bank's
    assert all(q.review_status == "approved" for q in created)
    assert [q.question_type for q in created] == [
        "technical",
        "system_design",
        "situational",
        "behavioral",
    ]
    assert [q.position for q in created] == [1, 2, 3, 4]


@pytest.mark.asyncio
async def test_import_bank_rolls_back_on_prompt_collision(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
    scenario: dict[str, Any],
) -> None:
    from abridgeai.core.exceptions import ConflictError

    seeded = await _create_published_config(
        engine,
        course_id=scenario["course_id"],
        module_id=scenario["module_id"],
        teacher_id=scenario["teacher_id"],
        questions=0,
        outcomes=0,
        status="draft",
    )
    # A bank item whose prompt already exists inside the config.
    async with session_factory() as session:
        await authoring_service.add_question(
            session,
            seeded["config_id"],
            _CreatePayload(**_bank_payload("technical", "Collides with bank")),
            _actor(scenario["teacher_id"]),
        )
        await session.commit()
        bank = await authoring_service.add_to_question_bank(
            session,
            scenario["course_id"],
            _CreatePayload(**_bank_payload("technical", "Collides with bank")),
            _actor(scenario["teacher_id"]),
        )
        await session.commit()
        bank_id = bank.id

    async with session_factory() as session:
        with pytest.raises(ConflictError):
            await authoring_service.import_question_bank_items(
                session,
                seeded["config_id"],
                [bank_id],
                _actor(scenario["teacher_id"]),
            )

    async with session_factory() as session:
        count = (
            await session.execute(
                text(
                    "SELECT count(*) FROM interview_questions "
                    "WHERE interview_config_id=:config_id"
                ),
                {"config_id": seeded["config_id"]},
            )
        ).scalar_one()
    assert count == 1  # the pre-existing question only; import added nothing


@pytest.mark.asyncio
async def test_import_bank_imports_standalone_singleton(
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
        status="draft",
    )
    async with session_factory() as session:
        bank = await authoring_service.add_to_question_bank(
            session,
            scenario["course_id"],
            _CreatePayload(**_bank_payload("technical", "Standalone import")),
            _actor(scenario["teacher_id"]),
        )
        await session.commit()
        bank_id = bank.id

    async with session_factory() as session:
        created = await authoring_service.import_question_bank_items(
            session,
            seeded["config_id"],
            [bank_id],
            _actor(scenario["teacher_id"]),
        )
        await session.commit()

    assert len(created) == 1
    assert created[0].variant_group_id is None
    assert created[0].position == 1


@pytest.mark.asyncio
async def test_import_bank_ignores_soft_deleted_config_prompt(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
    scenario: dict[str, Any],
) -> None:
    """A soft-deleted question's prompt must not block re-import (the ORM
    loader-criteria listener already hides deleted rows from authoring reads)."""
    seeded = await _create_published_config(
        engine,
        course_id=scenario["course_id"],
        module_id=scenario["module_id"],
        teacher_id=scenario["teacher_id"],
        questions=1,
        outcomes=0,
        status="draft",
    )
    async with session_factory() as session:
        await authoring_service.delete_question(
            session,
            seeded["config_id"],
            seeded["question_ids"][0],
            _actor(scenario["teacher_id"]),
        )
        await session.commit()
        bank = await authoring_service.add_to_question_bank(
            session,
            scenario["course_id"],
            _CreatePayload(**_bank_payload("technical", "Question 1?")),
            _actor(scenario["teacher_id"]),
        )
        await session.commit()
        bank_id = bank.id

    async with session_factory() as session:
        created = await authoring_service.import_question_bank_items(
            session,
            seeded["config_id"],
            [bank_id],
            _actor(scenario["teacher_id"]),
        )
        await session.commit()

    assert len(created) == 1
    assert created[0].position == 1  # deleted question freed the prompt + position


# --- Duplicate-import / scope / group-edit guards -------------------------------


@pytest.mark.asyncio
async def test_import_bank_rejects_duplicate_item_ids(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
    scenario: dict[str, Any],
) -> None:
    """The same bank item listed twice is a bad request, not two questions."""
    seeded = await _create_published_config(
        engine,
        course_id=scenario["course_id"],
        module_id=scenario["module_id"],
        teacher_id=scenario["teacher_id"],
        questions=0,
        outcomes=0,
        status="draft",
    )
    async with session_factory() as session:
        bank = await authoring_service.add_to_question_bank(
            session,
            scenario["course_id"],
            _CreatePayload(**_bank_payload("technical", "Twice selected")),
            _actor(scenario["teacher_id"]),
        )
        await session.commit()
        bank_id = bank.id

    async with session_factory() as session:
        with pytest.raises(AppError):
            await authoring_service.import_question_bank_items(
                session,
                seeded["config_id"],
                [bank_id, bank_id],
                _actor(scenario["teacher_id"]),
            )

    async with session_factory() as session:
        count = (
            await session.execute(
                text(
                    "SELECT count(*) FROM interview_questions "
                    "WHERE interview_config_id=:config_id"
                ),
                {"config_id": seeded["config_id"]},
            )
        ).scalar_one()
    assert count == 0  # nothing written on a rejected request


@pytest.mark.asyncio
async def test_import_bank_rejects_duplicate_prompts_within_request(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
    scenario: dict[str, Any],
) -> None:
    """Two DIFFERENT bank items sharing one prompt cannot both be imported.

    The pre-existing collision check only compared against prompts already in
    the config, so a request that carried the collision inside itself slipped
    through and produced two identical questions.
    """
    from abridgeai.core.exceptions import ConflictError

    seeded = await _create_published_config(
        engine,
        course_id=scenario["course_id"],
        module_id=scenario["module_id"],
        teacher_id=scenario["teacher_id"],
        questions=0,
        outcomes=0,
        status="draft",
    )
    async with session_factory() as session:
        first = await authoring_service.add_to_question_bank(
            session,
            scenario["course_id"],
            _CreatePayload(**_bank_payload("technical", "Same prompt")),
            _actor(scenario["teacher_id"]),
        )
        await session.commit()
        second = await authoring_service.add_to_question_bank(
            session,
            scenario["course_id"],
            # Normalised prompt equality: case + surrounding whitespace only.
            _CreatePayload(**_bank_payload("behavioral", "  SAME PROMPT ")),
            _actor(scenario["teacher_id"]),
        )
        await session.commit()
        ids = [first.id, second.id]

    async with session_factory() as session:
        with pytest.raises(ConflictError):
            await authoring_service.import_question_bank_items(
                session,
                seeded["config_id"],
                ids,
                _actor(scenario["teacher_id"]),
            )

    async with session_factory() as session:
        count = (
            await session.execute(
                text(
                    "SELECT count(*) FROM interview_questions "
                    "WHERE interview_config_id=:config_id"
                ),
                {"config_id": seeded["config_id"]},
            )
        ).scalar_one()
    assert count == 0


@pytest.mark.asyncio
async def test_create_logical_group_rejects_duplicate_prompts(
    session_factory: async_sessionmaker[AsyncSession],
    scenario: dict[str, Any],
) -> None:
    """Four distinct angles are not enough — the prompts must differ too."""
    from abridgeai.core.exceptions import ConflictError

    payload = InterviewQuestionBankLogicalGroupCreate(
        items=[
            InterviewQuestionBankItemCreate(**_bank_payload(angle, "Identical prompt"))
            for angle in ("technical", "system_design", "situational", "behavioral")
        ]
    )
    async with session_factory() as session:
        with pytest.raises(ConflictError):
            await authoring_service.create_question_bank_logical_group(
                session,
                scenario["course_id"],
                payload,
                _actor(scenario["teacher_id"]),
            )


@pytest.mark.asyncio
async def test_add_sibling_rejects_prompt_duplicate_of_group(
    session_factory: async_sessionmaker[AsyncSession],
    scenario: dict[str, Any],
) -> None:
    """A new angle may not reuse a prompt already present in the group."""
    from abridgeai.core.exceptions import ConflictError

    async with session_factory() as session:
        singleton = await authoring_service.add_to_question_bank(
            session,
            scenario["course_id"],
            _CreatePayload(**_bank_payload("technical", "Shared sibling prompt")),
            _actor(scenario["teacher_id"]),
        )
        await session.commit()
        singleton_id = singleton.id

    async with session_factory() as session:
        with pytest.raises(ConflictError):
            await authoring_service.add_question_bank_sibling(
                session,
                scenario["course_id"],
                singleton_id,
                InterviewQuestionBankSiblingCreate(
                    **_bank_payload("system_design", "shared sibling prompt")
                ),
                _actor(scenario["teacher_id"]),
            )


@pytest.mark.asyncio
async def test_start_generation_rejects_outcomes_outside_config(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
    scenario: dict[str, Any],
) -> None:
    """Unknown, soft-deleted, and foreign-config outcome ids are all rejected.

    Silently dropping them let the pipeline fall back to "all outcomes", so a
    teacher who targeted one rubric criterion got a full-scope run instead.
    """
    seeded = await _create_published_config(
        engine,
        course_id=scenario["course_id"],
        module_id=scenario["module_id"],
        teacher_id=scenario["teacher_id"],
        questions=0,
        outcomes=2,
        status="draft",
    )
    other = await _create_published_config(
        engine,
        course_id=scenario["course_id"],
        module_id=scenario["module_id"],
        teacher_id=scenario["teacher_id"],
        questions=0,
        outcomes=1,
        status="draft",
    )
    # Soft-delete one of this config's own outcomes.
    async with session_factory() as session:
        await authoring_service.delete_outcome(
            session,
            seeded["config_id"],
            seeded["outcome_ids"][1],
            _actor(scenario["teacher_id"]),
        )
        await session.commit()

    arq_pool = SimpleNamespace(enqueue_job=AsyncMock(return_value=None))
    bad_selections = {
        "unknown": [uuid.uuid4()],
        "soft_deleted": [seeded["outcome_ids"][1]],
        "foreign_config": [other["outcome_ids"][0]],
        "duplicate": [seeded["outcome_ids"][0], seeded["outcome_ids"][0]],
    }
    for label, target_ids in bad_selections.items():
        request = _CreatePayload(
                course_id=scenario["course_id"],
            module_id=scenario["module_id"],
            question_count=2,
            # _CreatePayload.model_dump(mode="json") only stringifies top-level
            # UUIDs; the real Pydantic schema serialises nested lists, so pass
            # the already-JSON shape the service actually receives.
            target_outcome_ids=[str(x) for x in target_ids],
        )
        async with session_factory() as session:
            with pytest.raises(AppError, match="target_outcome_ids"):
                await authoring_service.start_generation_run(
                    session,
                    seeded["config_id"],
                    request,
                    _actor(scenario["teacher_id"]),
                    arq_pool=arq_pool,
                )
        assert arq_pool.enqueue_job.await_count == 0, label

    # The live outcome of THIS config still passes and reaches config_json.
    request = _CreatePayload(
        course_id=scenario["course_id"],
        module_id=scenario["module_id"],
        question_count=2,
        target_outcome_ids=[str(seeded["outcome_ids"][0])],
    )
    async with session_factory() as session:
        run = await authoring_service.start_generation_run(
            session,
            seeded["config_id"],
            request,
            _actor(scenario["teacher_id"]),
            arq_pool=arq_pool,
        )
    assert run.config_json["target_outcome_ids"] == [str(seeded["outcome_ids"][0])]
    arq_pool.enqueue_job.assert_awaited_once()


@pytest.mark.asyncio
async def test_update_bank_item_keeps_group_invariants(
    session_factory: async_sessionmaker[AsyncSession],
    scenario: dict[str, Any],
) -> None:
    """Editing a grouped item cannot break its group — 400/409, never a 500."""
    from abridgeai.core.exceptions import ConflictError

    payload = InterviewQuestionBankLogicalGroupCreate(
        items=[
            InterviewQuestionBankItemCreate(**_bank_payload(angle, f"Edit guard {angle}?"))
            for angle in ("technical", "system_design", "situational", "behavioral")
        ]
    )
    async with session_factory() as session:
        group = await authoring_service.create_question_bank_logical_group(
            session,
            scenario["course_id"],
            payload,
            _actor(scenario["teacher_id"]),
        )
        await session.commit()
    technical = next(item for item in group if item.question_type == "technical")

    # 1. Leaving the four-angle vocabulary (e.g. `conceptual`) is a bad request.
    async with session_factory() as session:
        with pytest.raises(AppError) as excinfo:
            await authoring_service.update_question_bank_item(
                session,
                scenario["course_id"],
                technical.id,
                InterviewQuestionBankItemUpdate(question_type="conceptual"),
                _actor(scenario["teacher_id"]),
            )
        assert not isinstance(excinfo.value, ConflictError)

    # 2. Taking an angle a sibling already holds is a conflict.
    async with session_factory() as session:
        with pytest.raises(ConflictError):
            await authoring_service.update_question_bank_item(
                session,
                scenario["course_id"],
                technical.id,
                InterviewQuestionBankItemUpdate(question_type="behavioral"),
                _actor(scenario["teacher_id"]),
            )

    # 3. Reusing a sibling's prompt (normalised) is a conflict.
    async with session_factory() as session:
        with pytest.raises(ConflictError):
            await authoring_service.update_question_bank_item(
                session,
                scenario["course_id"],
                technical.id,
                InterviewQuestionBankItemUpdate(prompt_text="  EDIT GUARD BEHAVIORAL? "),
                _actor(scenario["teacher_id"]),
            )

    # 4. A blank prompt is rejected instead of writing an empty question.
    async with session_factory() as session:
        with pytest.raises(AppError):
            await authoring_service.update_question_bank_item(
                session,
                scenario["course_id"],
                technical.id,
                InterviewQuestionBankItemUpdate(prompt_text="   "),
                _actor(scenario["teacher_id"]),
            )

    # 5. The editable fields still save, and the group is unchanged.
    async with session_factory() as session:
        updated = await authoring_service.update_question_bank_item(
            session,
            scenario["course_id"],
            technical.id,
            InterviewQuestionBankItemUpdate(
                prompt_text="Edit guard technical, revised?",
                difficulty="senior",
                model_answer="Revised answer",
                tags=["revised"],
            ),
            _actor(scenario["teacher_id"]),
        )
        await session.commit()
    assert updated.question_type == "technical"
    assert updated.prompt_text == "Edit guard technical, revised?"
    assert updated.difficulty == "senior"
    assert updated.tags_json == ["revised"]

    async with session_factory() as session:
        angles = (
            await session.execute(
                text(
                    "SELECT question_type FROM interview_question_bank_items "
                    "WHERE variant_group_id=:g AND deleted_at IS NULL"
                ),
                {"g": technical.variant_group_id},
            )
        ).scalars().all()
    assert sorted(angles) == [
        "behavioral",
        "situational",
        "system_design",
        "technical",
    ]


@pytest.mark.asyncio
async def test_update_bank_item_ungrouped_can_change_type(
    session_factory: async_sessionmaker[AsyncSession],
    scenario: dict[str, Any],
) -> None:
    """A standalone item keeps full type freedom — the guard is group-scoped."""
    async with session_factory() as session:
        item = await authoring_service.add_to_question_bank(
            session,
            scenario["course_id"],
            _CreatePayload(**_bank_payload("technical", "Standalone retype")),
            _actor(scenario["teacher_id"]),
        )
        await session.commit()
        item_id = item.id

    async with session_factory() as session:
        updated = await authoring_service.update_question_bank_item(
            session,
            scenario["course_id"],
            item_id,
            InterviewQuestionBankItemUpdate(question_type="conceptual"),
            _actor(scenario["teacher_id"]),
        )
        await session.commit()
    assert updated.question_type == "conceptual"


@pytest.mark.asyncio
async def test_concurrent_sibling_adds_through_different_children_serialize(
    session_factory: async_sessionmaker[AsyncSession],
    scenario: dict[str, Any],
) -> None:
    """Two adds racing on the same group via DIFFERENT anchors: one 409, no break.

    Row locks cannot help here — each request locks the child it names, and the
    children are different rows. Two defences must hold together: the group
    advisory lock serialises the pre-checks, and ``uq_iq_bank_live_group_angle``
    is the last arbiter, mapped to a clean 409 rather than an IntegrityError
    surfacing as HTTP 500. Either way the group must not gain a duplicate angle.
    """
    import asyncio

    from abridgeai.core.exceptions import ConflictError

    # A PARTIAL group (2 of 4 angles) is the shape that has room for a race;
    # the bulk create endpoint only accepts complete four-angle payloads, so
    # build it the way the UI does: singleton, then one sibling.
    async with session_factory() as session:
        singleton = await authoring_service.add_to_question_bank(
            session,
            scenario["course_id"],
            _CreatePayload(**_bank_payload("technical", "Race technical?")),
            _actor(scenario["teacher_id"]),
        )
        await session.commit()
        group = await authoring_service.add_question_bank_sibling(
            session,
            scenario["course_id"],
            singleton.id,
            InterviewQuestionBankSiblingCreate(
                **_bank_payload("system_design", "Race system_design?")
            ),
            _actor(scenario["teacher_id"]),
        )
        await session.commit()
    group_id = group[0].variant_group_id
    anchors = {item.question_type: item.id for item in group}
    assert set(anchors) == {"technical", "system_design"}

    async def add_via(anchor_id: uuid.UUID, prompt: str) -> str:
        async with session_factory() as session:
            try:
                await authoring_service.add_question_bank_sibling(
                    session,
                    scenario["course_id"],
                    anchor_id,
                    InterviewQuestionBankSiblingCreate(
                        **_bank_payload("situational", prompt)
                    ),
                    _actor(scenario["teacher_id"]),
                )
                await session.commit()
            except ConflictError:
                await session.rollback()
                return "conflict"
            return "ok"

    results = await asyncio.gather(
        add_via(anchors["technical"], "Race situational A?"),
        add_via(anchors["system_design"], "Race situational B?"),
    )
    assert sorted(results) == ["conflict", "ok"]

    async with session_factory() as session:
        angles = (
            await session.execute(
                text(
                    "SELECT question_type FROM interview_question_bank_items "
                    "WHERE variant_group_id=:g AND deleted_at IS NULL"
                ),
                {"g": group_id},
            )
        ).scalars().all()
    assert sorted(angles) == ["situational", "system_design", "technical"]


@pytest.mark.asyncio
async def test_concurrent_bank_imports_do_not_collide_on_position(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
    scenario: dict[str, Any],
) -> None:
    """Two imports into one config must produce distinct, contiguous positions.

    ``next_question_position`` is a ``MAX(position)+1`` read, so two overlapping
    transactions both computed the same next position and one lost to
    ``uq_interview_questions_position``. The per-config append advisory lock
    serialises the read-then-insert window.
    """
    import asyncio

    seeded = await _create_published_config(
        engine,
        course_id=scenario["course_id"],
        module_id=scenario["module_id"],
        teacher_id=scenario["teacher_id"],
        questions=0,
        outcomes=0,
        status="draft",
    )
    async with session_factory() as session:
        bank_ids = []
        for idx in range(2):
            item = await authoring_service.add_to_question_bank(
                session,
                scenario["course_id"],
                _CreatePayload(**_bank_payload("technical", f"Position race {idx}?")),
                _actor(scenario["teacher_id"]),
            )
            await session.commit()
            bank_ids.append(item.id)

    async def import_one(bank_id: uuid.UUID) -> None:
        async with session_factory() as session:
            await authoring_service.import_question_bank_items(
                session,
                seeded["config_id"],
                [bank_id],
                _actor(scenario["teacher_id"]),
            )
            await session.commit()

    await asyncio.gather(*(import_one(bank_id) for bank_id in bank_ids))

    async with session_factory() as session:
        positions = (
            await session.execute(
                text(
                    "SELECT position FROM interview_questions "
                    "WHERE interview_config_id=:cfg AND deleted_at IS NULL "
                    "ORDER BY position"
                ),
                {"cfg": seeded["config_id"]},
            )
        ).scalars().all()
    assert positions == [1, 2]


@pytest.mark.asyncio
async def test_concurrent_sibling_adds_cannot_duplicate_a_prompt(
    session_factory: async_sessionmaker[AsyncSession],
    scenario: dict[str, Any],
) -> None:
    """Same prompt, DIFFERENT angles, racing through different anchors.

    ``uq_iq_bank_live_group_angle`` covers (group, angle) only, so it cannot
    catch this shape — the group advisory lock is the ONLY thing standing
    between the two prompt pre-checks. Without it both transactions read the
    group before either wrote, both passed, and the group ended up holding the
    same question text twice under two angles.
    """
    import asyncio

    from abridgeai.core.exceptions import ConflictError

    async with session_factory() as session:
        singleton = await authoring_service.add_to_question_bank(
            session,
            scenario["course_id"],
            _CreatePayload(**_bank_payload("technical", "Prompt race anchor?")),
            _actor(scenario["teacher_id"]),
        )
        await session.commit()
        group = await authoring_service.add_question_bank_sibling(
            session,
            scenario["course_id"],
            singleton.id,
            InterviewQuestionBankSiblingCreate(
                **_bank_payload("system_design", "Prompt race second?")
            ),
            _actor(scenario["teacher_id"]),
        )
        await session.commit()
    group_id = group[0].variant_group_id
    anchors = {item.question_type: item.id for item in group}

    shared_prompt = "Collided prompt under two angles?"

    async def add_via(anchor_id: uuid.UUID, angle: str) -> str:
        async with session_factory() as session:
            try:
                await authoring_service.add_question_bank_sibling(
                    session,
                    scenario["course_id"],
                    anchor_id,
                    InterviewQuestionBankSiblingCreate(
                        **_bank_payload(angle, shared_prompt)
                    ),
                    _actor(scenario["teacher_id"]),
                )
                await session.commit()
            except ConflictError:
                await session.rollback()
                return "conflict"
            return "ok"

    results = await asyncio.gather(
        add_via(anchors["technical"], "situational"),
        add_via(anchors["system_design"], "behavioral"),
    )
    assert sorted(results) == ["conflict", "ok"]

    async with session_factory() as session:
        prompts = (
            await session.execute(
                text(
                    "SELECT prompt_text FROM interview_question_bank_items "
                    "WHERE variant_group_id=:g AND deleted_at IS NULL"
                ),
                {"g": group_id},
            )
        ).scalars().all()
    assert len(prompts) == len(set(prompts)) == 3


@pytest.mark.asyncio
async def test_all_angles_question_count_capped_before_the_run_is_created(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
    scenario: dict[str, Any],
) -> None:
    """all_angles fans each logical question into 4 rows, so 50 meant 200 rows.

    The pipeline demands an EXACT hit on target_count and only accepts whole
    4-angle groups, so such a run burns every backfill round and then dies with
    "Generation underfilled". Reject it at enqueue time instead — before the
    generation_runs row and the ARQ job exist, so no LLM budget is spent.
    """
    seeded = await _create_published_config(
        engine,
        course_id=scenario["course_id"],
        module_id=scenario["module_id"],
        teacher_id=scenario["teacher_id"],
        questions=0,
        outcomes=0,
        status="draft",
    )
    arq_pool = SimpleNamespace(enqueue_job=AsyncMock(return_value=None))

    async with session_factory() as session:
        with pytest.raises(AppError, match="question_count_exceeds_variant_cap"):
            await authoring_service.start_generation_run(
                session,
                seeded["config_id"],
                _CreatePayload(
                    course_id=scenario["course_id"],
                    module_id=scenario["module_id"],
                    question_count=13,
                    variant_strategy="all_angles",
                ),
                _actor(scenario["teacher_id"]),
                arq_pool=arq_pool,
            )

    # Nothing was enqueued and no run row was left holding the dedup key.
    arq_pool.enqueue_job.assert_not_awaited()
    async with engine.connect() as conn:
        count = (
            await conn.execute(
                text("SELECT count(*) FROM generation_runs WHERE course_id = :c"),
                {"c": scenario["course_id"]},
            )
        ).scalar_one()
    assert count == 0


@pytest.mark.asyncio
async def test_all_angles_accepts_the_boundary_count(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
    scenario: dict[str, Any],
) -> None:
    """12 x 4 = 48 rows fits under the 50-row cap, so the boundary is allowed."""
    seeded = await _create_published_config(
        engine,
        course_id=scenario["course_id"],
        module_id=scenario["module_id"],
        teacher_id=scenario["teacher_id"],
        questions=0,
        outcomes=0,
        status="draft",
    )
    arq_pool = SimpleNamespace(enqueue_job=AsyncMock(return_value=None))

    async with session_factory() as session:
        run = await authoring_service.start_generation_run(
            session,
            seeded["config_id"],
            _CreatePayload(
                course_id=scenario["course_id"],
                module_id=scenario["module_id"],
                question_count=12,
                variant_strategy="all_angles",
            ),
            _actor(scenario["teacher_id"]),
            arq_pool=arq_pool,
        )
    assert run.config_json["question_count"] == 12
    arq_pool.enqueue_job.assert_awaited_once()


@pytest.mark.asyncio
async def test_non_variant_strategy_keeps_the_full_fifty(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
    scenario: dict[str, Any],
) -> None:
    """Only all_angles multiplies; legacy / role_only map 1:1 and keep 50."""
    seeded = await _create_published_config(
        engine,
        course_id=scenario["course_id"],
        module_id=scenario["module_id"],
        teacher_id=scenario["teacher_id"],
        questions=0,
        outcomes=0,
        status="draft",
    )
    arq_pool = SimpleNamespace(enqueue_job=AsyncMock(return_value=None))

    async with session_factory() as session:
        run = await authoring_service.start_generation_run(
            session,
            seeded["config_id"],
            _CreatePayload(
                course_id=scenario["course_id"],
                module_id=scenario["module_id"],
                question_count=50,
            ),
            _actor(scenario["teacher_id"]),
            arq_pool=arq_pool,
        )
    assert run.config_json["question_count"] == 50


@pytest.mark.asyncio
async def test_all_angles_cap_also_bounds_a_count_from_supplementary(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
    scenario: dict[str, Any],
) -> None:
    """question_count can also arrive inside supplementary_instructions JSON.

    The guard resolves through the same precedence as the pipeline, so that
    back door is bounded too rather than sneaking a 200-row target through.
    """
    seeded = await _create_published_config(
        engine,
        course_id=scenario["course_id"],
        module_id=scenario["module_id"],
        teacher_id=scenario["teacher_id"],
        questions=0,
        outcomes=0,
        status="draft",
    )
    arq_pool = SimpleNamespace(enqueue_job=AsyncMock(return_value=None))

    async with session_factory() as session:
        with pytest.raises(AppError, match="question_count_exceeds_variant_cap"):
            await authoring_service.start_generation_run(
                session,
                seeded["config_id"],
                _CreatePayload(
                    course_id=scenario["course_id"],
                    module_id=scenario["module_id"],
                    variant_strategy="all_angles",
                    supplementary_instructions='{"question_count": 40}',
                ),
                _actor(scenario["teacher_id"]),
                arq_pool=arq_pool,
            )
    arq_pool.enqueue_job.assert_not_awaited()
