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

import abridgeai.features.access_control.models  # noqa: F401  -- register users/orgs FK targets
import abridgeai.features.courses.models  # noqa: F401  -- register courses/modules/lessons FK targets
import abridgeai.features.identity.models  # noqa: F401  -- register users FK target
import abridgeai.features.interviews.models  # noqa: F401  -- T6.1 registers interview_* tables
import abridgeai.features.materials.models  # noqa: F401  -- register processing_jobs (audit FK)
import abridgeai.features.quizzes.models  # noqa: F401  -- register GenerationRun + quiz_attempts
from abridgeai.core.config import get_settings
from abridgeai.core.exceptions import ForbiddenError
from abridgeai.core.security import CurrentUser
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
from abridgeai.features.quizzes.models import GenerationRun


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
) -> dict[str, Any]:
    config_id = uuid.uuid4()
    question_ids = [uuid.uuid4() for _ in range(questions)]
    outcome_ids = [uuid.uuid4() for _ in range(outcomes)]
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO interview_configs "
                "(id, course_id, module_id, title, status, supported_modes, created_by) "
                "VALUES (:id, :c, :m, 'Pub Interview', 'published', 'text', :t)"
            ),
            {"id": config_id, "c": course_id, "m": module_id, "t": teacher_id},
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
        supported_modes="hybrid",
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
    payload = _CreatePayload(input_mode="text")
    async with session_factory() as session, session.begin():
        started = await taking_service.start_session(
            session, seeded["config_id"], payload, _actor(scenario["student_id"])
        )

    async with session_factory() as session, session.begin():
        result = await taking_service.take_session_step(
            session,
            started.id,
            "A thorough answer about the topic.",
            _actor(scenario["student_id"]),
        )

    assert result["followup_text"] is None
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
    payload = _CreatePayload(input_mode="text")
    async with session_factory() as session, session.begin():
        started = await taking_service.start_session(
            session, seeded["config_id"], payload, _actor(scenario["student_id"])
        )

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

    assert isinstance(run, GenerationRun)
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
    assert refreshed.pass_verdict is True
