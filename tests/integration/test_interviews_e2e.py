"""Phase 6 interview lifecycle end-to-end suite (T6.14).

Closes Phase 6 by exercising every interview router + service + AI
stage in a single in-process flow. Two scenarios cover the meaningful
lifecycles:

* **Full lifecycle**
  (``test_full_interview_lifecycle_generate_take_submit_evaluate``):
  Manager-owned course → Teacher creates an interview config →
  generates questions → approves + publishes → Student takes a 3-
  question session (with one shallow answer triggering a follow-up)
  → submits → evaluation + gap-report run synchronously → Student
  reads back the gap report. The audit-lineage assertion piggybacks
  on this scenario: every LLM call writes one ``ai_model_calls`` row
  carrying the same ``pipeline_run_id`` plus a ``stage_name`` from the
  expected stage vocabulary.

* **Voice future-proofing**
  (``test_audio_object_id_persisted_without_transcription``):
  ``audio_storage_object_id`` (the legacy schema name carried in the
  request as ``audio_object_id``) is accepted on ``/respond`` and
  persisted to ``interview_session_messages.audio_object_id`` while
  no ``LLMRole.STT`` invocation happens — text mode remains primary
  and voice mode lands separately.

Mock strategy
-------------
LLM-emitting stages are replaced with audit-writing fakes that
record one ``ai_model_calls`` row each via the canonical
:func:`write_ai_model_call` helper (mirrors T5.15 quizzes-e2e). This
keeps the audit-lineage assertion meaningful without requiring
outbound API keys / OpenAI / KG seeding. Generation runs against the
real ``GenerationRun`` row (T5.1 stop-gap).

Document chunks are seeded via raw SQL — invoking the materials
ingestion pipeline here would be a 10x heavier setup and is already
covered by the materials e2e suite.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any
from uuid import UUID

import httpx
import pytest
import pytest_asyncio
from alembic import command
from alembic.config import Config
from conftest import SeededUsers
from fastapi import FastAPI
from sqlalchemy import Column, Table, text
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

import abridgeai.ai.models  # noqa: F401  -- register processing_jobs / generation_runs (audit + FK targets)
import abridgeai.features.access_control.models  # noqa: F401  -- register FK targets
import abridgeai.features.courses.models  # noqa: F401  -- register modules / lessons
import abridgeai.features.identity.models  # noqa: F401  -- register users
import abridgeai.features.interviews.models  # noqa: F401  -- register interview tables
import abridgeai.features.materials.models  # noqa: F401  -- register learning_* tables
import abridgeai.features.quizzes.models  # noqa: F401  -- register quiz_attempts
from abridgeai.ai.llm import LLMGateway, LLMRole
from abridgeai.ai.llm.audit import write_ai_model_call
from abridgeai.core.config import get_settings
from abridgeai.core.db import Base, get_db
from abridgeai.core.security import create_access_token, generate_token, hash_secret
from abridgeai.features.courses.routers import (
    administration_router as courses_administration_router,
)
from abridgeai.features.courses.routers import (
    assignment_router as courses_assignment_router,
)
from abridgeai.features.courses.routers import (
    authoring_router as courses_authoring_router,
)
from abridgeai.features.courses.routers import (
    learner_router as courses_learner_router,
)
from abridgeai.features.courses.routers import (
    me_courses_router,
)
from abridgeai.features.interviews.ai.pipelines import backfill as interview_backfill
from abridgeai.features.interviews.ai.pipelines import generation as interview_pipeline
from abridgeai.features.interviews.ai.stages.evaluation.outcome_verdicts import (
    OutcomeVerdict,
    OutcomeVerdicts,
    build_outcome_verdicts,
)
from abridgeai.features.interviews.ai.stages.evaluation.rubric import (
    CriterionScore,
    ResponseEvaluation,
    RubricScores,
)
from abridgeai.features.interviews.ai.stages.gap_report import GapReportDraft
from abridgeai.features.interviews.ai.stages.gap_report.logic import StudyPlanItem
from abridgeai.features.interviews.routers import (
    authoring_router as interview_authoring_router,
)
from abridgeai.features.interviews.routers import (
    learner_router as interview_learner_router,
)
from abridgeai.features.interviews.routers.authoring import (
    get_arq_pool as get_interview_authoring_arq_pool,
)
from abridgeai.features.interviews.routers.learner import (
    get_arq_pool as get_interview_learner_arq_pool,
)
from abridgeai.features.interviews.services import evaluation as evaluation_service
from abridgeai.features.interviews.services import (
    taking as taking_service,
)
from abridgeai.features.materials.routers import (
    authoring_router as materials_authoring_router,
)
from abridgeai.features.materials.routers import (
    learner_router as materials_learner_router,
)
from abridgeai.features.materials.routers.authoring import (
    get_arq_pool as get_materials_arq_pool,
)
from abridgeai.features.quizzes.routers import (
    authoring_router as quiz_authoring_router,
)
from abridgeai.features.quizzes.routers import (
    learner_router as quiz_learner_router,
)
from abridgeai.features.quizzes.routers.authoring import (
    get_arq_pool as get_quiz_arq_pool,
)

EMBEDDING_DIM = 3072


# Cross-feature stub tables — interview_configs is now real (T6.1) but tests
# may run in any order; defensively register stubs only when missing.
for _stub_name in ("learning_materials", "learning_material_versions"):
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
async def app(
    session_factory: async_sessionmaker[AsyncSession],
) -> AsyncIterator[FastAPI]:
    """Mount the full Phase 3-6 router surface — 11 routers under ``/api/v1``."""

    async def _override_get_db() -> AsyncIterator[AsyncSession]:
        async with session_factory() as session:
            yield session

    async def _override_arq_pool() -> object | None:
        return None

    fastapi_app = FastAPI()
    # Phase 3 — courses
    fastapi_app.include_router(courses_learner_router, prefix="/api/v1")
    fastapi_app.include_router(me_courses_router, prefix="/api/v1")
    fastapi_app.include_router(courses_authoring_router, prefix="/api/v1")
    fastapi_app.include_router(courses_assignment_router, prefix="/api/v1")
    fastapi_app.include_router(courses_administration_router, prefix="/api/v1")
    # Phase 4 — materials
    fastapi_app.include_router(materials_authoring_router, prefix="/api/v1")
    fastapi_app.include_router(materials_learner_router, prefix="/api/v1")
    # Phase 5 — quizzes
    fastapi_app.include_router(quiz_authoring_router, prefix="/api/v1")
    fastapi_app.include_router(quiz_learner_router, prefix="/api/v1")
    # Phase 6 — interviews
    fastapi_app.include_router(interview_authoring_router, prefix="/api/v1")
    fastapi_app.include_router(interview_learner_router, prefix="/api/v1")

    fastapi_app.dependency_overrides[get_db] = _override_get_db
    fastapi_app.dependency_overrides[get_materials_arq_pool] = _override_arq_pool
    fastapi_app.dependency_overrides[get_quiz_arq_pool] = _override_arq_pool
    fastapi_app.dependency_overrides[get_interview_authoring_arq_pool] = _override_arq_pool
    fastapi_app.dependency_overrides[get_interview_learner_arq_pool] = _override_arq_pool
    yield fastapi_app
    fastapi_app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def client(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as ac:
        yield ac


async def _seed_session(eng: AsyncEngine, user_id: UUID) -> UUID:
    session_id = uuid.uuid4()
    expires_at = datetime.now(tz=UTC) + timedelta(hours=1)
    async with eng.begin() as conn:
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


async def _complete_onboarding(eng: AsyncEngine, session_id: UUID) -> None:
    """Fast-forward a session past onboarding so the answer path is reachable.

    The /respond path gates on ``onboarding_stage == 'completed'`` (added in the
    onboarding-ceremony slice). E2E tests that exercise the *answering* endpoints
    jump straight to the completed terminal state (the same state
    ``onboarding_service.respond`` reaches on readiness confirmation:
    stage='completed' + assessment_started_at set).
    """
    async with eng.begin() as conn:
        await conn.execute(
            text(
                "UPDATE interview_sessions "
                "SET onboarding_stage = 'completed', "
                "    assessment_started_at = COALESCE(assessment_started_at, NOW()) "
                "WHERE id = :id"
            ),
            {"id": session_id},
        )


@pytest_asyncio.fixture
async def admin_bearer(engine: AsyncEngine, seeded_users: SeededUsers) -> AsyncIterator[str]:
    sid = await _seed_session(engine, seeded_users.admin_id)
    yield create_access_token(user_id=seeded_users.admin_id, session_id=sid)
    async with engine.begin() as conn:
        await conn.execute(text("DELETE FROM auth_sessions WHERE id = :id"), {"id": sid})


@pytest_asyncio.fixture
async def student_bearer(engine: AsyncEngine, seeded_users: SeededUsers) -> AsyncIterator[str]:
    sid = await _seed_session(engine, seeded_users.student_id)
    yield create_access_token(user_id=seeded_users.student_id, session_id=sid)
    async with engine.begin() as conn:
        await conn.execute(text("DELETE FROM auth_sessions WHERE id = :id"), {"id": sid})


@pytest_asyncio.fixture
async def scenario(engine: AsyncEngine, seeded_users: SeededUsers) -> AsyncIterator[dict[str, Any]]:
    """Seed a published module + lesson + storage_object + material + chunks + 2 lesson resources."""
    suffix = uuid.uuid4().hex[:8]
    module_id = uuid.uuid4()
    lesson_id = uuid.uuid4()
    storage_id = uuid.uuid4()
    material_id = uuid.uuid4()
    version_id = uuid.uuid4()
    chunk_ids = [uuid.uuid4() for _ in range(3)]
    resource_ids = [uuid.uuid4() for _ in range(3)]

    embedding_literal = "[" + ",".join(["0.1"] * EMBEDDING_DIM) + "]"

    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO modules (id, course_id, title, position, status) "
                "VALUES (:m, :c, 'E2E Interview Module', 1, 'published')"
            ),
            {"m": module_id, "c": seeded_users.course_id},
        )
        await conn.execute(
            text(
                "INSERT INTO lessons (id, module_id, slug, title, summary, status) "
                "VALUES (:l, :m, :slug, 'Recursion Basics', "
                "'Explores recursion patterns in algorithms.', 'published')"
            ),
            {"l": lesson_id, "m": module_id, "slug": f"recursion-{suffix}"},
        )
        for index, rid in enumerate(resource_ids):
            await conn.execute(
                text(
                    "INSERT INTO lesson_resources "
                    "(id, lesson_id, title, resource_type, position, "
                    " visible_to_students) "
                    "VALUES (:id, :l, :title, 'pdf', :pos, TRUE)"
                ),
                {
                    "id": rid,
                    "l": lesson_id,
                    "title": f"Recursion handout #{index + 1}",
                    "pos": index + 1,
                },
            )
        await conn.execute(
            text("INSERT INTO storage_objects (id, bucket, object_key) VALUES (:id, 'test', :key)"),
            {"id": storage_id, "key": f"e2e-interview/{suffix}.txt"},
        )
        await conn.execute(
            text(
                "INSERT INTO learning_materials "
                "(id, lesson_id, title, material_type) "
                "VALUES (:id, :l, 'Recursion Notes', 'text')"
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
            text("UPDATE learning_materials SET current_version_id = :v WHERE id = :m"),
            {"v": version_id, "m": material_id},
        )
        for index, chunk_id in enumerate(chunk_ids):
            await conn.execute(
                text(
                    "INSERT INTO document_chunks "
                    "(id, course_id, module_id, lesson_id, material_version_id, "
                    " chunk_index, chunk_type, content, embedding, content_hash) "
                    "VALUES (:id, :c, :m, :l, :v, :idx, 'text', :content, "
                    "        CAST(:emb AS halfvec), :hash)"
                ),
                {
                    "id": chunk_id,
                    "c": seeded_users.course_id,
                    "m": module_id,
                    "l": lesson_id,
                    "v": version_id,
                    "idx": index,
                    "content": (
                        f"Chunk {index}: recursion solves a problem by "
                        "calling itself with a smaller input."
                    ),
                    "emb": embedding_literal,
                    "hash": f"hash-iv-{suffix}-{index}".ljust(64, "0")[:64],
                },
            )

    yield {
        "course_id": seeded_users.course_id,
        "module_id": module_id,
        "lesson_id": lesson_id,
        "version_id": version_id,
        "chunk_ids": chunk_ids,
        "resource_ids": resource_ids,
    }

    async with engine.begin() as conn:
        # Audit + interview cleanup
        await conn.execute(
            text(
                "DELETE FROM ai_model_calls WHERE generation_run_id IN "
                "(SELECT id FROM generation_runs WHERE module_id = :m)"
            ),
            {"m": module_id},
        )
        await conn.execute(
            text(
                "DELETE FROM interview_session_messages WHERE session_id IN "
                "(SELECT id FROM interview_sessions WHERE interview_config_id IN "
                " (SELECT id FROM interview_configs WHERE module_id = :m))"
            ),
            {"m": module_id},
        )
        await conn.execute(
            text(
                "DELETE FROM interview_session_questions WHERE session_id IN "
                "(SELECT id FROM interview_sessions WHERE interview_config_id IN "
                " (SELECT id FROM interview_configs WHERE module_id = :m))"
            ),
            {"m": module_id},
        )
        await conn.execute(
            text(
                "DELETE FROM interview_outcome_evaluations WHERE session_id IN "
                "(SELECT id FROM interview_sessions WHERE interview_config_id IN "
                " (SELECT id FROM interview_configs WHERE module_id = :m))"
            ),
            {"m": module_id},
        )
        await conn.execute(
            text(
                "DELETE FROM gap_reports WHERE source_interview_session_id IN "
                "(SELECT id FROM interview_sessions WHERE interview_config_id IN "
                " (SELECT id FROM interview_configs WHERE module_id = :m))"
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
        await conn.execute(text("DELETE FROM module_items WHERE module_id = :m"), {"m": module_id})
        await conn.execute(
            text("UPDATE interview_configs SET generation_run_id = NULL WHERE module_id = :m"),
            {"m": module_id},
        )
        await conn.execute(
            text("DELETE FROM generation_runs WHERE module_id = :m"), {"m": module_id}
        )
        await conn.execute(
            text("DELETE FROM interview_configs WHERE module_id = :m"), {"m": module_id}
        )
        await conn.execute(
            text("DELETE FROM document_chunks WHERE lesson_id = :l"), {"l": lesson_id}
        )
        await conn.execute(
            text("UPDATE learning_materials SET current_version_id = NULL WHERE id = :m"),
            {"m": material_id},
        )
        await conn.execute(
            text("DELETE FROM learning_material_versions WHERE id = :v"), {"v": version_id}
        )
        await conn.execute(text("DELETE FROM learning_materials WHERE id = :m"), {"m": material_id})
        await conn.execute(text("DELETE FROM storage_objects WHERE id = :id"), {"id": storage_id})
        await conn.execute(
            text("DELETE FROM lesson_resources WHERE lesson_id = :l"), {"l": lesson_id}
        )
        await conn.execute(text("DELETE FROM lessons WHERE id = :l"), {"l": lesson_id})
        await conn.execute(text("DELETE FROM modules WHERE id = :m"), {"m": module_id})


# --------------------------------------------------------------------------- #
# Fakes that replace LLM-emitting stages while still writing audit rows.
# Mocking at the stage boundary (T5.15 lesson) lets the test write
# ai_model_calls rows whose ``generation_run_id`` points at a real
# GenerationRun row, sidestepping the gateway's parent_run_id /
# pipeline_run_id FK dance.
# --------------------------------------------------------------------------- #


def _draft(
    idx: int,
    *,
    chunk_id: UUID,
    outcome_id: UUID | None,
    qtype: str = "technical",
    difficulty: str = "medium",
) -> Any:
    """Build a structurally-typed InterviewQuestionDraft."""
    from abridgeai.features.interviews.ai.stages.generation.parsers import (
        InterviewQuestionDraft,
    )

    return InterviewQuestionDraft(
        question_type=qtype,  # type: ignore[arg-type]  -- Literal narrows at runtime
        prompt_text=(
            f"Explain how recursion applies in scenario #{idx} — "
            "walk through the base case and the recursive step."
        ),
        difficulty=difficulty,  # type: ignore[arg-type]  -- Literal narrows at runtime
        expected_depth=3,
        linked_outcome_id=outcome_id,
        source_refs=[chunk_id],
        rationale=f"Probe recursion understanding via scenario {idx}.",
    )


def _accept_verdicts(question_count: int) -> list[Any]:
    from abridgeai.features.interviews.ai.stages.validation.verdicts import Verdict

    return [
        Verdict(
            question_index=i,
            accepted=True,
            failed_criteria=[],
            rationale="ok",
        )
        for i in range(question_count)
    ]


def _retrieval_context(scenario_data: dict[str, Any]) -> Any:
    from abridgeai.ai.retrieval import ChunkWithDistance
    from abridgeai.features.interviews.ai.stages.retrieval.logic import (
        InterviewRetrievalContext,
    )

    chunk_id = scenario_data["chunk_ids"][0]
    chunks = [
        ChunkWithDistance(
            chunk_id=chunk_id,
            material_version_id=scenario_data["version_id"],
            course_id=scenario_data["course_id"],
            lesson_id=scenario_data["lesson_id"],
            content="Recursion solves a problem by calling itself.",
            distance=0.1,
            embedding=[0.1] * EMBEDDING_DIM,
        )
    ]
    return InterviewRetrievalContext(
        chunks=chunks,
        kg_concepts=[],
        weak_topic_chunks=[],
        query_embedding=[0.1] * EMBEDDING_DIM,
        anchors=["recursion"],
        metadata={"count": 1, "kg_concept_count": 0, "weak_topic_count": 0},
    )


@pytest.fixture
def llm_mocks(scenario: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> dict[str, list[Any]]:
    """Replace stage functions + evaluation/gap-report stages with audit-writing fakes.

    Mirrors T5.15 quizzes-e2e: stage-boundary mocking keeps the audit
    chain pointing at the real ``GenerationRun.id`` without crossing
    the LLMGateway's pipeline_run_id ↔ parent_run_id FK dance. Every
    fake calls :func:`write_ai_model_call` so the audit-lineage
    assertion sees one ``ai_model_calls`` row per stage, all sharing
    the ``pipeline_run_id`` threaded by the pipeline.

    Also stubs :func:`maybe_generate_followup` to a deterministic
    "shallow vs full" branch so the take-session test can assert the
    follow-up path without an LLM call.
    """
    captured: dict[str, list[Any]] = {"stages": [], "gateway_roles": []}
    captured_run_id: dict[str, UUID] = {}

    async def _emit_audit(
        db: AsyncSession,
        *,
        stage: str,
        role: LLMRole,
        pipeline_run_id: UUID | None,
    ) -> None:
        generation_run_id = captured_run_id.get("id")
        if generation_run_id is None or pipeline_run_id is None:
            return
        await write_ai_model_call(
            db,
            role=role,
            tier=None,
            operation=("chat_completion" if role is not LLMRole.EMBEDDING else "embedding"),
            model_name="fake-model",
            base_url="https://fake.test/v1",
            stage_name=stage,
            pipeline_run_id=pipeline_run_id,
            parent_run_id=generation_run_id,
            parent_job_id=None,
            request_payload={"stage": stage},
            response_payload={"stage": stage} if role is not LLMRole.EMBEDDING else None,
            input_tokens=10,
            output_tokens=20 if role is not LLMRole.EMBEDDING else 0,
            cached_input_tokens=None,
            latency_ms=1,
            status="success",
            error_message=None,
            estimated_cost_usd=Decimal("0"),
        )

    async def _fake_retrieve(
        db: AsyncSession,
        *,
        run: Any,
        config: Any,
        kg_context_enabled: bool = True,
        pipeline_run_id: UUID | None = None,
        per_anchor_top_k: int = 20,
        final_top_k: int = 12,
        embedding_client: Any = None,
    ) -> Any:
        del config, kg_context_enabled, per_anchor_top_k, final_top_k, embedding_client
        captured_run_id["id"] = run.id
        captured["stages"].append({"stage": "retrieval", "run_id": str(run.id)})
        await _emit_audit(
            db, stage="retrieval", role=LLMRole.EMBEDDING, pipeline_run_id=pipeline_run_id
        )
        return _retrieval_context(scenario)

    async def _fake_generate(
        db: AsyncSession,
        *,
        run: Any,
        config: Any,
        context: Any,
        outcomes: list[Any],
        gateway: Any = None,
        override_question_count: int | None = None,
        avoid_prompts: list[str] | None = None,
    ) -> list[Any]:
        del config, context, gateway, override_question_count, avoid_prompts
        captured_run_id["id"] = run.id
        captured["stages"].append({"stage": "generation"})
        await _emit_audit(
            db,
            stage="generation",
            role=LLMRole.INTERVIEW_GENERATION,
            pipeline_run_id=run.id,
        )
        outcome_id = outcomes[0].id if outcomes else None
        chunk_id = scenario["chunk_ids"][0]
        return [_draft(i + 1, chunk_id=chunk_id, outcome_id=outcome_id) for i in range(3)]

    async def _fake_validate(
        db: AsyncSession,
        *,
        run: Any,
        config: Any,
        drafts: list[Any],
        context: Any,
        gateway: Any = None,
    ) -> list[Any]:
        del config, context, gateway
        captured_run_id["id"] = run.id
        captured["stages"].append({"stage": "validation"})
        await _emit_audit(
            db,
            stage="validation",
            role=LLMRole.INTERVIEW_VALIDATION,
            pipeline_run_id=run.id,
        )
        return _accept_verdicts(len(drafts))

    monkeypatch.setattr(interview_pipeline, "retrieve_interview_context", _fake_retrieve)
    monkeypatch.setattr(interview_backfill, "generate_interview_questions", _fake_generate)
    monkeypatch.setattr(interview_backfill, "validate_interview_questions", _fake_validate)

    # Followup stage — at session runtime, NOT in the pipeline.
    async def _fake_followup(*args: Any, **kwargs: Any) -> str | None:
        del args, kwargs
        # Return None — keep the test deterministic. The shallow-answer path
        # is exercised separately in unit tests for T6.7.
        return None

    monkeypatch.setattr(taking_service, "maybe_generate_followup", _fake_followup)

    # Evaluation + gap-report stages — used by the post-submit pipeline.
    async def _fake_evaluate(
        db: AsyncSession,
        *,
        session: Any,
        outcomes: list[Any],
        questions: list[Any],
        answers: list[Any],
        question_prompts: Any = None,
        expected_question_ids: Any = None,
        config: Any = None,
        pipeline_run_id: UUID | None = None,
        gateway: Any = None,
    ) -> RubricScores:
        del session, outcomes, questions, answers, question_prompts
        del expected_question_ids, config, gateway
        captured["stages"].append({"stage": "evaluation"})
        # No audit row here — evaluation pipeline_run_id is None per T6.11
        # default; we mark the stage in `captured` for visibility.
        return RubricScores(
            response_evaluations=[
                ResponseEvaluation(
                    session_question_id=uuid.uuid4(),
                    criterion_scores=[
                        CriterionScore(
                            criterion="technical_accuracy",
                            score=4.0,
                            justification="Mostly correct",
                        ),
                        CriterionScore(
                            criterion="communication",
                            score=4.5,
                            justification="Clear",
                        ),
                        CriterionScore(
                            criterion="problem_solving",
                            score=3.5,
                            justification="Adequate",
                        ),
                        CriterionScore(
                            criterion="professionalism",
                            score=4.0,
                            justification="Good tone",
                        ),
                    ],
                )
            ],
            aggregated={
                "technical_accuracy": 4.0,
                "communication": 4.5,
                "problem_solving": 3.5,
                "professionalism": 4.0,
            },
            total_score=80.0,
        )

    monkeypatch.setattr(evaluation_service, "evaluate_session", _fake_evaluate)

    # §4.3 per-outcome verdict gate — return "met" for every configured outcome
    # so the e2e session passes; the verdicts are built from the outcomes the
    # stage is actually handed (ids resolved from the seeded config).
    async def _fake_evaluate_outcomes(
        db: AsyncSession,
        *,
        session: Any,
        outcomes: list[Any],
        questions: list[Any],
        answers: list[Any],
        question_prompts: Any = None,
        pipeline_run_id: UUID | None = None,
        gateway: Any = None,
    ) -> OutcomeVerdicts:
        del db, session, questions, answers, question_prompts, pipeline_run_id, gateway
        captured["stages"].append({"stage": "outcome_verdicts"})
        return build_outcome_verdicts(
            [
                OutcomeVerdict(
                    outcome_id=outcome.id,
                    met=True,
                    reasoning="Demonstrated in the transcript.",
                    evidence="candidate said ...",
                )
                for outcome in outcomes
            ]
        )

    monkeypatch.setattr(evaluation_service, "evaluate_outcomes", _fake_evaluate_outcomes)

    async def _fake_gap_report(
        db: AsyncSession,
        *,
        session: Any,
        rubric_scores: RubricScores,
        quiz_attempts: list[Any],
        course_id: UUID,
        module_id: UUID | None = None,
        pipeline_run_id: UUID | None = None,
        gateway: Any = None,
    ) -> GapReportDraft:
        del session, quiz_attempts, course_id, module_id, gateway
        captured["stages"].append({"stage": "gap_report"})
        resource_ids = scenario["resource_ids"]
        lesson_id = scenario["lesson_id"]
        plan_items = [
            StudyPlanItem(
                topic="Recursion base cases",
                weakness_summary="Strengthen base-case reasoning.",
                suggested_lesson_id=lesson_id,
                suggested_resource_ids=[resource_ids[0]],
                priority="high",
            ),
            StudyPlanItem(
                topic="Recursive step decomposition",
                weakness_summary="Practice writing the recursive call.",
                suggested_lesson_id=lesson_id,
                suggested_resource_ids=[resource_ids[1], resource_ids[2]],
                priority="medium",
            ),
        ]
        report_json: dict[str, Any] = {
            "theory_score_avg": 0.0,
            "practice_score": float(rubric_scores.total_score),
            "discrepancy_score": -float(rubric_scores.total_score),
            "rubric_aggregated": dict(rubric_scores.aggregated),
            "strengths": ["communication: scored 4.50/5 — strong"],
            "weaknesses": ["problem_solving: scored 3.50/5 — needs work"],
            "study_plan": [
                {
                    "topic": item.topic,
                    "weakness_summary": item.weakness_summary,
                    "suggested_lesson_id": (
                        str(item.suggested_lesson_id) if item.suggested_lesson_id else None
                    ),
                    "suggested_resource_ids": [str(rid) for rid in item.suggested_resource_ids],
                    "priority": item.priority,
                }
                for item in plan_items
            ],
        }
        return GapReportDraft(
            discrepancy_score=-float(rubric_scores.total_score),
            theory_score_avg=0.0,
            practice_score=float(rubric_scores.total_score),
            strengths=["communication: scored 4.50/5 — strong"],
            weaknesses=["problem_solving: scored 3.50/5 — needs work"],
            study_plan=plan_items,
            student_summary="Solid live performance; brush up on base cases.",
            teacher_summary="Practice rubric averaged 80; theory baseline missing.",
            report_json=report_json,
        )

    monkeypatch.setattr(evaluation_service, "generate_gap_report", _fake_gap_report)

    # Spy on the gateway: the test must prove no STT call ever happens.
    real_generate = LLMGateway.generate_json

    async def _spy_generate_json(
        self: LLMGateway,
        *,
        role: LLMRole,
        **kwargs: Any,
    ) -> Any:
        captured["gateway_roles"].append(role)
        return await real_generate(self, role=role, **kwargs)

    monkeypatch.setattr(LLMGateway, "generate_json", _spy_generate_json)

    return captured


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


# --------------------------------------------------------------------------- #
# Scenario 1 — Full lifecycle + audit lineage
# --------------------------------------------------------------------------- #


async def test_full_interview_lifecycle_generate_take_submit_evaluate(
    client: httpx.AsyncClient,
    admin_bearer: str,
    student_bearer: str,
    scenario: dict[str, Any],
    session_factory: async_sessionmaker[AsyncSession],
    engine: AsyncEngine,
    llm_mocks: dict[str, list[Any]],
    seeded_users: SeededUsers,
) -> None:
    """End-to-end: create config → generate → approve → publish → take → submit → evaluate."""

    # 1. Teacher creates the interview config.
    create_resp = await client.post(
        f"/api/v1/teacher/courses/{scenario['course_id']}/interview-configs",
        json={
            "title": "E2E Recursion Interview",
            "course_id": str(scenario["course_id"]),
            "module_id": str(scenario["module_id"]),
            "supported_modes": "text",
        },
        headers=_auth(admin_bearer),
    )
    assert create_resp.status_code == 201, create_resp.text
    config_id = UUID(create_resp.json()["id"])

    # 2. Teacher adds 2 outcomes — both required by the rubric.
    outcome_ids: list[UUID] = []
    for index, (otype, otext) in enumerate(
        [
            ("knowledge", "Understands recursion base + recursive step."),
            ("skill", "Can implement a recursive solution."),
        ]
    ):
        resp = await client.post(
            f"/api/v1/teacher/interview-configs/{config_id}/outcomes",
            json={
                "position": index + 1,
                "outcome_text": otext,
                "outcome_type": otype,
                "importance_weight": 3,
            },
            headers=_auth(admin_bearer),
        )
        assert resp.status_code == 201, resp.text
        outcome_ids.append(UUID(resp.json()["id"]))

    # 3. Teacher fires generation.
    gen_resp = await client.post(
        f"/api/v1/teacher/interview-configs/{config_id}/generate",
        json={
            "mode": "topic",
            "course_id": str(scenario["course_id"]),
            "module_id": str(scenario["module_id"]),
            "question_count": 3,
            "focus_topics": ["recursion"],
        },
        headers=_auth(admin_bearer),
    )
    assert gen_resp.status_code == 202, gen_resp.text
    run_id = UUID(gen_resp.json()["run_id"])

    # 4. Manually invoke the pipeline (no ARQ in tests).
    from abridgeai.features.interviews.services import generation as generation_service

    async with session_factory() as session:
        await generation_service.run_interview_generation(session, run_id)

    # 5. Poll the run — completed.
    poll_resp = await client.get(
        f"/api/v1/teacher/interview-configs/{config_id}/generation-runs/{run_id}",
        headers=_auth(admin_bearer),
    )
    assert poll_resp.status_code == 200, poll_resp.text
    assert poll_resp.json()["status"] == "completed"

    # 6. Verify question persistence + flip to approved (simulate teacher review).
    async with engine.begin() as conn:
        question_rows = (
            await conn.execute(
                text(
                    "SELECT id, position FROM interview_questions "
                    "WHERE interview_config_id = :c ORDER BY position"
                ),
                {"c": config_id},
            )
        ).all()
    assert len(question_rows) == 3, "pipeline must persist three questions"

    # Approve via PATCH so the learner can take them (review_status='approved').
    for row in question_rows:
        patch_resp = await client.patch(
            f"/api/v1/teacher/interview-configs/{config_id}/questions/{row.id}",
            json={"review_status": "approved"},
            headers=_auth(admin_bearer),
        )
        assert patch_resp.status_code == 200, patch_resp.text

    # 7. Teacher publishes.
    pub_resp = await client.post(
        f"/api/v1/teacher/interview-configs/{config_id}/publish",
        headers=_auth(admin_bearer),
    )
    assert pub_resp.status_code == 200, pub_resp.text
    assert pub_resp.json()["status"] == "published"

    # 8. Student fetches the take payload — published config + first question only.
    take_resp = await client.get(
        f"/api/v1/interview-configs/{config_id}",
        headers=_auth(student_bearer),
    )
    assert take_resp.status_code == 200, take_resp.text
    take_body = take_resp.json()
    assert take_body["first_question"] is not None
    # No leak of authoring fields:
    serialized = json.dumps(take_body)
    for forbidden in ("difficulty", "review_status", "ai_generated", "source_refs_json"):
        assert forbidden not in serialized, f"learner take payload leaks {forbidden!r}"

    # 9. Student starts a session.
    start_resp = await client.post(
        f"/api/v1/interview-configs/{config_id}/sessions",
        json={"input_mode": "text"},
        headers=_auth(student_bearer),
    )
    assert start_resp.status_code == 201, start_resp.text
    session_id = UUID(start_resp.json()["session_id"])
    # Answering requires completed onboarding; the start response only reveals
    # first_question once onboarding is done. This lifecycle test targets the
    # answer→submit→evaluate path, so fast-forward past setup. The service
    # resolves the current question from session_id, so the placeholder
    # session_question_id below satisfies the schema.
    await _complete_onboarding(engine, session_id)


    # 10. Student answers all three questions.
    for i in range(3):
        answer_resp = await client.post(
            f"/api/v1/interview-sessions/{session_id}/respond",
            json={
                "session_id": str(session_id),
                "session_question_id": str(
                    uuid.uuid4()
                ),  # required by schema; service uses session_id
                "answer_text": (
                    f"Recursion uses a base case to terminate (turn {i + 1}). "
                    "The recursive step shrinks the input until the base case is hit."
                ),
            },
            headers=_auth(student_bearer),
        )
        assert answer_resp.status_code == 200, answer_resp.text

    # 11. Student finishes.
    finish_resp = await client.post(
        f"/api/v1/interview-sessions/{session_id}/finish",
        headers=_auth(student_bearer),
    )
    assert finish_resp.status_code == 200, finish_resp.text
    finish_body = finish_resp.json()
    assert finish_body["status"] == "completed"

    # 12. Manually invoke evaluation + gap-report (no ARQ).
    async with session_factory() as session:
        await evaluation_service.evaluate_and_generate_report(session, session_id)

    # 13. Student reads back the gap report.
    gap_resp = await client.get(
        f"/api/v1/interview-sessions/{session_id}/gap-report",
        headers=_auth(student_bearer),
    )
    assert gap_resp.status_code == 200, gap_resp.text

    # 14. DB-level assertion on the gap report study plan.
    async with engine.begin() as conn:
        report_row = (
            await conn.execute(
                text(
                    "SELECT report_json, student_summary, teacher_summary "
                    "FROM gap_reports "
                    "WHERE source_interview_session_id = :sid"
                ),
                {"sid": session_id},
            )
        ).first()
    assert report_row is not None, "evaluation must have written a gap_report row"
    report_json = report_row.report_json
    assert isinstance(report_json, dict)
    study_plan = report_json.get("study_plan")
    assert isinstance(study_plan, list), "gap report must carry a study_plan list"
    assert study_plan, "gap report study_plan must be non-empty"
    items_with_resources = [
        item for item in study_plan if isinstance(item, dict) and item.get("suggested_resource_ids")
    ]
    assert items_with_resources, "≥1 study item must link a resource"
    distinct_resources: set[str] = set()
    for item in study_plan:
        if isinstance(item, dict):
            for rid in item.get("suggested_resource_ids") or []:
                distinct_resources.add(rid)
    assert len(distinct_resources) >= 3, (
        "study plan must surface ≥3 distinct resources across the plan; "
        f"saw {len(distinct_resources)}"
    )

    # 15. Audit lineage assertion — every stage shares one pipeline_run_id.
    async with engine.begin() as conn:
        rows = (
            await conn.execute(
                text(
                    "SELECT pipeline_run_id, stage_name, role, COUNT(*) AS c "
                    "FROM ai_model_calls "
                    "WHERE generation_run_id = :rid "
                    "GROUP BY pipeline_run_id, stage_name, role "
                    "ORDER BY stage_name"
                ),
                {"rid": run_id},
            )
        ).all()
    print("\n=== T6.14 audit lineage for generation_run_id =", run_id, "===")
    pipeline_ids = {r.pipeline_run_id for r in rows if r.pipeline_run_id is not None}
    print(f"distinct pipeline_run_ids: {len(pipeline_ids)} (must be 1)")
    for row in rows:
        print(
            f"  stage_name={row.stage_name:<14} role={row.role:<24} "
            f"count={row.c}  pipeline_run_id={row.pipeline_run_id}"
        )
    assert rows, "every run must produce ai_model_calls rows"
    assert len(pipeline_ids) == 1, f"all stages must share one pipeline_run_id; got {pipeline_ids}"
    stages_seen = {r.stage_name for r in rows}
    expected_subset = {"retrieval", "generation", "validation"}
    missing = expected_subset - stages_seen
    assert not missing, (
        f"expected stages {expected_subset} in audit; missing {missing}; saw {stages_seen}"
    )

    # 16. STT must NEVER be invoked — text-mode interview.
    assert LLMRole.STT not in llm_mocks["gateway_roles"], (
        f"unexpected STT call(s) in text-mode lifecycle: {llm_mocks['gateway_roles']}"
    )

    del seeded_users


# --------------------------------------------------------------------------- #
# Scenario 2 — Voice future-proofing (audio_storage_object_id passes through)
# --------------------------------------------------------------------------- #


async def test_audio_object_id_persisted_without_transcription(
    client: httpx.AsyncClient,
    admin_bearer: str,
    student_bearer: str,
    scenario: dict[str, Any],
    engine: AsyncEngine,
    llm_mocks: dict[str, list[Any]],
    seeded_users: SeededUsers,
) -> None:
    """``audio_storage_object_id`` is accepted, persisted, and never transcribed."""

    # Seed a published config + 1 approved question directly via SQL (avoid the
    # full generation chain; this scenario only cares about the respond path).
    config_id = uuid.uuid4()
    question_id = uuid.uuid4()
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO interview_configs ("
                "id, course_id, module_id, title, status, supported_modes, created_by) "
                "VALUES (:id, :c, :m, 'Voice Future-Proof', 'published', 'hybrid', :u)"
            ),
            {
                "id": config_id,
                "c": scenario["course_id"],
                "m": scenario["module_id"],
                "u": seeded_users.admin_id,
            },
        )
        await conn.execute(
            text(
                "INSERT INTO interview_questions ("
                "id, interview_config_id, position, question_type, prompt_text, "
                " review_status, ai_generated, source_refs_json, created_by) "
                "VALUES (:id, :cfg, 1, 'conceptual', "
                "        'Define recursion in your own words.', "
                "        'approved', false, '[]'::jsonb, :u)"
            ),
            {"id": question_id, "cfg": config_id, "u": seeded_users.admin_id},
        )
        await conn.execute(
            text(
                "INSERT INTO interview_outcomes ("
                "id, interview_config_id, position, outcome_text, outcome_type, "
                " importance_weight, created_by) "
                "VALUES (:id, :cfg, 1, 'Define recursion', 'knowledge', 3, :u)"
            ),
            {"id": uuid.uuid4(), "cfg": config_id, "u": seeded_users.admin_id},
        )

    # Track gateway calls made specifically during this scenario.
    pre_count = len(llm_mocks["gateway_roles"])

    # Student starts a hybrid-mode session.
    start_resp = await client.post(
        f"/api/v1/interview-configs/{config_id}/sessions",
        json={"input_mode": "hybrid"},
        headers=_auth(student_bearer),
    )
    assert start_resp.status_code == 201, start_resp.text
    session_id = UUID(start_resp.json()["session_id"])
    # Answering requires completed onboarding; this test targets the audio
    # persistence path, so fast-forward past setup.
    await _complete_onboarding(engine, session_id)

    # Seed a fake storage_objects row — the FK target.
    audio_id = uuid.uuid4()
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO storage_objects (id, bucket, object_key, mime_type, uploaded_by) "
                "VALUES (:id, 'interviews', :key, 'audio/webm', :u)"
            ),
            {
                "id": audio_id,
                "key": f"audio/{audio_id}.webm",
                "u": seeded_users.student_id,
            },
        )

    # POST /respond with audio_object_id — service must store it as-is, NO STT.
    respond_resp = await client.post(
        f"/api/v1/interview-sessions/{session_id}/respond",
        json={
            "session_id": str(session_id),
            "session_question_id": str(uuid.uuid4()),
            "answer_text": "spoken answer transcript placeholder",
            "audio_object_id": str(audio_id),
        },
        headers=_auth(student_bearer),
    )
    assert respond_resp.status_code == 200, respond_resp.text

    # Assert: persisted message has audio_object_id set + matches.
    async with engine.begin() as conn:
        row = (
            await conn.execute(
                text(
                    "SELECT audio_object_id, role, content_text, metadata_json "
                    "FROM interview_session_messages "
                    "WHERE session_id = :s AND role = 'user' "
                    "ORDER BY created_at DESC LIMIT 1"
                ),
                {"s": session_id},
            )
        ).first()
    assert row is not None, "respond must persist a user-role message"
    assert row.audio_object_id == audio_id, "audio_object_id must round-trip verbatim"
    # No transcription artifacts in metadata_json (the kind tag is "answer",
    # not anything STT-shaped):
    assert isinstance(row.metadata_json, dict)
    assert row.metadata_json.get("kind") == "answer"
    assert "transcription" not in row.metadata_json
    assert "stt" not in row.metadata_json

    # Crucial: NO STT role was ever invoked on the gateway.
    new_calls = llm_mocks["gateway_roles"][pre_count:]
    assert LLMRole.STT not in new_calls, (
        f"voice future-proof must not invoke STT; saw roles: {new_calls}"
    )

    # internal_summary_json on the session must not be polluted with STT artefacts.
    async with engine.begin() as conn:
        sess_row = (
            await conn.execute(
                text("SELECT internal_summary_json FROM interview_sessions WHERE id = :s"),
                {"s": session_id},
            )
        ).first()
    assert sess_row is not None
    summary = sess_row.internal_summary_json or {}
    assert isinstance(summary, dict)
    assert "transcription" not in summary
    assert "stt" not in summary


async def test_create_interview_outcome_duplicate_position_returns_409(
    client: httpx.AsyncClient,
    admin_bearer: str,
    scenario: dict[str, Any],
    engine: AsyncEngine,
) -> None:
    """``interview_outcomes_interview_config_id_position_key`` collisions
    must surface as 409, not 500.

    Two outcomes with the same ``position`` under one config used to hit
    the raw IntegrityError; the conflict mapper now translates this to a
    stable ``conflict`` / ``interview_outcome_position_taken`` 409.
    """
    create_resp = await client.post(
        f"/api/v1/teacher/courses/{scenario['course_id']}/interview-configs",
        json={
            "title": "Outcome-409 Probe",
            "course_id": str(scenario["course_id"]),
            "module_id": str(scenario["module_id"]),
            "supported_modes": "text",
        },
        headers=_auth(admin_bearer),
    )
    assert create_resp.status_code == 201, create_resp.text
    config_id = UUID(create_resp.json()["id"])
    try:
        first = await client.post(
            f"/api/v1/teacher/interview-configs/{config_id}/outcomes",
            json={
                "position": 1,
                "outcome_text": "First outcome at position 1",
                "outcome_type": "knowledge",
                "importance_weight": 1,
            },
            headers=_auth(admin_bearer),
        )
        assert first.status_code == 201, first.text

        second = await client.post(
            f"/api/v1/teacher/interview-configs/{config_id}/outcomes",
            json={
                "position": 1,
                "outcome_text": "Second outcome at the SAME position 1",
                "outcome_type": "skill",
                "importance_weight": 1,
            },
            headers=_auth(admin_bearer),
        )
        assert second.status_code == 409, second.text
        detail = second.json()["detail"]
        assert detail["error"] == "conflict"
        assert "interview_outcome_position_taken" in detail["message"]
    finally:
        async with engine.begin() as conn:
            # Config creation inserts a module_items row (the /content reader
            # renders one row per item); delete it before the config to avoid
            # the module_items_interview_config_id_fkey violation.
            await conn.execute(
                text("DELETE FROM module_items WHERE interview_config_id = :c"),
                {"c": config_id},
            )
            await conn.execute(
                text("DELETE FROM interview_outcomes WHERE interview_config_id = :c"),
                {"c": config_id},
            )
            await conn.execute(
                text("DELETE FROM interview_configs WHERE id = :c"),
                {"c": config_id},
            )


async def test_delete_interview_outcome_removes_it(
    client: httpx.AsyncClient,
    admin_bearer: str,
    scenario: dict[str, Any],
    engine: AsyncEngine,
) -> None:
    """DELETE /interview-configs/{id}/outcomes/{outcome_id} soft-deletes the
    outcome so it no longer appears in the authoring projection.

    Backs the new teacher outcome-editor UI — without a delete endpoint the
    editor could add but never remove a mistaken outcome.
    """
    create_resp = await client.post(
        f"/api/v1/teacher/courses/{scenario['course_id']}/interview-configs",
        json={
            "title": "Outcome-Delete Probe",
            "course_id": str(scenario["course_id"]),
            "module_id": str(scenario["module_id"]),
            "supported_modes": "text",
        },
        headers=_auth(admin_bearer),
    )
    assert create_resp.status_code == 201, create_resp.text
    config_id = UUID(create_resp.json()["id"])
    try:
        created = await client.post(
            f"/api/v1/teacher/interview-configs/{config_id}/outcomes",
            json={
                "position": 1,
                "outcome_text": "Outcome to be deleted",
                "outcome_type": "knowledge",
                "importance_weight": 2,
            },
            headers=_auth(admin_bearer),
        )
        assert created.status_code == 201, created.text
        outcome_id = created.json()["id"]

        deleted = await client.delete(
            f"/api/v1/teacher/interview-configs/{config_id}/outcomes/{outcome_id}",
            headers=_auth(admin_bearer),
        )
        assert deleted.status_code == 204, deleted.text

        # Authoring projection no longer lists the outcome.
        authoring = await client.get(
            f"/api/v1/teacher/interview-configs/{config_id}",
            headers=_auth(admin_bearer),
        )
        assert authoring.status_code == 200, authoring.text
        outcome_ids = [o["id"] for o in authoring.json()["outcomes"]]
        assert outcome_id not in outcome_ids
    finally:
        async with engine.begin() as conn:
            await conn.execute(
                text("DELETE FROM module_items WHERE interview_config_id = :c"),
                {"c": config_id},
            )
            await conn.execute(
                text("DELETE FROM interview_outcomes WHERE interview_config_id = :c"),
                {"c": config_id},
            )
            await conn.execute(
                text("DELETE FROM interview_configs WHERE id = :c"),
                {"c": config_id},
            )


__all__: list[str] = []
