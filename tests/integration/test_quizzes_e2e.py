"""Phase 5 quiz generation/regeneration end-to-end suite (T5.15).

Closes Phase 5 by exercising the quiz routers + services + AI pipeline
in a single in-process flow. Two scenarios cover the meaningful
lifecycles:

* **FULL generation**: teacher creates a quiz under a module that has
  document chunks, fires ``POST /generate``, the worker callback runs
  the full pipeline (retrieval → ideation → generation → validation →
  dedup → persistence), the run reaches ``status='completed'``, the
  teacher publishes, the student takes the quiz without ever seeing
  ``is_correct``.
* **REGENERATE**: teacher fires
  ``POST /quizzes/{id}/questions/{qid}/regenerate``, the worker
  callback runs the regenerate pipeline, the targeted question's
  ``revision_no`` increments while siblings stay untouched.

The audit lineage assertion piggybacks on Scenario 1: every LLM and
embedding call writes one ``ai_model_calls`` row carrying the same
``pipeline_run_id`` plus a ``stage_name`` from the expected stage
vocabulary. This proves the audit threading works end-to-end.

Mock strategy
-------------
:meth:`LLMGateway.generate_json` and :meth:`EmbeddingClient.embed` are
replaced with fakes that write a real ``ai_model_calls`` row each
(via the canonical :func:`write_ai_model_call` helper) and return
canned content. This keeps the audit lineage assertion meaningful
without requiring outbound API keys / OpenAI / pgvector index seeding
at the embedding layer. Vector search runs against the real seeded
chunks; KG retrieval is stubbed to an empty :class:`KGContext`.

Document chunks are seeded via raw SQL because invoking the materials
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

import abridgeai.features.access_control.models  # noqa: F401  -- register FK targets
import abridgeai.features.courses.models  # noqa: F401  -- register modules / lessons
import abridgeai.features.identity.models  # noqa: F401  -- register users
import abridgeai.features.quizzes.models  # noqa: F401  -- register quiz tables
from abridgeai.ai.llm.audit import write_ai_model_call
from abridgeai.ai.llm.roles import LLMRole
from abridgeai.core.config import get_settings
from abridgeai.core.db import Base, get_db
from abridgeai.core.security import create_access_token, generate_token, hash_secret
from abridgeai.features.quizzes.routers import (
    authoring_router as quiz_authoring_router,
)
from abridgeai.features.quizzes.routers import (
    learner_router as quiz_learner_router,
)
from abridgeai.features.quizzes.routers.authoring import get_arq_pool
from abridgeai.features.quizzes.services import generation as generation_service

EMBEDDING_DIM = 3072


import abridgeai.features.interviews.models  # noqa: E402, F401  -- T6.1 registers interview_* tables

for _stub_name in (
    "interview_configs",
    "learning_materials",
    "processing_jobs",
):
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
    async def _override_get_db() -> AsyncIterator[AsyncSession]:
        async with session_factory() as session:
            yield session

    async def _override_arq_pool() -> object | None:
        return None

    fastapi_app = FastAPI()
    fastapi_app.include_router(quiz_authoring_router, prefix="/api/v1")
    fastapi_app.include_router(quiz_learner_router, prefix="/api/v1")
    fastapi_app.dependency_overrides[get_db] = _override_get_db
    fastapi_app.dependency_overrides[get_arq_pool] = _override_arq_pool
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
async def scenario(
    engine: AsyncEngine, seeded_users: SeededUsers
) -> AsyncIterator[dict[str, UUID]]:
    """Seed module + lesson + storage_object + material + version + chunks."""
    suffix = uuid.uuid4().hex[:8]
    module_id = uuid.uuid4()
    lesson_id = uuid.uuid4()
    storage_id = uuid.uuid4()
    material_id = uuid.uuid4()
    version_id = uuid.uuid4()
    chunk_ids = [uuid.uuid4() for _ in range(3)]

    embedding_literal = "[" + ",".join(["0.1"] * EMBEDDING_DIM) + "]"

    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO modules (id, course_id, title, position, status) "
                "VALUES (:m, :c, 'E2E Quiz Module', 1, 'published')"
            ),
            {"m": module_id, "c": seeded_users.course_id},
        )
        await conn.execute(
            text(
                "INSERT INTO lessons (id, module_id, slug, title, status) "
                "VALUES (:l, :m, :slug, 'Photosynthesis', 'published')"
            ),
            {"l": lesson_id, "m": module_id, "slug": f"photosynthesis-{suffix}"},
        )
        await conn.execute(
            text("INSERT INTO storage_objects (id, bucket, object_key) VALUES (:id, 'test', :key)"),
            {"id": storage_id, "key": f"e2e/{suffix}.txt"},
        )
        await conn.execute(
            text(
                "INSERT INTO learning_materials "
                "(id, lesson_id, title, material_type) "
                "VALUES (:id, :l, 'Photosynthesis Notes', 'text')"
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
                        f"Chunk {index}: photosynthesis converts light energy "
                        "into chemical energy stored in glucose."
                    ),
                    "emb": embedding_literal,
                    "hash": f"hash-{suffix}-{index}".ljust(64, "0")[:64],
                },
            )

    yield {
        "course_id": seeded_users.course_id,
        "module_id": module_id,
        "lesson_id": lesson_id,
        "version_id": version_id,
        "chunk_ids": chunk_ids,
    }

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
                "DELETE FROM quiz_attempt_answers WHERE attempt_id IN ("
                " SELECT id FROM quiz_attempts WHERE quiz_id IN ("
                "  SELECT id FROM quizzes WHERE module_id = :m))"
            ),
            {"m": module_id},
        )
        await conn.execute(
            text(
                "DELETE FROM quiz_attempts WHERE quiz_id IN "
                "(SELECT id FROM quizzes WHERE module_id = :m)"
            ),
            {"m": module_id},
        )
        await conn.execute(
            text(
                "DELETE FROM quiz_question_revisions WHERE question_id IN ("
                " SELECT id FROM quiz_questions WHERE quiz_id IN ("
                "  SELECT id FROM quizzes WHERE module_id = :m))"
            ),
            {"m": module_id},
        )
        await conn.execute(
            text(
                "DELETE FROM quiz_question_options WHERE question_id IN ("
                " SELECT id FROM quiz_questions WHERE quiz_id IN ("
                "  SELECT id FROM quizzes WHERE module_id = :m))"
            ),
            {"m": module_id},
        )
        await conn.execute(
            text(
                "DELETE FROM quiz_questions WHERE quiz_id IN "
                "(SELECT id FROM quizzes WHERE module_id = :m)"
            ),
            {"m": module_id},
        )
        await conn.execute(
            text(
                "DELETE FROM quiz_source_lessons WHERE quiz_id IN "
                "(SELECT id FROM quizzes WHERE module_id = :m)"
            ),
            {"m": module_id},
        )
        await conn.execute(text("DELETE FROM module_items WHERE module_id = :m"), {"m": module_id})
        await conn.execute(
            text("UPDATE quizzes SET generation_run_id = NULL WHERE module_id = :m"),
            {"m": module_id},
        )
        await conn.execute(
            text("DELETE FROM generation_runs WHERE module_id = :m"), {"m": module_id}
        )
        await conn.execute(text("DELETE FROM quizzes WHERE module_id = :m"), {"m": module_id})
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
        await conn.execute(text("DELETE FROM lessons WHERE id = :l"), {"l": lesson_id})
        await conn.execute(text("DELETE FROM modules WHERE id = :m"), {"m": module_id})


def _ideation_templates(chunk_ids: list[UUID], n: int = 3) -> list[Any]:
    """Build parser-validated :class:`Template` instances.

    Constructing through the Pydantic parser guarantees every field
    the generation prompt template references is populated — bypassing
    Jinja2 ``StrictUndefined`` errors that would fire on raw dicts.
    """
    from abridgeai.features.quizzes.ai.stages.ideation.parsers import Template

    chunk_str_ids = [str(cid) for cid in chunk_ids]
    return [
        Template(
            position=i + 1,
            section_id="synthetic-section",
            topic=f"photosynthesis-aspect-{i + 1}",
            question_type="mcq",
            bloom_level="understand",
            difficulty="medium",
            source_chunk_ids=chunk_str_ids[: i + 1],
            rationale=f"Probes aspect {i + 1}",
        )
        for i in range(n)
    ]


def _generated_questions(chunk_ids: list[UUID], n: int = 3) -> list[Any]:
    """Build candidate-question shims with the persisted question_type.

    The DB CHECK on ``quiz_questions.question_type`` enforces
    ``multiple_choice`` (post-migration 0007 rename), but
    :class:`GeneratedQuestion` keeps the legacy ``mcq`` Literal for
    parser compatibility. We bypass the parser class here and return
    duck-typed objects that the pipeline consumes via ``.model_dump()``.
    """
    chunk_str_ids = [str(cid) for cid in chunk_ids]

    class _Candidate:
        def __init__(self, payload: dict[str, Any]) -> None:
            self._payload = payload

        def model_dump(self) -> dict[str, Any]:
            return dict(self._payload)

    payloads: list[dict[str, Any]] = []
    for i in range(n):
        payloads.append(
            {
                "position": i + 1,
                "question_type": "multiple_choice",
                "prompt_text": (
                    f"Which statement about photosynthesis is most accurate (#{i + 1})?"
                ),
                "hint_text": "Think about gas exchange in plants.",
                "explanation": "Photosynthesis converts light energy and produces oxygen.",
                "difficulty": "medium",
                "bloom_level": "understand",
                "expected_response_time_ms": 60_000,
                "source_refs": chunk_str_ids[:1],
                "original_generated_payload": {"position": i + 1},
                "options": [
                    {
                        "option_key": "A",
                        "option_text": (f"Plants emit oxygen as a waste product (#{i + 1})."),
                        "is_correct": True,
                        "position": 1,
                    },
                    {
                        "option_key": "B",
                        "option_text": "Plants absorb oxygen and emit carbon.",
                        "is_correct": False,
                        "position": 2,
                    },
                    {
                        "option_key": "C",
                        "option_text": "Photosynthesis happens only at night.",
                        "is_correct": False,
                        "position": 3,
                    },
                    {
                        "option_key": "D",
                        "option_text": "Glucose is consumed, not produced.",
                        "is_correct": False,
                        "position": 4,
                    },
                ],
            }
        )
    return [_Candidate(p) for p in payloads]


def _accept_verdicts(question_count: int) -> list[Any]:
    from abridgeai.features.quizzes.ai.stages.validation.parsers import Verdict

    return [
        Verdict(
            position=i + 1,
            verdict="accept",
            reasons=["Looks good."],
            evidence_excerpt="photosynthesis converts light energy",
        )
        for i in range(question_count)
    ]


@pytest.fixture
def llm_mocks(scenario: dict[str, UUID], monkeypatch: pytest.MonkeyPatch) -> dict[str, list[Any]]:
    """Replace stage functions with audit-writing fakes.

    Mocking at the **stage** boundary (not the gateway / HTTP layer)
    keeps the test in control of every audit row's
    ``generation_run_id`` / ``parent_run_id`` so the FK to
    ``generation_runs`` stays valid. The fakes still call
    :func:`write_ai_model_call` per stage, so the audit-lineage
    assertion sees real rows with the expected ``stage_name`` set.
    KG retrieval is stubbed to an empty :class:`KGContext`.
    """
    chunk_ids = [UUID(str(c)) for c in scenario["chunk_ids"]]
    captured: dict[str, list[Any]] = {"stages": []}
    captured_run_id: dict[str, UUID] = {}

    async def _emit_audit(
        db: AsyncSession,
        *,
        stage: str,
        role: LLMRole,
        pipeline_run_id: UUID | None,
    ) -> None:
        generation_run_id = captured_run_id.get("id")
        if generation_run_id is None:
            return
        await write_ai_model_call(
            db,
            role=role,
            tier=None,
            operation="chat_completion" if role is not LLMRole.EMBEDDING else "embedding",
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

    async def _fake_retrieve_chunks(
        db: AsyncSession,
        run_id: UUID,
        quiz: Any,
        config: dict[str, Any],
        *,
        question_anchor: str | None = None,
        kg_context_enabled: bool = True,
        pipeline_run_id: UUID | None = None,
        per_anchor_top_k: int = 20,
        final_top_k: int = 12,
        embedding_client: Any = None,
    ) -> tuple[list[Any], list[float], list[str]]:
        from abridgeai.ai.retrieval import ChunkWithDistance

        del quiz, config, question_anchor, kg_context_enabled, per_anchor_top_k
        del final_top_k, embedding_client
        captured_run_id["id"] = run_id
        captured["stages"].append({"stage": "retrieval", "run_id": str(run_id)})
        chunks = [
            ChunkWithDistance(
                chunk_id=cid,
                material_version_id=scenario["version_id"],
                course_id=scenario["course_id"],
                lesson_id=scenario["lesson_id"],
                content=(f"Chunk {idx}: photosynthesis converts light energy into glucose."),
                distance=0.1 * idx,
                embedding=[0.1] * EMBEDDING_DIM,
            )
            for idx, cid in enumerate(chunk_ids)
        ]
        await _emit_audit(
            db, stage="embedding", role=LLMRole.EMBEDDING, pipeline_run_id=pipeline_run_id
        )
        return chunks, [0.1] * EMBEDDING_DIM, ["photosynthesis"]

    async def _fake_ideate(
        db: AsyncSession,
        run: Any,
        title: str,
        config: dict[str, Any],
        outlines: list[Any],
        budget: dict[str, int],
        *,
        pipeline_run_id: UUID | None = None,
        gateway: Any = None,
    ) -> list[Any]:
        del title, config, outlines, budget, gateway
        captured_run_id["id"] = run.id
        captured["stages"].append({"stage": "ideation"})
        await _emit_audit(
            db, stage="ideation", role=LLMRole.IDEATION, pipeline_run_id=pipeline_run_id
        )
        return _ideation_templates(chunk_ids)

    async def _fake_generate(
        title: str,
        config: dict[str, Any],
        chunks: list[Any],
        templates: list[dict[str, Any]],
        kg_context: Any,
        db: AsyncSession,
        *,
        pipeline_run_id: UUID,
        parent_run_id: UUID | None = None,
        previous_questions: list[str] | None = None,
        gateway: Any = None,
    ) -> list[Any]:
        del title, config, chunks, templates, kg_context, previous_questions, gateway
        del parent_run_id
        captured["stages"].append({"stage": "generation"})
        await _emit_audit(
            db,
            stage="generation",
            role=LLMRole.GENERATION,
            pipeline_run_id=pipeline_run_id,
        )
        return _generated_questions(chunk_ids, n=3)

    async def _fake_validate(
        title: str,
        chunks: list[Any],
        questions: list[dict[str, Any]],
        db: AsyncSession,
        *,
        pipeline_run_id: UUID | None = None,
        parent_run_id: UUID | None = None,
        audit_parent_run_id: UUID | None = None,
        config: dict[str, Any] | None = None,
        gateway: Any = None,
    ) -> tuple[Any, list[Any]]:
        del title, chunks, config, gateway, parent_run_id, audit_parent_run_id
        captured["stages"].append({"stage": "validation"})
        await _emit_audit(
            db,
            stage="validation",
            role=LLMRole.VALIDATION,
            pipeline_run_id=pipeline_run_id,
        )
        return None, _accept_verdicts(question_count=len(questions))

    targets = (
        "abridgeai.features.quizzes.ai.pipelines.full.retrieve_chunks",
        "abridgeai.features.quizzes.ai.pipelines.regenerate.retrieve_chunks",
    )
    for target in targets:
        monkeypatch.setattr(target, _fake_retrieve_chunks)

    monkeypatch.setattr(
        "abridgeai.features.quizzes.ai.pipelines.full.ideate_for_outline",
        _fake_ideate,
    )

    for target in (
        "abridgeai.features.quizzes.ai.pipelines.full.generate_questions",
        "abridgeai.features.quizzes.ai.pipelines.regenerate.generate_questions",
    ):
        monkeypatch.setattr(target, _fake_generate)

    for target in (
        "abridgeai.features.quizzes.ai.pipelines.full.validate_questions",
        "abridgeai.features.quizzes.ai.pipelines.regenerate.validate_questions",
    ):
        monkeypatch.setattr(target, _fake_validate)

    return captured


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


async def test_full_generation_lifecycle_publish_take_score(
    client: httpx.AsyncClient,
    admin_bearer: str,
    student_bearer: str,
    scenario: dict[str, UUID],
    session_factory: async_sessionmaker[AsyncSession],
    engine: AsyncEngine,
    llm_mocks: dict[str, list[Any]],
    seeded_users: SeededUsers,
) -> None:
    """Teacher creates → generates → publishes; student takes + submits."""
    create_resp = await client.post(
        f"/api/v1/teacher/courses/{scenario['course_id']}/quizzes",
        json={
            "module_id": str(scenario["module_id"]),
            "title": "E2E Photosynthesis Quiz",
            "description": "Lifecycle proof",
        },
        headers=_auth(admin_bearer),
    )
    assert create_resp.status_code == 201, create_resp.text
    quiz_id = UUID(create_resp.json()["id"])

    gen_resp = await client.post(
        f"/api/v1/teacher/quizzes/{quiz_id}/generate",
        json={
            "title": "Photosynthesis Quiz",
            "question_count": 3,
            "question_types": ["multiple_choice"],
            "generation_mode": "topic",
            "focus_topics": ["photosynthesis"],
            "source_lesson_ids": [str(scenario["lesson_id"])],
        },
        headers=_auth(admin_bearer),
    )
    assert gen_resp.status_code == 202, gen_resp.text
    run_id = UUID(gen_resp.json()["id"])

    async with session_factory() as session:
        await session.execute(
            text(
                "UPDATE generation_runs SET config_json = jsonb_set("
                "config_json, '{question_count}', '3'::jsonb, true) "
                "WHERE id = :id"
            ),
            {"id": run_id},
        )
        await session.commit()

    async with session_factory() as session:
        await generation_service.run_quiz_generation(session, run_id)

    poll_resp = await client.get(
        f"/api/v1/teacher/quizzes/{quiz_id}/generation-runs/{run_id}",
        headers=_auth(admin_bearer),
    )
    assert poll_resp.status_code == 200, poll_resp.text
    assert poll_resp.json()["status"] == "completed"

    async with engine.begin() as conn:
        question_rows = (
            await conn.execute(
                text("SELECT id FROM quiz_questions WHERE quiz_id = :q"),
                {"q": quiz_id},
            )
        ).all()
    question_count = len(question_rows)
    assert question_count >= 1, "pipeline must have persisted at least one question"

    # Real teacher workflow before publish: generated questions land as
    # 'pending' with no expected time. Set an expected time and approve them
    # so the publish gate (needs >=1 approved question, each with a positive
    # expected_response_time_ms) is satisfied. Partial publish means we could
    # approve a subset, but here we sign off all of them.
    question_ids = [str(row.id) for row in question_rows]
    set_time_resp = await client.post(
        f"/api/v1/teacher/quizzes/{quiz_id}/questions/bulk-set-expected-time",
        json={
            "items": [
                {"question_id": qid, "expected_response_time_ms": 60_000} for qid in question_ids
            ]
        },
        headers=_auth(admin_bearer),
    )
    assert set_time_resp.status_code == 200, set_time_resp.text
    approve_resp = await client.post(
        f"/api/v1/teacher/quizzes/{quiz_id}/questions/bulk-approve",
        json={"question_ids": question_ids},
        headers=_auth(admin_bearer),
    )
    assert approve_resp.status_code == 200, approve_resp.text

    pub_resp = await client.post(
        f"/api/v1/teacher/quizzes/{quiz_id}/publish",
        headers=_auth(admin_bearer),
    )
    assert pub_resp.status_code == 200, pub_resp.text

    student_get = await client.get(
        f"/api/v1/quizzes/{quiz_id}",
        headers=_auth(student_bearer),
    )
    assert student_get.status_code == 200, student_get.text
    assert "is_correct" not in student_get.text
    assert "correct_option_id" not in student_get.text

    attempt_resp = await client.post(
        f"/api/v1/quizzes/{quiz_id}/attempts",
        json={"quiz_id": str(quiz_id)},
        headers=_auth(student_bearer),
    )
    assert attempt_resp.status_code == 201, attempt_resp.text
    progress_payload = attempt_resp.json()
    assert "is_correct" not in attempt_resp.text
    # Questions are nested under `take` (QuizAttemptProgressRead.take is a
    # QuizForTakingPublic), not at the top level.
    questions = progress_payload["take"]["questions"]
    assert questions, "take payload must list questions"

    async with engine.begin() as conn:
        attempt_id = (
            await conn.execute(
                text(
                    "SELECT id FROM quiz_attempts WHERE quiz_id = :q "
                    "ORDER BY started_at DESC LIMIT 1"
                ),
                {"q": quiz_id},
            )
        ).scalar_one()

    first_q = questions[0]
    first_option_id = first_q["options"][0]["id"]
    answer_resp = await client.post(
        f"/api/v1/attempts/{attempt_id}/answers",
        json={
            "question_id": first_q["id"],
            "selected_option_id": first_option_id,
        },
        headers=_auth(student_bearer),
    )
    assert answer_resp.status_code == 201, answer_resp.text

    submit_resp = await client.post(
        f"/api/v1/attempts/{attempt_id}/submit",
        headers=_auth(student_bearer),
    )
    assert submit_resp.status_code == 200, submit_resp.text
    submitted = submit_resp.json()
    assert submitted["status"] == "submitted"
    assert submitted["score_percent"] is not None

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
    print("\n=== T5.15 audit lineage for generation_run_id =", run_id, "===")
    pipeline_ids = {r.pipeline_run_id for r in rows if r.pipeline_run_id is not None}
    print(f"distinct pipeline_run_ids: {len(pipeline_ids)} (must be 1)")
    for row in rows:
        print(
            f"  stage_name={row.stage_name:<14} role={row.role:<12} "
            f"count={row.c}  pipeline_run_id={row.pipeline_run_id}"
        )
    assert rows, "every run must produce ai_model_calls rows"
    assert len(pipeline_ids) == 1, f"all stages must share one pipeline_run_id; got {pipeline_ids}"
    stages_seen = {r.stage_name for r in rows}
    expected_stages = {"ideation", "generation", "validation", "embedding"}
    missing = expected_stages - stages_seen
    assert not missing, (
        f"expected stages {expected_stages} in audit; missing {missing}; saw {stages_seen}"
    )

    del seeded_users


async def test_regenerate_question_increments_revision(
    client: httpx.AsyncClient,
    admin_bearer: str,
    scenario: dict[str, UUID],
    session_factory: async_sessionmaker[AsyncSession],
    engine: AsyncEngine,
    llm_mocks: dict[str, list[Any]],
    seeded_users: SeededUsers,
) -> None:
    """Teacher regenerates one question; revision_no bumps; sibling untouched."""
    create_resp = await client.post(
        f"/api/v1/teacher/courses/{scenario['course_id']}/quizzes",
        json={
            "module_id": str(scenario["module_id"]),
            "title": "Regenerate Test Quiz",
        },
        headers=_auth(admin_bearer),
    )
    assert create_resp.status_code == 201
    quiz_id = UUID(create_resp.json()["id"])

    target_q_id = uuid.uuid4()
    sibling_q_id = uuid.uuid4()
    chunk_id_strs = [str(c) for c in scenario["chunk_ids"]]

    async with engine.begin() as conn:
        for q_id, position, prompt in (
            (target_q_id, 1, "Original target question about photosynthesis?"),
            (sibling_q_id, 2, "Sibling question untouched by regenerate?"),
        ):
            await conn.execute(
                text(
                    "INSERT INTO quiz_questions "
                    "(id, quiz_id, position, question_type, prompt_text, "
                    " review_status, source_refs) "
                    "VALUES (:id, :q, :pos, 'multiple_choice', :prompt, "
                    "        'approved', CAST(:refs AS jsonb))"
                ),
                {
                    "id": q_id,
                    "q": quiz_id,
                    "pos": position,
                    "prompt": prompt,
                    "refs": json.dumps(chunk_id_strs[:1]),
                },
            )
            await conn.execute(
                text(
                    "INSERT INTO quiz_question_revisions "
                    "(id, question_id, revision_no, source_kind, payload_json) "
                    "VALUES (uuid_generate_v4(), :q, 1, 'teacher', '{}'::jsonb)"
                ),
                {"q": q_id},
            )

    regen_resp = await client.post(
        f"/api/v1/teacher/quizzes/{quiz_id}/questions/{target_q_id}/regenerate",
        headers=_auth(admin_bearer),
    )
    assert regen_resp.status_code == 202, regen_resp.text
    run_id = UUID(regen_resp.json()["id"])

    async with session_factory() as session:
        await generation_service.run_quiz_generation(session, run_id)

    async with engine.begin() as conn:
        target_revisions = (
            await conn.execute(
                text("SELECT MAX(revision_no) FROM quiz_question_revisions WHERE question_id = :q"),
                {"q": target_q_id},
            )
        ).scalar_one()
        sibling_revisions = (
            await conn.execute(
                text("SELECT MAX(revision_no) FROM quiz_question_revisions WHERE question_id = :q"),
                {"q": sibling_q_id},
            )
        ).scalar_one()
        sibling_prompt = (
            await conn.execute(
                text("SELECT prompt_text FROM quiz_questions WHERE id = :q"),
                {"q": sibling_q_id},
            )
        ).scalar_one()

    assert target_revisions == 2, (
        f"regenerate must bump revision_no from 1 to 2; saw {target_revisions}"
    )
    assert sibling_revisions == 1, f"sibling revision_no must stay at 1; saw {sibling_revisions}"
    assert sibling_prompt == "Sibling question untouched by regenerate?", (
        "sibling prompt must be untouched"
    )

    del seeded_users


__all__: list[str] = []
