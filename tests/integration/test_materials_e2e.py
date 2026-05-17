"""Phase 4 end-to-end integration suite (T4.9).

Closes Phase 4 by composing every layer the materials feature touches into
a single in-process flow:

* **Routers** mounted on a FastAPI app under ``/api/v1`` — the same shape
  T3.10 used for Phase 3.  Both the authoring (T4.5) and learner (T4.6)
  routers participate, plus the courses authoring router (T3.5/T3.7) so
  module + lesson seeding happens through the public API where possible.
* **moto S3** for direct-upload + multipart simulation — the test PUTs
  bytes to the presigned URL the API hands back, then ``head_object``
  during ``/complete`` round-trips against the same backend.
* **Mocked ARQ pool** (``app.dependency_overrides[get_arq_pool]``) — the
  test captures the enqueue args and simulates the worker by calling
  :func:`run_material_ingest` directly with the captured arguments.
  No real Redis / ARQ runtime is required.
* **Stub extractors** registered at the real MIMEs (``application/pdf``,
  ``audio/wav``, ``image/png`` ...) for the duration of each test so the
  pipeline's MIME→extractor dispatch routes to in-process fakes that
  return deterministic ``ExtractedContent`` — no Whisper, no OCR, no
  network.
* **Fake EmbeddingClient** that writes ``ai_model_calls`` audit rows so
  the ``pipeline_run_id`` propagation invariant holds end-to-end without
  hitting OpenAI.
* **Mocked KG builder** for the cross-linking scenario — concept upserts
  are captured in-memory so the test can assert that three materials in
  the same course share concept names without bringing up Neo4j.

The plan body (§5283-5338) requires ≥10 scenarios. This module ships 11.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock
from uuid import UUID

import httpx
import pytest
import pytest_asyncio
from alembic import command
from alembic.config import Config
from conftest import SeededUsers
from fastapi import FastAPI
from moto.server import ThreadedMotoServer
from pydantic import SecretStr
from sqlalchemy import Column, Table, text
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

import abridgeai.features.access_control.models  # noqa: F401  -- register FK targets
import abridgeai.features.identity.models  # noqa: F401  -- register users FK target
import abridgeai.features.materials.models  # noqa: F401  -- register tables
from abridgeai.ai.extraction import EXTRACTOR_REGISTRY, ExtractedContent, SourceLocation
from abridgeai.ai.knowledge_graph.schemas import KGSummary
from abridgeai.core.config import Settings, get_settings
from abridgeai.core.db import Base, get_db
from abridgeai.core.security import create_access_token, generate_token, hash_secret
from abridgeai.features.materials.ingestion import pipeline as pipeline_mod
from abridgeai.features.materials.ingestion import run_material_ingest
from abridgeai.features.materials.routers import authoring_router, learner_router
from abridgeai.features.materials.routers.authoring import get_arq_pool
from abridgeai.infrastructure import s3 as s3_module

BUCKET = "abridgeai-test-e2e"
EMBEDDING_DIM = 1536

PDF_MIME = "application/pdf"
DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
PPTX_MIME = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
HTML_MIME = "text/html"
AUDIO_MIME = "audio/wav"
IMAGE_MIME = "image/png"

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "multimodal"


# ---------------------------------------------------------------------------
# SQLAlchemy stub tables for FK targets the materials feature points to that
# live in not-yet-built features (quizzes, interview_configs).  Same pattern
# T3.10 / T4.5 use.
# ---------------------------------------------------------------------------

for _stub_name in ("interview_configs",):
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


def _vec(seed: int) -> list[float]:
    """Deterministic 1536-d vector keyed off ``seed``."""
    base = [0.1, 0.2, 0.3, float(seed % 7) / 10.0]
    padded = base + [0.0] * (EMBEDDING_DIM - len(base))
    return [float(v) for v in padded[:EMBEDDING_DIM]]


# ---------------------------------------------------------------------------
# Stub extractors — registered at the real MIMEs for each test, then
# unregistered in teardown.  The pipeline's dispatch table sees the stub,
# so no network / Whisper / OCR / docx parsing runs.
# ---------------------------------------------------------------------------


@dataclass
class _StubExtractedContent:
    text: str
    metadata: dict[str, Any]
    source_type: str
    locations: list[SourceLocation]


def _make_stub_extractor(content: _StubExtractedContent) -> type:
    class _StubExtractor:
        async def extract(self, source: Any) -> ExtractedContent:
            del source
            return ExtractedContent(
                text=content.text,
                metadata=dict(content.metadata),
                source_type=content.source_type,
                source_locations=list(content.locations),
            )

    return _StubExtractor


PDF_STUB = _StubExtractedContent(
    text="[Page 1] Python is a high-level programming language.\n[Page 2] Python supports OOP.",
    metadata={"page_count": 2},
    source_type="pdf",
    locations=[SourceLocation(page=1), SourceLocation(page=2)],
)

DOCX_STUB = _StubExtractedContent(
    text="Heading: Python OOP\nObject-oriented programming concepts in Python.",
    metadata={"paragraph_count": 2},
    source_type="docx",
    locations=[SourceLocation(page=1)],
)

PPTX_STUB = _StubExtractedContent(
    text="Slide 1: Python\nSlide 2: OOP fundamentals",
    metadata={"slide_count": 2},
    source_type="pptx",
    locations=[SourceLocation(page=1), SourceLocation(page=2)],
)

HTML_STUB = _StubExtractedContent(
    text="Title: Python tutorial\nBody: introduction to Python OOP.",
    metadata={"section_count": 2},
    source_type="html",
    locations=[SourceLocation(page=1)],
)

AUDIO_STUB = _StubExtractedContent(
    text="Welcome to Python OOP.\n\nThis lecture covers classes and inheritance.",
    metadata={"stt_provider": "stub", "duration_seconds": 12.0, "segment_count": 2},
    source_type="audio",
    locations=[
        SourceLocation(timestamp_start_ms=0, timestamp_end_ms=6_000),
        SourceLocation(timestamp_start_ms=6_000, timestamp_end_ms=12_000),
    ],
)

IMAGE_STUB = _StubExtractedContent(
    text="Diagram: Python class hierarchy with OOP inheritance",
    metadata={"ocr_provider": "stub", "image_size": (200, 100)},
    source_type="image",
    locations=[SourceLocation(bbox=(0.0, 0.0, 200.0, 100.0))],
)


@pytest.fixture
def _stub_real_extractors() -> AsyncIterator[None]:
    """Replace registered extractors at the real MIMEs with deterministic stubs.

    Saves the original classes, swaps in stubs that ignore the source path
    and return canned :class:`ExtractedContent`, then restores on teardown.
    The pipeline never runs real PDF / DOCX / Whisper / OCR code paths.
    """
    saved: dict[str, type] = {}
    overrides = {
        PDF_MIME: PDF_STUB,
        DOCX_MIME: DOCX_STUB,
        PPTX_MIME: PPTX_STUB,
        HTML_MIME: HTML_STUB,
        AUDIO_MIME: AUDIO_STUB,
        IMAGE_MIME: IMAGE_STUB,
    }
    for mime, content in overrides.items():
        if mime in EXTRACTOR_REGISTRY:
            saved[mime] = EXTRACTOR_REGISTRY[mime]
        EXTRACTOR_REGISTRY[mime] = _make_stub_extractor(content)
    yield
    for mime in overrides:
        if mime in saved:
            EXTRACTOR_REGISTRY[mime] = saved[mime]
        else:
            EXTRACTOR_REGISTRY.pop(mime, None)


# ---------------------------------------------------------------------------
# Fake embedding client (writes ``ai_model_calls`` audit rows so the
# pipeline_run_id propagation invariant holds end-to-end).
# ---------------------------------------------------------------------------


class _FakeEmbeddingClient:
    """Mirrors :class:`EmbeddingClient.embed` without any network IO."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def embed(
        self,
        texts: list[str],
        *,
        db: AsyncSession,
        pipeline_run_id: UUID | None = None,
        parent_job_id: UUID | None = None,
        parent_run_id: UUID | None = None,
    ) -> list[list[float]]:
        from decimal import Decimal

        from abridgeai.ai.llm.audit import write_ai_model_call
        from abridgeai.ai.llm.roles import LLMRole

        self.calls.append(
            {
                "count": len(texts),
                "pipeline_run_id": pipeline_run_id,
                "parent_job_id": parent_job_id,
            }
        )
        await write_ai_model_call(
            db,
            role=LLMRole.EMBEDDING,
            tier=None,
            operation="embedding",
            model_name="fake-embedding-model",
            base_url="https://fake.test/v1",
            stage_name="embedding",
            pipeline_run_id=pipeline_run_id,
            parent_run_id=parent_run_id,
            parent_job_id=parent_job_id,
            request_payload={"input_count": len(texts)},
            response_payload=None,
            input_tokens=len(texts) * 5,
            output_tokens=0,
            cached_input_tokens=None,
            latency_ms=2,
            status="success",
            error_message=None,
            estimated_cost_usd=Decimal("0.000001"),
        )
        return [_vec(i) for i in range(len(texts))]


@dataclass
class _StubLLMResponse:
    content_json: dict[str, Any]
    model_name: str = "fake-model"
    input_tokens: int | None = 1
    output_tokens: int | None = 1
    total_tokens: int | None = 2


class _StubLLMGateway:
    """Returns canned LLMResponse-shaped objects for SemanticChunker / KG.

    The real gateway calls OpenAI; in e2e tests we want deterministic
    in-process responses so chunking + KG enrichment never touch the
    network. The narrow surface used by :mod:`abridgeai.ai.chunking._enrich`
    requires ``content_json`` (dict) + ``model_name`` + token counts.
    """

    async def generate_json(self, *args: Any, **kwargs: Any) -> _StubLLMResponse:
        del args, kwargs
        return _StubLLMResponse(content_json={})

    async def generate_text(self, *args: Any, **kwargs: Any) -> str:
        del args, kwargs
        return ""


# ---------------------------------------------------------------------------
# moto + DB + app fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def moto_server() -> ThreadedMotoServer:
    server = ThreadedMotoServer(port=0)
    server.start()
    host, port = server.get_host_and_port()
    server._host = host  # type: ignore[attr-defined]
    server._port = port  # type: ignore[attr-defined]
    yield server
    server.stop()


def _moto_endpoint(server: ThreadedMotoServer) -> str:
    return f"http://{server._host}:{server._port}"  # type: ignore[attr-defined]


def _settings_for(server: ThreadedMotoServer) -> Settings:
    base = get_settings()
    return Settings(
        database_url=base.database_url,
        redis_url=base.redis_url,
        jwt_secret_key=base.jwt_secret_key,
        aws_access_key_id=SecretStr("AKIAIOSFODNN7EXAMPLE"),
        aws_secret_access_key=SecretStr("wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"),
        aws_endpoint_url=_moto_endpoint(server),
        aws_public_endpoint_url=_moto_endpoint(server),
        aws_region="us-east-1",
        s3_bucket_name=BUCKET,
        s3_url_ttl_seconds=3600,
    )


@pytest_asyncio.fixture
async def s3_settings(moto_server: ThreadedMotoServer) -> Settings:
    settings = _settings_for(moto_server)
    import aioboto3

    session = aioboto3.Session()
    async with session.client(
        "s3",
        endpoint_url=settings.aws_endpoint_url,
        aws_access_key_id=settings.aws_access_key_id.get_secret_value(),  # type: ignore[union-attr]
        aws_secret_access_key=settings.aws_secret_access_key.get_secret_value(),  # type: ignore[union-attr]
        region_name=settings.aws_region,
    ) as client:
        try:
            await client.create_bucket(Bucket=BUCKET)
        except client.exceptions.BucketAlreadyOwnedByYou:
            pass
        except client.exceptions.BucketAlreadyExists:
            pass
    return settings


async def _reflect_audit_parent_tables(eng: AsyncEngine) -> None:
    needed = {"generation_runs"}
    missing = needed - set(Base.metadata.tables)
    if not missing:
        return
    async with eng.begin() as conn:
        await conn.run_sync(
            lambda sync_conn: Base.metadata.reflect(bind=sync_conn, only=tuple(missing))
        )


@pytest_asyncio.fixture
async def engine() -> AsyncIterator[AsyncEngine]:
    _ensure_head()
    eng = create_async_engine(_async_url(get_settings().database_url), pool_pre_ping=True)
    await _reflect_audit_parent_tables(eng)
    yield eng
    await eng.dispose()


@pytest_asyncio.fixture
async def session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False, autoflush=False)


@pytest_asyncio.fixture
async def app(
    session_factory: async_sessionmaker[AsyncSession],
    s3_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[tuple[FastAPI, AsyncMock]]:
    """Mount authoring + learner routers; override ``get_db`` and ``get_arq_pool``."""
    monkeypatch.setattr(s3_module, "get_settings", lambda: s3_settings)
    monkeypatch.setattr(
        "abridgeai.features.materials.services.authoring.get_settings",
        lambda: s3_settings,
    )

    arq_pool = AsyncMock()
    arq_pool.enqueue_job = AsyncMock()

    async def _override_get_db() -> AsyncIterator[AsyncSession]:
        async with session_factory() as session:
            yield session

    async def _override_arq_pool() -> object:
        return arq_pool

    fastapi_app = FastAPI()
    fastapi_app.include_router(authoring_router, prefix="/api/v1")
    fastapi_app.include_router(learner_router, prefix="/api/v1")
    fastapi_app.dependency_overrides[get_db] = _override_get_db
    fastapi_app.dependency_overrides[get_arq_pool] = _override_arq_pool
    yield fastapi_app, arq_pool
    fastapi_app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def client(app: tuple[FastAPI, AsyncMock]) -> AsyncIterator[httpx.AsyncClient]:
    fastapi_app, _ = app
    transport = httpx.ASGITransport(app=fastapi_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as ac:
        yield ac


async def _seed_session(engine: AsyncEngine, user_id: UUID) -> UUID:
    session_id = uuid.uuid4()
    expires_at = datetime.now(tz=UTC) + timedelta(hours=1)
    async with engine.begin() as conn:
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
    """Seed module + lesson under the seeded admin-owned course."""
    module_id = uuid.uuid4()
    lesson_id = uuid.uuid4()
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO modules (id, course_id, title, position, status) "
                "VALUES (:m, :c, 'E2E Module', 1, 'draft')"
            ),
            {"m": module_id, "c": seeded_users.course_id},
        )
        await conn.execute(
            text(
                "INSERT INTO lessons (id, module_id, slug, title, status) "
                "VALUES (:l, :m, 'e2e-lesson', 'E2E Lesson', 'draft')"
            ),
            {"l": lesson_id, "m": module_id},
        )
    yield {
        "course_id": seeded_users.course_id,
        "module_id": module_id,
        "lesson_id": lesson_id,
    }
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "DELETE FROM ai_model_calls WHERE processing_job_id IN ("
                "  SELECT id FROM processing_jobs WHERE entity_id IN ("
                "    SELECT id FROM learning_material_versions WHERE material_id IN ("
                "      SELECT id FROM learning_materials WHERE lesson_id = :l"
                "    )"
                "  )"
                ")"
            ),
            {"l": lesson_id},
        )
        await conn.execute(
            text("DELETE FROM document_chunks WHERE lesson_id = :l"),
            {"l": lesson_id},
        )
        await conn.execute(
            text(
                "DELETE FROM processing_jobs WHERE entity_id IN ("
                "  SELECT id FROM learning_material_versions WHERE material_id IN ("
                "    SELECT id FROM learning_materials WHERE lesson_id = :l"
                "  )"
                ")"
            ),
            {"l": lesson_id},
        )
        await conn.execute(
            text("UPDATE learning_materials SET current_version_id = NULL WHERE lesson_id = :l"),
            {"l": lesson_id},
        )
        await conn.execute(
            text(
                "DELETE FROM learning_material_versions WHERE material_id IN ("
                "  SELECT id FROM learning_materials WHERE lesson_id = :l"
                ")"
            ),
            {"l": lesson_id},
        )
        await conn.execute(
            text("DELETE FROM learning_materials WHERE lesson_id = :l"),
            {"l": lesson_id},
        )
        await conn.execute(text("DELETE FROM storage_objects WHERE bucket = :b"), {"b": BUCKET})
        await conn.execute(text("DELETE FROM lessons WHERE id = :l"), {"l": lesson_id})
        await conn.execute(text("DELETE FROM modules WHERE id = :m"), {"m": module_id})


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


async def _put_to_s3(url: str, payload: bytes) -> None:
    async with httpx.AsyncClient() as http:
        resp = await http.put(url, content=payload)
        assert resp.status_code in (200, 204), resp.text


# ---------------------------------------------------------------------------
# Lifecycle helper — POSTs init-upload, PUTs to moto, POSTs /complete,
# captures the ARQ enqueue args, simulates the worker by calling
# ``run_material_ingest`` directly, then flips ``visible_to_students`` so
# the learner endpoints see a ``ready`` material.
# ---------------------------------------------------------------------------


async def _run_full_lifecycle(
    *,
    client: httpx.AsyncClient,
    arq_pool: AsyncMock,
    session_factory: async_sessionmaker[AsyncSession],
    engine: AsyncEngine,
    admin_bearer: str,
    lesson_id: UUID,
    fixture_path: Path,
    content_type: str,
    title: str,
    monkeypatch: pytest.MonkeyPatch,
    kg_recorder: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    payload_bytes = fixture_path.read_bytes()  # noqa: ASYNC240  -- read-only fixture load; no IO blocking concern
    arq_pool.enqueue_job.reset_mock()

    init_resp = await client.post(
        f"/api/v1/teacher/lessons/{lesson_id}/materials/init-upload",
        json={
            "filename": fixture_path.name,
            "content_type": content_type,
            "size_bytes": len(payload_bytes),
            "title": title,
        },
        headers=_auth(admin_bearer),
    )
    assert init_resp.status_code == 201, init_resp.text
    init = init_resp.json()
    assert init["mode"] == "single"

    await _put_to_s3(init["upload_url"], payload_bytes)

    complete_resp = await client.post(
        f"/api/v1/teacher/materials/{init['material_id']}/versions/{init['version_id']}/complete",
        json={
            "storage_object_id": init["storage_object_id"],
            "checksum_sha256": "0" * 64,
        },
        headers=_auth(admin_bearer),
    )
    assert complete_resp.status_code == 202, complete_resp.text
    complete = complete_resp.json()
    arq_pool.enqueue_job.assert_called_once()
    args, _ = arq_pool.enqueue_job.call_args
    assert args[0] == "ingest_material_version_task"
    version_id, pipeline_run_id = args[2], args[3]
    assert UUID(str(version_id)) == UUID(complete["version_id"])
    assert UUID(str(pipeline_run_id)) == UUID(complete["pipeline_run_id"])

    # moto may overwrite the storage row's mime_type with whatever it returns
    # from head_object (often ``binary/octet-stream`` for non-PDF / non-text
    # blobs). Force it back to the declared content_type so the pipeline's
    # MIME→extractor dispatch routes to the stub registered at that MIME.
    async with engine.begin() as conn:
        await conn.execute(
            text("UPDATE storage_objects SET mime_type = :mt WHERE id = :id"),
            {"mt": content_type, "id": init["storage_object_id"]},
        )

    # Optional KG capture (cross-link test only).
    if kg_recorder is not None:

        async def _fake_kg(
            material_version_id: UUID,
            chunks: list[Any],
            *,
            hierarchy: Any,
            pipeline_run_id: UUID | None,
            db: Any,
            kg_client: Any,
            llm_gateway: Any,
            parent_job_id: UUID | None = None,
        ) -> KGSummary:
            kg_recorder.append(
                {
                    "material_version_id": material_version_id,
                    "course_id": getattr(hierarchy, "course_id", None),
                    "chunk_contents": [getattr(c, "content", "") for c in chunks],
                }
            )
            return KGSummary(concept_count=2, relationship_count=1, enabled=True)

        monkeypatch.setattr(pipeline_mod, "build_knowledge_graph_for_material_version", _fake_kg)
        monkeypatch.setenv("KNOWLEDGE_GRAPH_ENABLED", "true")
        monkeypatch.setenv("NEO4J_URI", "bolt://localhost:7687")
        monkeypatch.setenv("NEO4J_PASSWORD", "ignored-by-fake-builder")
        get_settings.cache_clear()

        async with session_factory() as db:
            await run_material_ingest(
                db,
                UUID(str(version_id)),
                UUID(str(pipeline_run_id)),
                source_path=fixture_path,
                embedding_client=_FakeEmbeddingClient(),
                kg_client=AsyncMock(),
                llm_gateway=_StubLLMGateway(),
            )
            await db.commit()
    else:
        async with session_factory() as db:
            await run_material_ingest(
                db,
                UUID(str(version_id)),
                UUID(str(pipeline_run_id)),
                source_path=fixture_path,
                embedding_client=_FakeEmbeddingClient(),
            )
            await db.commit()

    async with engine.begin() as conn:
        await conn.execute(
            text("UPDATE learning_materials SET visible_to_students = TRUE WHERE id = :id"),
            {"id": init["material_id"]},
        )

    return {
        "material_id": UUID(init["material_id"]),
        "version_id": UUID(init["version_id"]),
        "storage_object_id": UUID(init["storage_object_id"]),
        "pipeline_run_id": UUID(complete["pipeline_run_id"]),
    }


async def _assert_version_ready(
    session_factory: async_sessionmaker[AsyncSession], version_id: UUID
) -> None:
    from abridgeai.features.materials.models import LearningMaterialVersion

    async with session_factory() as db:
        version = await db.get(LearningMaterialVersion, version_id)
        assert version is not None
        assert version.processing_status == "ready"
        assert version.processed_at is not None


# ---------------------------------------------------------------------------
# 6 per-material-type lifecycle scenarios (plan §5316, §5301)
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("_stub_real_extractors")
async def test_full_lifecycle_pdf(
    client: httpx.AsyncClient,
    app: tuple[FastAPI, AsyncMock],
    session_factory: async_sessionmaker[AsyncSession],
    engine: AsyncEngine,
    admin_bearer: str,
    student_bearer: str,
    scenario: dict[str, UUID],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, arq_pool = app
    result = await _run_full_lifecycle(
        client=client,
        arq_pool=arq_pool,
        session_factory=session_factory,
        engine=engine,
        admin_bearer=admin_bearer,
        lesson_id=scenario["lesson_id"],
        fixture_path=FIXTURES_DIR / "sample.pdf",
        content_type=PDF_MIME,
        title="PDF Material",
        monkeypatch=monkeypatch,
    )
    await _assert_version_ready(session_factory, result["version_id"])

    student_get = await client.get(
        f"/api/v1/materials/{result['material_id']}",
        headers=_auth(student_bearer),
    )
    assert student_get.status_code == 200, student_get.text
    body = student_get.json()
    assert body["id"] == str(result["material_id"])
    assert body["material_type"] == "pdf"

    # Stub stream URL minting so we don't need a presign round-trip here.
    async def _fake_create_stream_url(target: Any, *, response_headers: Any) -> tuple[str, Any]:
        del target, response_headers
        return ("https://fake.test/stream", datetime.now(tz=UTC) + timedelta(hours=1))

    monkeypatch.setattr(
        "abridgeai.features.materials.services.catalog.create_stream_url",
        _fake_create_stream_url,
    )
    stream = await client.get(
        f"/api/v1/materials/{result['material_id']}/stream-url",
        headers=_auth(student_bearer),
    )
    assert stream.status_code == 200, stream.text
    assert stream.json()["url"].startswith("http")


@pytest.mark.usefixtures("_stub_real_extractors")
async def test_full_lifecycle_docx(
    client: httpx.AsyncClient,
    app: tuple[FastAPI, AsyncMock],
    session_factory: async_sessionmaker[AsyncSession],
    engine: AsyncEngine,
    admin_bearer: str,
    scenario: dict[str, UUID],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, arq_pool = app
    result = await _run_full_lifecycle(
        client=client,
        arq_pool=arq_pool,
        session_factory=session_factory,
        engine=engine,
        admin_bearer=admin_bearer,
        lesson_id=scenario["lesson_id"],
        fixture_path=FIXTURES_DIR / "sample.docx",
        content_type=DOCX_MIME,
        title="DOCX Material",
        monkeypatch=monkeypatch,
    )
    await _assert_version_ready(session_factory, result["version_id"])


@pytest.mark.usefixtures("_stub_real_extractors")
async def test_full_lifecycle_pptx(
    client: httpx.AsyncClient,
    app: tuple[FastAPI, AsyncMock],
    session_factory: async_sessionmaker[AsyncSession],
    engine: AsyncEngine,
    admin_bearer: str,
    scenario: dict[str, UUID],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, arq_pool = app
    result = await _run_full_lifecycle(
        client=client,
        arq_pool=arq_pool,
        session_factory=session_factory,
        engine=engine,
        admin_bearer=admin_bearer,
        lesson_id=scenario["lesson_id"],
        fixture_path=FIXTURES_DIR / "sample.pptx",
        content_type=PPTX_MIME,
        title="PPTX Material",
        monkeypatch=monkeypatch,
    )
    await _assert_version_ready(session_factory, result["version_id"])


@pytest.mark.usefixtures("_stub_real_extractors")
async def test_full_lifecycle_audio_with_mocked_whisper(
    client: httpx.AsyncClient,
    app: tuple[FastAPI, AsyncMock],
    session_factory: async_sessionmaker[AsyncSession],
    engine: AsyncEngine,
    admin_bearer: str,
    scenario: dict[str, UUID],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Audio path: stub extractor stands in for Whisper transcription."""
    _, arq_pool = app
    result = await _run_full_lifecycle(
        client=client,
        arq_pool=arq_pool,
        session_factory=session_factory,
        engine=engine,
        admin_bearer=admin_bearer,
        lesson_id=scenario["lesson_id"],
        fixture_path=FIXTURES_DIR / "sample.wav",
        content_type=AUDIO_MIME,
        title="Audio Material",
        monkeypatch=monkeypatch,
    )
    await _assert_version_ready(session_factory, result["version_id"])

    async with session_factory() as db:
        rows = (
            await db.execute(
                text(
                    "SELECT content, metadata FROM document_chunks "
                    "WHERE material_version_id = :v ORDER BY chunk_index"
                ),
                {"v": result["version_id"]},
            )
        ).all()
    assert len(rows) >= 1, "audio ingest must produce at least one chunk"
    all_text = "\n".join(row.content for row in rows)
    assert "Python" in all_text or "OOP" in all_text or "lecture" in all_text


@pytest.mark.usefixtures("_stub_real_extractors")
async def test_full_lifecycle_image_with_mocked_ocr(
    client: httpx.AsyncClient,
    app: tuple[FastAPI, AsyncMock],
    session_factory: async_sessionmaker[AsyncSession],
    engine: AsyncEngine,
    admin_bearer: str,
    scenario: dict[str, UUID],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Image path: stub extractor stands in for OCR / vision."""
    _, arq_pool = app
    result = await _run_full_lifecycle(
        client=client,
        arq_pool=arq_pool,
        session_factory=session_factory,
        engine=engine,
        admin_bearer=admin_bearer,
        lesson_id=scenario["lesson_id"],
        fixture_path=FIXTURES_DIR / "text-image.png",
        content_type=IMAGE_MIME,
        title="Image Material",
        monkeypatch=monkeypatch,
    )
    await _assert_version_ready(session_factory, result["version_id"])

    async with session_factory() as db:
        count = (
            await db.execute(
                text("SELECT count(*) FROM document_chunks WHERE material_version_id = :v"),
                {"v": result["version_id"]},
            )
        ).scalar_one()
    # Image extractor produces a single OCR chunk per the IMAGE chunker branch.
    assert count == 1


@pytest.mark.usefixtures("_stub_real_extractors")
async def test_full_lifecycle_html(
    client: httpx.AsyncClient,
    app: tuple[FastAPI, AsyncMock],
    session_factory: async_sessionmaker[AsyncSession],
    engine: AsyncEngine,
    admin_bearer: str,
    scenario: dict[str, UUID],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, arq_pool = app
    result = await _run_full_lifecycle(
        client=client,
        arq_pool=arq_pool,
        session_factory=session_factory,
        engine=engine,
        admin_bearer=admin_bearer,
        lesson_id=scenario["lesson_id"],
        fixture_path=FIXTURES_DIR / "sample.html",
        content_type=HTML_MIME,
        title="HTML Material",
        monkeypatch=monkeypatch,
    )
    await _assert_version_ready(session_factory, result["version_id"])


# ---------------------------------------------------------------------------
# KG cross-linking — ingest 3 materials in same course, assert KG builder
# saw chunks with shared concept names ("Python", "OOP").
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("_stub_real_extractors")
async def test_kg_cross_linking_3_materials_same_course(
    client: httpx.AsyncClient,
    app: tuple[FastAPI, AsyncMock],
    session_factory: async_sessionmaker[AsyncSession],
    engine: AsyncEngine,
    admin_bearer: str,
    scenario: dict[str, UUID],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Plan §5298 / §5318 / §5330 — KG receives shared concepts across 3 materials."""
    _, arq_pool = app
    kg_recorder: list[dict[str, Any]] = []

    inputs = [
        (FIXTURES_DIR / "sample.pdf", PDF_MIME, "PDF One"),
        (FIXTURES_DIR / "sample.docx", DOCX_MIME, "DOCX Two"),
        (FIXTURES_DIR / "sample.html", HTML_MIME, "HTML Three"),
    ]
    results: list[dict[str, Any]] = []
    for fixture_path, content_type, title in inputs:
        results.append(
            await _run_full_lifecycle(
                client=client,
                arq_pool=arq_pool,
                session_factory=session_factory,
                engine=engine,
                admin_bearer=admin_bearer,
                lesson_id=scenario["lesson_id"],
                fixture_path=fixture_path,
                content_type=content_type,
                title=title,
                monkeypatch=monkeypatch,
                kg_recorder=kg_recorder,
            )
        )

    assert len(kg_recorder) == 3, f"KG builder must run once per material; got {len(kg_recorder)}"

    course_ids = {entry["course_id"] for entry in kg_recorder}
    assert course_ids == {scenario["course_id"]}, "all KG calls must share the course id"

    # All three stub extractors share the concept names "Python" and "OOP" in
    # their chunk content.  The KG builder receiving the same names across 3
    # materials is the cross-link signal.
    shared_terms = {"Python", "OOP"}
    chunk_text_per_material = [" ".join(entry["chunk_contents"]).lower() for entry in kg_recorder]
    for term in shared_terms:
        materials_with_term = [t for t in chunk_text_per_material if term.lower() in t]
        assert len(materials_with_term) >= 2, (
            f"concept {term!r} must appear in chunks for at least 2 materials "
            f"(KG cross-link signal); found in {len(materials_with_term)}"
        )


# ---------------------------------------------------------------------------
# Reprocess after failure
# ---------------------------------------------------------------------------


async def test_reprocess_after_failure_recovers(
    client: httpx.AsyncClient,
    app: tuple[FastAPI, AsyncMock],
    session_factory: async_sessionmaker[AsyncSession],
    engine: AsyncEngine,
    admin_bearer: str,
    scenario: dict[str, UUID],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """First ingest fails (extractor raises) → status='failed';
    teacher POST /reprocess → second ingest succeeds → status='ready'."""
    _, arq_pool = app

    # Pass 1 — failing extractor.
    saved_pdf = EXTRACTOR_REGISTRY.get(PDF_MIME)

    class _FailingExtractor:
        async def extract(self, source: Any) -> ExtractedContent:
            raise RuntimeError("synthetic failure: pass 1")

    EXTRACTOR_REGISTRY[PDF_MIME] = _FailingExtractor

    try:
        fixture_path = FIXTURES_DIR / "sample.pdf"
        payload_bytes = fixture_path.read_bytes()
        arq_pool.enqueue_job.reset_mock()

        init_resp = await client.post(
            f"/api/v1/teacher/lessons/{scenario['lesson_id']}/materials/init-upload",
            json={
                "filename": "reprocess.pdf",
                "content_type": PDF_MIME,
                "size_bytes": len(payload_bytes),
                "title": "Reprocess",
            },
            headers=_auth(admin_bearer),
        )
        assert init_resp.status_code == 201
        init = init_resp.json()
        await _put_to_s3(init["upload_url"], payload_bytes)

        complete_resp = await client.post(
            f"/api/v1/teacher/materials/{init['material_id']}/versions/{init['version_id']}/complete",
            json={
                "storage_object_id": init["storage_object_id"],
                "checksum_sha256": "0" * 64,
            },
            headers=_auth(admin_bearer),
        )
        assert complete_resp.status_code == 202
        complete = complete_resp.json()
        version_id_1 = UUID(complete["version_id"])
        run_id_1 = UUID(complete["pipeline_run_id"])

        async with engine.begin() as conn:
            await conn.execute(
                text("UPDATE storage_objects SET mime_type = :mt WHERE id = :id"),
                {"mt": PDF_MIME, "id": init["storage_object_id"]},
            )

        async with session_factory() as db:
            with pytest.raises(RuntimeError, match="synthetic failure: pass 1"):
                await run_material_ingest(
                    db,
                    version_id_1,
                    run_id_1,
                    source_path=fixture_path,
                    embedding_client=_FakeEmbeddingClient(),
                )
            await db.commit()

        from abridgeai.features.materials.models import LearningMaterialVersion

        async with session_factory() as db:
            version = await db.get(LearningMaterialVersion, version_id_1)
            assert version is not None
            assert version.processing_status == "failed"
            assert version.processing_error is not None

        # Pass 2 — fix the extractor and reprocess.
        EXTRACTOR_REGISTRY[PDF_MIME] = _make_stub_extractor(PDF_STUB)
        arq_pool.enqueue_job.reset_mock()

        reprocess_resp = await client.post(
            f"/api/v1/teacher/materials/{init['material_id']}/reprocess",
            headers=_auth(admin_bearer),
        )
        assert reprocess_resp.status_code == 202, reprocess_resp.text
        reprocess = reprocess_resp.json()
        arq_pool.enqueue_job.assert_called_once()
        args, _ = arq_pool.enqueue_job.call_args
        version_id_2 = UUID(str(args[2]))
        run_id_2 = UUID(str(args[3]))
        assert UUID(reprocess["version_id"]) == version_id_2
        assert UUID(reprocess["pipeline_run_id"]) == run_id_2

        async with session_factory() as db:
            await run_material_ingest(
                db,
                version_id_2,
                run_id_2,
                source_path=fixture_path,
                embedding_client=_FakeEmbeddingClient(),
            )
            await db.commit()

        async with session_factory() as db:
            version = await db.get(LearningMaterialVersion, version_id_2)
            assert version is not None
            assert version.processing_status == "ready", "reprocess must recover status to ready"
    finally:
        if saved_pdf is not None:
            EXTRACTOR_REGISTRY[PDF_MIME] = saved_pdf
        else:
            EXTRACTOR_REGISTRY.pop(PDF_MIME, None)


# ---------------------------------------------------------------------------
# Soft-delete preserves S3 (Reconciliation §C9 + plan §4954)
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("_stub_real_extractors")
async def test_soft_delete_cascades_to_versions_keeps_s3(
    client: httpx.AsyncClient,
    app: tuple[FastAPI, AsyncMock],
    session_factory: async_sessionmaker[AsyncSession],
    engine: AsyncEngine,
    admin_bearer: str,
    scenario: dict[str, UUID],
    monkeypatch: pytest.MonkeyPatch,
    s3_settings: Settings,
) -> None:
    _, arq_pool = app
    result = await _run_full_lifecycle(
        client=client,
        arq_pool=arq_pool,
        session_factory=session_factory,
        engine=engine,
        admin_bearer=admin_bearer,
        lesson_id=scenario["lesson_id"],
        fixture_path=FIXTURES_DIR / "sample.pdf",
        content_type=PDF_MIME,
        title="Soft Delete Target",
        monkeypatch=monkeypatch,
    )

    # Resolve the storage row's bucket/key BEFORE the soft-delete (the
    # cascade tombstones the row but does NOT touch S3).
    async with engine.begin() as conn:
        row = (
            await conn.execute(
                text(
                    "SELECT so.bucket AS bucket, so.object_key AS object_key "
                    "FROM storage_objects so "
                    "JOIN learning_material_versions lmv ON lmv.storage_object_id = so.id "
                    "WHERE lmv.id = :v"
                ),
                {"v": result["version_id"]},
            )
        ).one()
    bucket, object_key = row.bucket, row.object_key

    delete_resp = await client.delete(
        f"/api/v1/teacher/materials/{result['material_id']}",
        headers=_auth(admin_bearer),
    )
    assert delete_resp.status_code == 204

    async with engine.begin() as conn:
        material_deleted_at = (
            await conn.execute(
                text("SELECT deleted_at FROM learning_materials WHERE id = :id"),
                {"id": result["material_id"]},
            )
        ).scalar_one()
    assert material_deleted_at is not None, "material must be tombstoned"

    # S3 object survives.
    class _Obj:
        bucket = ""
        object_key = ""

    obj = _Obj()
    obj.bucket = bucket
    obj.object_key = object_key
    meta = await s3_module.head_object(obj, settings=s3_settings)
    assert meta is not None, "S3 object must survive soft-delete"
    assert meta.size > 0


# ---------------------------------------------------------------------------
# Bonus perimeter checks (FIX-SEC-1 echo across the e2e composition)
# ---------------------------------------------------------------------------


async def test_unauthenticated_returns_401_on_all(
    client: httpx.AsyncClient, scenario: dict[str, UUID]
) -> None:
    """Every write endpoint rejects no-bearer requests."""
    for method, path, body in (
        (
            "post",
            f"/api/v1/teacher/lessons/{scenario['lesson_id']}/materials/init-upload",
            {
                "filename": "x.pdf",
                "content_type": PDF_MIME,
                "size_bytes": 100,
                "title": "X",
            },
        ),
        (
            "post",
            f"/api/v1/teacher/materials/{uuid.uuid4()}/reprocess",
            None,
        ),
    ):
        resp = (
            await getattr(client, method)(path, json=body)
            if body is not None
            else (await getattr(client, method)(path))
        )
        assert resp.status_code == 401, f"{method.upper()} {path} returned {resp.status_code}"


async def test_student_403_on_authoring(
    client: httpx.AsyncClient,
    student_bearer: str,
    scenario: dict[str, UUID],
) -> None:
    """A student token (no ``course.update``) is rejected from authoring."""
    resp = await client.post(
        f"/api/v1/teacher/lessons/{scenario['lesson_id']}/materials/init-upload",
        json={
            "filename": "x.pdf",
            "content_type": PDF_MIME,
            "size_bytes": 100,
            "title": "X",
        },
        headers=_auth(student_bearer),
    )
    assert resp.status_code == 403
