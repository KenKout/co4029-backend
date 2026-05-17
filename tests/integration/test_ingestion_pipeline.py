"""Materials ingestion pipeline integration tests (T4.4).

Mocks every external dependency: extraction is faked through purpose-built
test extractors registered at sentinel MIMEs, embeddings come from an
in-process fake that still writes ``ai_model_calls`` rows so the
``pipeline_run_id`` audit-grouping invariant can be asserted, KG runs are
faked at the ``build_knowledge_graph_for_material_version`` boundary, and
``download_to_temp`` is monkey-patched.

Real wiring exercised end-to-end:

  * ``run_material_ingest`` orchestration (5 stages + failure capture)
  * ``DocumentChunk`` persistence with denormalized course/module/lesson FKs
  * ``LearningMaterialVersion.processing_status`` + ``ProcessingJob``
    state-machine transitions
  * ``ai_model_calls.pipeline_run_id`` propagation through embeddings
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
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

import abridgeai.features.courses.models  # noqa: F401  -- register course/module/lesson FK targets
import abridgeai.features.identity.models  # noqa: F401  -- register users + storage_objects FK targets
from abridgeai.ai.extraction import (
    EXTRACTOR_REGISTRY,
    ExtractedContent,
    SourceLocation,
)
from abridgeai.ai.knowledge_graph.schemas import KGSummary
from abridgeai.ai.llm.audit import write_ai_model_call
from abridgeai.ai.llm.roles import LLMRole
from abridgeai.core.config import get_settings
from abridgeai.features.materials import ingestion as ingestion_pkg
from abridgeai.features.materials.ingestion import pipeline as pipeline_mod
from abridgeai.features.materials.ingestion import run_material_ingest
from abridgeai.features.materials.models import LearningMaterialVersion

_TEST_PDF_MIME = "application/x-test-pdf"
_TEST_AUDIO_MIME = "audio/x-test"
_TEST_VIDEO_MIME = "video/x-test"
_TEST_IMAGE_MIME = "image/x-test"
_TEST_FAILING_MIME = "application/x-test-fail"
_EMBEDDING_DIM = 1536


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


def _vec(values: list[float]) -> list[float]:
    padded = list(values) + [0.0] * (_EMBEDDING_DIM - len(values))
    return [float(v) for v in padded[:_EMBEDDING_DIM]]


class _PdfStubExtractor:
    async def extract(self, source: Any) -> ExtractedContent:
        text_body = (
            "[Page 1]\nThe quick brown fox jumps over the lazy dog.\n\n"
            "[Page 2]\nLorem ipsum dolor sit amet, consectetur adipiscing elit."
        )
        return ExtractedContent(
            text=text_body,
            metadata={"page_count": 2},
            source_type="pdf",
            source_locations=[SourceLocation(page=1), SourceLocation(page=2)],
        )


class _AudioStubExtractor:
    async def extract(self, source: Any) -> ExtractedContent:
        segments = [
            ("Hello world.", 0, 10_000),
            ("This is a second utterance.", 10_000, 20_000),
            ("Closing thoughts.", 20_000, 30_000),
        ]
        body = "\n\n".join(seg for seg, _, _ in segments)
        locations = [
            SourceLocation(timestamp_start_ms=start, timestamp_end_ms=end)
            for _, start, end in segments
        ]
        return ExtractedContent(
            text=body,
            metadata={"segment_count": 3, "duration_seconds": 30.0},
            source_type="audio",
            source_locations=locations,
        )


class _VideoStubExtractor:
    async def extract(self, source: Any) -> ExtractedContent:
        segments = [
            ("[Audio @ 0ms] Welcome to the lecture.", 0, 4_000),
            ("[Frame OCR @ 1000ms] Slide title: Algorithms", 1_000, 1_000),
            ("[Audio @ 5000ms] Today we cover binary search.", 5_000, 9_000),
            ("[Frame OCR @ 6000ms] Pseudocode: while lo <= hi", 6_000, 6_000),
        ]
        body = "\n\n".join(seg for seg, _, _ in segments)
        locations = [
            SourceLocation(timestamp_start_ms=start, timestamp_end_ms=end)
            for _, start, end in segments
        ]
        return ExtractedContent(
            text=body,
            metadata={"audio_segment_count": 2, "frame_count": 2},
            source_type="video",
            source_locations=locations,
        )


class _ImageStubExtractor:
    async def extract(self, source: Any) -> ExtractedContent:
        body = "Slide title: Sorting Algorithms\nMerge Sort\nTime: O(n log n)"
        return ExtractedContent(
            text=body,
            metadata={"ocr_provider": "stub", "image_size": (640, 480)},
            source_type="image",
            source_locations=[SourceLocation(bbox=(0.0, 0.0, 640.0, 480.0))],
        )


class _FailingExtractor:
    async def extract(self, source: Any) -> ExtractedContent:
        raise RuntimeError("synthetic extraction failure: test")


class _FakeEmbeddingClient:
    """Returns deterministic vectors AND writes ``ai_model_calls`` rows.

    Mimics the real ``EmbeddingClient.embed`` contract so the pipeline-run-id
    audit assertion can hit a real DB row without any HTTP.
    """

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
        return [_vec([0.1, 0.2, 0.3, float(i % 7) / 10.0]) for i in range(len(texts))]


@pytest.fixture(autouse=True)
def _reset_settings_cache() -> None:
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def _stub_extractors() -> AsyncIterator[None]:
    """Register sentinel MIMEs so the pipeline routes to test extractors."""
    EXTRACTOR_REGISTRY[_TEST_PDF_MIME] = _PdfStubExtractor
    EXTRACTOR_REGISTRY[_TEST_AUDIO_MIME] = _AudioStubExtractor
    EXTRACTOR_REGISTRY[_TEST_VIDEO_MIME] = _VideoStubExtractor
    EXTRACTOR_REGISTRY[_TEST_IMAGE_MIME] = _ImageStubExtractor
    EXTRACTOR_REGISTRY[_TEST_FAILING_MIME] = _FailingExtractor
    yield
    for mime in (
        _TEST_PDF_MIME,
        _TEST_AUDIO_MIME,
        _TEST_VIDEO_MIME,
        _TEST_IMAGE_MIME,
        _TEST_FAILING_MIME,
    ):
        EXTRACTOR_REGISTRY.pop(mime, None)


@pytest_asyncio.fixture
async def engine() -> AsyncIterator[AsyncEngine]:
    _ensure_head()
    eng = create_async_engine(_async_url(get_settings().database_url), pool_pre_ping=True)
    await _reflect_audit_parent_tables(eng)
    yield eng
    await eng.dispose()


async def _reflect_audit_parent_tables(eng: AsyncEngine) -> None:
    from abridgeai.core.db import Base

    needed = {"generation_runs"}
    missing = needed - set(Base.metadata.tables)
    if not missing:
        return
    async with eng.begin() as conn:
        await conn.run_sync(
            lambda sync_conn: Base.metadata.reflect(bind=sync_conn, only=tuple(missing))
        )


@pytest_asyncio.fixture
async def session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False, autoflush=False)


@dataclass
class _Scope:
    org_id: UUID
    owner_id: UUID
    course_id: UUID
    module_id: UUID
    lesson_id: UUID
    storage_id: UUID
    material_id: UUID
    version_id: UUID
    mime_type: str


async def _seed_scope(engine: AsyncEngine, *, mime_type: str, material_type: str) -> _Scope:
    scope = _Scope(
        org_id=uuid.uuid4(),
        owner_id=uuid.uuid4(),
        course_id=uuid.uuid4(),
        module_id=uuid.uuid4(),
        lesson_id=uuid.uuid4(),
        storage_id=uuid.uuid4(),
        material_id=uuid.uuid4(),
        version_id=uuid.uuid4(),
        mime_type=mime_type,
    )
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO organizations (id, slug, name, status) VALUES (:id, :s, :n, 'active')"
            ),
            {"id": scope.org_id, "s": f"ing-{scope.org_id.hex[:8]}", "n": "Ing Org"},
        )
        await conn.execute(
            text("INSERT INTO users (id, primary_email, status) VALUES (:id, :e, 'active')"),
            {"id": scope.owner_id, "e": f"ing-{scope.owner_id.hex[:8]}@test.local"},
        )
        await conn.execute(
            text(
                "INSERT INTO courses (id, organization_id, owner_user_id, slug, title, status) "
                "VALUES (:id, :o, :u, :s, 'Ing Course', 'draft')"
            ),
            {
                "id": scope.course_id,
                "o": scope.org_id,
                "u": scope.owner_id,
                "s": f"ing-c-{scope.course_id.hex[:6]}",
            },
        )
        await conn.execute(
            text("INSERT INTO modules (id, course_id, title, position) VALUES (:id, :c, 'Mod', 1)"),
            {"id": scope.module_id, "c": scope.course_id},
        )
        await conn.execute(
            text("INSERT INTO lessons (id, module_id, slug, title) VALUES (:id, :m, :s, 'Lsn')"),
            {
                "id": scope.lesson_id,
                "m": scope.module_id,
                "s": f"ing-l-{scope.lesson_id.hex[:6]}",
            },
        )
        await conn.execute(
            text(
                "INSERT INTO storage_objects (id, bucket, object_key, mime_type) "
                "VALUES (:id, 'test-bucket', :key, :mime)"
            ),
            {
                "id": scope.storage_id,
                "key": f"ing/{scope.storage_id.hex}",
                "mime": mime_type,
            },
        )
        await conn.execute(
            text(
                "INSERT INTO learning_materials (id, lesson_id, title, material_type) "
                "VALUES (:id, :l, 'Test Mat', :mt)"
            ),
            {"id": scope.material_id, "l": scope.lesson_id, "mt": material_type},
        )
        await conn.execute(
            text(
                "INSERT INTO learning_material_versions "
                "(id, material_id, storage_object_id, version_no, processing_status) "
                "VALUES (:id, :m, :so, 1, 'pending')"
            ),
            {"id": scope.version_id, "m": scope.material_id, "so": scope.storage_id},
        )
    return scope


async def _teardown_scope(engine: AsyncEngine, scope: _Scope) -> None:
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "DELETE FROM ai_model_calls WHERE processing_job_id IN ("
                "SELECT id FROM processing_jobs WHERE entity_id = :v)"
            ),
            {"v": scope.version_id},
        )
        await conn.execute(
            text("DELETE FROM document_chunks WHERE material_version_id = :v"),
            {"v": scope.version_id},
        )
        await conn.execute(
            text("DELETE FROM processing_jobs WHERE entity_id = :v"),
            {"v": scope.version_id},
        )
        await conn.execute(
            text("DELETE FROM learning_material_versions WHERE id = :id"),
            {"id": scope.version_id},
        )
        await conn.execute(
            text("DELETE FROM learning_materials WHERE id = :id"),
            {"id": scope.material_id},
        )
        await conn.execute(
            text("DELETE FROM storage_objects WHERE id = :id"),
            {"id": scope.storage_id},
        )
        await conn.execute(text("DELETE FROM lessons WHERE id = :id"), {"id": scope.lesson_id})
        await conn.execute(text("DELETE FROM modules WHERE id = :id"), {"id": scope.module_id})
        await conn.execute(text("DELETE FROM courses WHERE id = :id"), {"id": scope.course_id})
        await conn.execute(text("DELETE FROM users WHERE id = :id"), {"id": scope.owner_id})
        await conn.execute(text("DELETE FROM organizations WHERE id = :id"), {"id": scope.org_id})


@asynccontextmanager
async def _scope_for(
    engine: AsyncEngine, *, mime: str, material_type: str
) -> AsyncIterator[_Scope]:
    scope = await _seed_scope(engine, mime_type=mime, material_type=material_type)
    try:
        yield scope
    finally:
        await _teardown_scope(engine, scope)


def test_public_api_smoke() -> None:
    assert run_material_ingest is ingestion_pkg.run_material_ingest


@pytest.mark.asyncio
@pytest.mark.usefixtures("_stub_extractors")
async def test_pdf_full_pipeline(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    async with _scope_for(engine, mime=_TEST_PDF_MIME, material_type="pdf") as scope:
        run_id = uuid.uuid4()
        embedder = _FakeEmbeddingClient()

        async with session_factory() as db:
            await run_material_ingest(
                db,
                scope.version_id,
                run_id,
                source_path=Path("/dev/null"),
                embedding_client=embedder,
            )
            await db.commit()

        async with session_factory() as db:
            version = await db.get(LearningMaterialVersion, scope.version_id)
            assert version is not None
            assert version.processing_status == "ready"
            assert version.processed_at is not None
            chunk_count = (
                await db.execute(
                    text("SELECT count(*) FROM document_chunks WHERE material_version_id = :v"),
                    {"v": scope.version_id},
                )
            ).scalar_one()
            assert chunk_count >= 1

        assert len(embedder.calls) == 1
        assert embedder.calls[0]["pipeline_run_id"] == run_id


@pytest.mark.asyncio
@pytest.mark.usefixtures("_stub_extractors")
async def test_audio_timestamps_preserved(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    async with _scope_for(engine, mime=_TEST_AUDIO_MIME, material_type="audio") as scope:
        run_id = uuid.uuid4()
        async with session_factory() as db:
            await run_material_ingest(
                db,
                scope.version_id,
                run_id,
                source_path=Path("/dev/null"),
                embedding_client=_FakeEmbeddingClient(),
            )
            await db.commit()

        async with session_factory() as db:
            rows = (
                await db.execute(
                    text(
                        "SELECT metadata FROM document_chunks "
                        "WHERE material_version_id = :v ORDER BY chunk_index"
                    ),
                    {"v": scope.version_id},
                )
            ).all()
        assert len(rows) >= 1
        starts = [row.metadata.get("timestamp_start_ms") for row in rows]
        ends = [row.metadata.get("timestamp_end_ms") for row in rows]
        assert any(s is not None for s in starts)
        assert any(e is not None for e in ends)


@pytest.mark.asyncio
@pytest.mark.usefixtures("_stub_extractors")
async def test_video_interleaves_transcript_and_ocr(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    async with _scope_for(engine, mime=_TEST_VIDEO_MIME, material_type="video") as scope:
        async with session_factory() as db:
            await run_material_ingest(
                db,
                scope.version_id,
                uuid.uuid4(),
                source_path=Path("/dev/null"),
                embedding_client=_FakeEmbeddingClient(),
            )
            await db.commit()

        async with session_factory() as db:
            rows = (
                await db.execute(
                    text(
                        "SELECT content FROM document_chunks "
                        "WHERE material_version_id = :v ORDER BY chunk_index"
                    ),
                    {"v": scope.version_id},
                )
            ).all()
        all_text = "\n".join(r.content for r in rows)
        assert "Audio @" in all_text
        assert "Frame OCR @" in all_text


@pytest.mark.asyncio
@pytest.mark.usefixtures("_stub_extractors")
async def test_image_single_chunk(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    async with _scope_for(engine, mime=_TEST_IMAGE_MIME, material_type="image") as scope:
        async with session_factory() as db:
            await run_material_ingest(
                db,
                scope.version_id,
                uuid.uuid4(),
                source_path=Path("/dev/null"),
                embedding_client=_FakeEmbeddingClient(),
            )
            await db.commit()

        async with session_factory() as db:
            count = (
                await db.execute(
                    text("SELECT count(*) FROM document_chunks WHERE material_version_id = :v"),
                    {"v": scope.version_id},
                )
            ).scalar_one()
        assert count == 1


@pytest.mark.asyncio
@pytest.mark.usefixtures("_stub_extractors")
async def test_source_path_provided_skips_download(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    download_mock = AsyncMock(side_effect=AssertionError("download_to_temp must NOT be called"))
    monkeypatch.setattr(pipeline_mod, "download_to_temp", download_mock)

    async with _scope_for(engine, mime=_TEST_PDF_MIME, material_type="pdf") as scope:  # noqa: SIM117
        async with session_factory() as db:
            await run_material_ingest(
                db,
                scope.version_id,
                uuid.uuid4(),
                source_path=Path("/dev/null"),
                embedding_client=_FakeEmbeddingClient(),
            )
            await db.commit()

    download_mock.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.usefixtures("_stub_extractors")
async def test_source_path_none_calls_download(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_local = tmp_path / "downloaded.bin"
    fake_local.write_bytes(b"fake")

    async def _fake_download(storage_object: Any, dest_dir: Path, **_: Any) -> Path:
        return fake_local

    download_mock = AsyncMock(side_effect=_fake_download)
    monkeypatch.setattr(pipeline_mod, "download_to_temp", download_mock)

    async with _scope_for(engine, mime=_TEST_PDF_MIME, material_type="pdf") as scope:  # noqa: SIM117
        async with session_factory() as db:
            await run_material_ingest(
                db,
                scope.version_id,
                uuid.uuid4(),
                embedding_client=_FakeEmbeddingClient(),
            )
            await db.commit()

    assert download_mock.await_count == 1


@pytest.mark.asyncio
@pytest.mark.usefixtures("_stub_extractors")
async def test_pipeline_run_id_threads_to_audit(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    async with _scope_for(engine, mime=_TEST_PDF_MIME, material_type="pdf") as scope:
        run_id = uuid.uuid4()
        async with session_factory() as db:
            await run_material_ingest(
                db,
                scope.version_id,
                run_id,
                source_path=Path("/dev/null"),
                embedding_client=_FakeEmbeddingClient(),
            )
            await db.commit()

        async with engine.connect() as conn:
            rows = (
                await conn.execute(
                    text(
                        "SELECT role, stage_name, status FROM ai_model_calls "
                        "WHERE pipeline_run_id = :rid"
                    ),
                    {"rid": run_id},
                )
            ).all()
    assert len(rows) >= 1
    assert any(r.role == "embedding" and r.stage_name == "embedding" for r in rows)
    assert all(r.status == "success" for r in rows)


@pytest.mark.asyncio
@pytest.mark.usefixtures("_stub_extractors")
async def test_failure_captures_error_state(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    async with _scope_for(engine, mime=_TEST_FAILING_MIME, material_type="pdf") as scope:
        async with session_factory() as db:
            with pytest.raises(RuntimeError, match="synthetic extraction failure"):
                await run_material_ingest(
                    db,
                    scope.version_id,
                    uuid.uuid4(),
                    source_path=Path("/dev/null"),
                    embedding_client=_FakeEmbeddingClient(),
                )
            await db.commit()

        async with session_factory() as db:
            version = await db.get(LearningMaterialVersion, scope.version_id)
            assert version is not None
            assert version.processing_status == "failed"
            assert version.processing_error is not None
            assert "synthetic extraction failure" in version.processing_error

            job_row = (
                await db.execute(
                    text(
                        "SELECT status, error_message FROM processing_jobs "
                        "WHERE entity_id = :v ORDER BY created_at DESC LIMIT 1"
                    ),
                    {"v": scope.version_id},
                )
            ).first()
            assert job_row is not None
            assert job_row.status == "failed"
            assert job_row.error_message is not None
            assert "synthetic extraction failure" in job_row.error_message


@pytest.mark.asyncio
@pytest.mark.usefixtures("_stub_extractors")
async def test_kg_disabled_skips_stage(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("KNOWLEDGE_GRAPH_ENABLED", raising=False)
    get_settings.cache_clear()

    kg_mock = AsyncMock(side_effect=AssertionError("KG builder must NOT be called"))
    monkeypatch.setattr(pipeline_mod, "build_knowledge_graph_for_material_version", kg_mock)

    async with _scope_for(engine, mime=_TEST_AUDIO_MIME, material_type="audio") as scope:
        async with session_factory() as db:
            await run_material_ingest(
                db,
                scope.version_id,
                uuid.uuid4(),
                source_path=Path("/dev/null"),
                embedding_client=_FakeEmbeddingClient(),
            )
            await db.commit()

        async with session_factory() as db:
            version = await db.get(LearningMaterialVersion, scope.version_id)
            assert version is not None
            assert version.processing_status == "ready"

    kg_mock.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.usefixtures("_stub_extractors")
async def test_kg_enabled_calls_builder(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KNOWLEDGE_GRAPH_ENABLED", "true")
    monkeypatch.setenv("NEO4J_URI", "bolt://localhost:7687")
    monkeypatch.setenv("NEO4J_PASSWORD", "ignored-by-fake-builder")
    get_settings.cache_clear()

    kg_calls: list[dict[str, Any]] = []

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
        kg_calls.append(
            {
                "material_version_id": material_version_id,
                "pipeline_run_id": pipeline_run_id,
                "chunk_count": len(chunks),
            }
        )
        return KGSummary(concept_count=2, relationship_count=1, enabled=True)

    monkeypatch.setattr(pipeline_mod, "build_knowledge_graph_for_material_version", _fake_kg)

    async with _scope_for(engine, mime=_TEST_AUDIO_MIME, material_type="audio") as scope:
        run_id = uuid.uuid4()
        async with session_factory() as db:
            await run_material_ingest(
                db,
                scope.version_id,
                run_id,
                source_path=Path("/dev/null"),
                kg_client=AsyncMock(),
                llm_gateway=AsyncMock(),
                embedding_client=_FakeEmbeddingClient(),
            )
            await db.commit()

    assert len(kg_calls) == 1
    assert kg_calls[0]["pipeline_run_id"] == run_id
    assert kg_calls[0]["chunk_count"] >= 1


@pytest.mark.asyncio
@pytest.mark.usefixtures("_stub_extractors")
async def test_progress_updates_at_each_stage(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    async with _scope_for(engine, mime=_TEST_PDF_MIME, material_type="pdf") as scope:
        async with session_factory() as db:
            await run_material_ingest(
                db,
                scope.version_id,
                uuid.uuid4(),
                source_path=Path("/dev/null"),
                embedding_client=_FakeEmbeddingClient(),
            )
            await db.commit()

        async with session_factory() as db:
            version = await db.get(LearningMaterialVersion, scope.version_id)
            assert version is not None
            assert version.processing_status == "ready"

            job_row = (
                await db.execute(
                    text(
                        "SELECT status, progress_percent, started_at, finished_at "
                        "FROM processing_jobs WHERE entity_id = :v "
                        "ORDER BY created_at DESC LIMIT 1"
                    ),
                    {"v": scope.version_id},
                )
            ).first()
            assert job_row is not None
            assert job_row.status == "completed"
            assert job_row.progress_percent == 100
            assert job_row.started_at is not None
            assert job_row.finished_at is not None


@pytest.mark.asyncio
@pytest.mark.usefixtures("_stub_extractors")
async def test_chunks_have_denormalized_fks(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    async with _scope_for(engine, mime=_TEST_PDF_MIME, material_type="pdf") as scope:
        async with session_factory() as db:
            await run_material_ingest(
                db,
                scope.version_id,
                uuid.uuid4(),
                source_path=Path("/dev/null"),
                embedding_client=_FakeEmbeddingClient(),
            )
            await db.commit()

        async with session_factory() as db:
            rows = (
                await db.execute(
                    text(
                        "SELECT course_id, module_id, lesson_id, material_version_id "
                        "FROM document_chunks WHERE material_version_id = :v"
                    ),
                    {"v": scope.version_id},
                )
            ).all()

    assert len(rows) >= 1
    for row in rows:
        assert row.course_id == scope.course_id
        assert row.module_id == scope.module_id
        assert row.lesson_id == scope.lesson_id
        assert row.material_version_id == scope.version_id
