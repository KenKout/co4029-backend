"""Materials ARQ worker tests (T4.7).

The worker's responsibilities (the T4.4 pipeline owns the rest):

* Set ``current_actor_var`` from ``actor_id`` so audit columns get filled
  via the SQLAlchemy ``before_flush`` listener.
* Pull source bytes from S3 directly (``download_to_temp``) rather than
  via backend HTTP — preserves the bandwidth-saving architecture.
* Wrap pipeline invocation in a ``TemporaryDirectory`` so all temp files
  vanish when the task ends, even on failure.
* Commit whether the pipeline succeeded or raised, so any
  ``_capture_failure`` audit rows survive the retry boundary.
"""

from __future__ import annotations

import inspect
import tempfile
import uuid
from collections.abc import AsyncIterator
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
from abridgeai.ai.models import ProcessingJob
from abridgeai.core.config import get_settings
from abridgeai.features.materials.models import LearningMaterialVersion
from abridgeai.features.materials.workers import ingest as worker_mod
from abridgeai.features.materials.workers import ingest_material_version_task
from abridgeai.workers.arq_app import WorkerSettings


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


@pytest.fixture(autouse=True)
def _reset_settings_cache() -> AsyncIterator[None]:
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


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


class _Scope:
    __slots__ = (
        "course_id",
        "lesson_id",
        "material_id",
        "module_id",
        "org_id",
        "owner_id",
        "storage_id",
        "version_id",
    )

    def __init__(self) -> None:
        self.org_id: UUID = uuid.uuid4()
        self.owner_id: UUID = uuid.uuid4()
        self.course_id: UUID = uuid.uuid4()
        self.module_id: UUID = uuid.uuid4()
        self.lesson_id: UUID = uuid.uuid4()
        self.storage_id: UUID = uuid.uuid4()
        self.material_id: UUID = uuid.uuid4()
        self.version_id: UUID = uuid.uuid4()


async def _seed(engine: AsyncEngine) -> _Scope:
    scope = _Scope()
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO organizations (id, slug, name, status) VALUES (:id, :s, :n, 'active')"
            ),
            {"id": scope.org_id, "s": f"wkr-{scope.org_id.hex[:8]}", "n": "Wkr Org"},
        )
        await conn.execute(
            text("INSERT INTO users (id, primary_email, status) VALUES (:id, :e, 'active')"),
            {"id": scope.owner_id, "e": f"wkr-{scope.owner_id.hex[:8]}@test.local"},
        )
        await conn.execute(
            text(
                "INSERT INTO courses (id, organization_id, owner_user_id, slug, title, status) "
                "VALUES (:id, :o, :u, :s, 'Wkr Course', 'draft')"
            ),
            {
                "id": scope.course_id,
                "o": scope.org_id,
                "u": scope.owner_id,
                "s": f"wkr-c-{scope.course_id.hex[:6]}",
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
                "s": f"wkr-l-{scope.lesson_id.hex[:6]}",
            },
        )
        await conn.execute(
            text(
                "INSERT INTO storage_objects (id, bucket, object_key, mime_type) "
                "VALUES (:id, 'test-bucket', :key, 'application/pdf')"
            ),
            {"id": scope.storage_id, "key": f"wkr/{scope.storage_id.hex}"},
        )
        await conn.execute(
            text(
                "INSERT INTO learning_materials (id, lesson_id, title, material_type) "
                "VALUES (:id, :l, 'Test Mat', 'pdf')"
            ),
            {"id": scope.material_id, "l": scope.lesson_id},
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


async def _teardown(engine: AsyncEngine, scope: _Scope) -> None:
    async with engine.begin() as conn:
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


@pytest_asyncio.fixture
async def scope(engine: AsyncEngine) -> AsyncIterator[_Scope]:
    seeded = await _seed(engine)
    try:
        yield seeded
    finally:
        await _teardown(engine, seeded)


def test_actor_id_is_first_arg_after_ctx() -> None:
    """Project convention (plan §5108 / T0.8): every worker task has
    ``ctx`` followed by ``actor_id`` as the first two parameters.
    """
    sig = inspect.signature(ingest_material_version_task)
    params = list(sig.parameters)
    assert params[0] == "ctx", f"first arg must be ctx, got {params[0]!r}"
    assert params[1] == "actor_id", f"second arg must be actor_id, got {params[1]!r}"


def test_worker_settings_aggregates_material_jobs() -> None:
    assert ingest_material_version_task in WorkerSettings.functions


@pytest.mark.asyncio
async def test_actor_id_propagates_to_audit(
    engine: AsyncEngine,
    scope: _Scope,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    actor_id = uuid.uuid4()
    async with engine.begin() as conn:
        await conn.execute(
            text("INSERT INTO users (id, primary_email, status) VALUES (:id, :e, 'active')"),
            {"id": actor_id, "e": f"actor-{actor_id.hex[:8]}@test.local"},
        )

    fake_local = tmp_path / "downloaded.bin"
    fake_local.write_bytes(b"stub")

    async def _fake_download(storage_object: Any, dest_dir: Path, **_: Any) -> Path:
        return fake_local

    async def _fake_pipeline(
        db: AsyncSession,
        material_version_id: UUID,
        pipeline_run_id: UUID,
        *,
        source_path: Path | None = None,
        **_: Any,
    ) -> None:
        version = await db.get(LearningMaterialVersion, material_version_id)
        assert version is not None
        version.processing_status = "ready"
        await db.flush()

    monkeypatch.setattr(worker_mod, "download_to_temp", _fake_download)
    monkeypatch.setattr(worker_mod, "run_material_ingest", _fake_pipeline)

    try:
        await ingest_material_version_task(
            ctx={},
            actor_id=actor_id,
            material_version_id=scope.version_id,
            pipeline_run_id=uuid.uuid4(),
        )

        async with engine.connect() as conn:
            row = (
                await conn.execute(
                    text(
                        "SELECT updated_by, processing_status FROM learning_material_versions "
                        "WHERE id = :id"
                    ),
                    {"id": scope.version_id},
                )
            ).one()
        assert row.updated_by == actor_id
        assert row.processing_status == "ready"
    finally:
        async with engine.begin() as conn:
            await conn.execute(text("DELETE FROM users WHERE id = :id"), {"id": actor_id})


@pytest.mark.asyncio
async def test_worker_pulls_s3_direct_not_via_backend_http(
    scope: _Scope,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_local = tmp_path / "downloaded.bin"
    fake_local.write_bytes(b"stub")

    download_calls: list[dict[str, Any]] = []

    async def _capturing_download(storage_object: Any, dest_dir: Path, **_: Any) -> Path:
        download_calls.append(
            {
                "bucket": storage_object.bucket,
                "object_key": storage_object.object_key,
                "dest_dir": dest_dir,
            }
        )
        return fake_local

    pipeline_mock = AsyncMock()
    monkeypatch.setattr(worker_mod, "download_to_temp", _capturing_download)
    monkeypatch.setattr(worker_mod, "run_material_ingest", pipeline_mock)

    await ingest_material_version_task(
        ctx={},
        actor_id=uuid.uuid4(),
        material_version_id=scope.version_id,
        pipeline_run_id=uuid.uuid4(),
    )

    assert len(download_calls) == 1
    assert download_calls[0]["bucket"] == "test-bucket"
    assert download_calls[0]["object_key"].startswith("wkr/")
    assert pipeline_mock.await_count == 1


@pytest.mark.asyncio
async def test_temp_cleanup_after_task(
    scope: _Scope,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_local = tmp_path / "downloaded.bin"
    fake_local.write_bytes(b"stub")

    async def _fake_download(storage_object: Any, dest_dir: Path, **_: Any) -> Path:
        return fake_local

    async def _fake_pipeline(*_: Any, **__: Any) -> None:
        return None

    monkeypatch.setattr(worker_mod, "download_to_temp", _fake_download)
    monkeypatch.setattr(worker_mod, "run_material_ingest", _fake_pipeline)

    tmp_root = Path(tempfile.gettempdir())
    before = {p.name for p in tmp_root.glob("abridgeai-worker-*")}  # noqa: ASYNC240  -- read-only listing of /tmp; no IO blocking concern

    await ingest_material_version_task(
        ctx={},
        actor_id=uuid.uuid4(),
        material_version_id=scope.version_id,
        pipeline_run_id=uuid.uuid4(),
    )

    after = {p.name for p in tmp_root.glob("abridgeai-worker-*")}  # noqa: ASYNC240  -- read-only listing of /tmp; no IO blocking concern
    assert before == after, f"orphaned temp dirs: {after - before}"


@pytest.mark.asyncio
async def test_failure_captures_exception_in_processing_jobs(
    engine: AsyncEngine,
    scope: _Scope,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_local = tmp_path / "downloaded.bin"
    fake_local.write_bytes(b"stub")

    async def _fake_download(storage_object: Any, dest_dir: Path, **_: Any) -> Path:
        return fake_local

    async def _failing_pipeline(
        db: AsyncSession,
        material_version_id: UUID,
        pipeline_run_id: UUID,
        *,
        source_path: Path | None = None,
        **_: Any,
    ) -> None:
        job = ProcessingJob(
            entity_type="material_version",
            entity_id=material_version_id,
            job_type="full_pipeline",
            status="failed",
            error_message="extractor blew up: synthetic",
        )
        db.add(job)
        await db.flush()
        raise RuntimeError("extractor blew up: synthetic")

    monkeypatch.setattr(worker_mod, "download_to_temp", _fake_download)
    monkeypatch.setattr(worker_mod, "run_material_ingest", _failing_pipeline)

    with pytest.raises(RuntimeError, match="extractor blew up"):
        await ingest_material_version_task(
            ctx={},
            actor_id=uuid.uuid4(),
            material_version_id=scope.version_id,
            pipeline_run_id=uuid.uuid4(),
        )

    async with engine.connect() as conn:
        row = (
            await conn.execute(
                text(
                    "SELECT status, error_message FROM processing_jobs "
                    "WHERE entity_id = :v ORDER BY created_at DESC LIMIT 1"
                ),
                {"v": scope.version_id},
            )
        ).first()
    assert row is not None, "failure-state job row was not committed"
    assert row.status == "failed"
    assert row.error_message is not None
    assert "synthetic" in row.error_message


@pytest.mark.asyncio
async def test_missing_version_returns_without_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline_mock = AsyncMock(side_effect=AssertionError("pipeline must not run"))
    download_mock = AsyncMock(side_effect=AssertionError("download must not run"))
    monkeypatch.setattr(worker_mod, "download_to_temp", download_mock)
    monkeypatch.setattr(worker_mod, "run_material_ingest", pipeline_mock)

    await ingest_material_version_task(
        ctx={},
        actor_id=uuid.uuid4(),
        material_version_id=uuid.uuid4(),
        pipeline_run_id=uuid.uuid4(),
    )

    pipeline_mock.assert_not_awaited()
    download_mock.assert_not_awaited()
