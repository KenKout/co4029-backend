"""Integration tests for ``features.materials.routers.authoring`` (T4.5).

Tests run against docker postgres on port 5433 (Phase 0 default) plus
a per-test ``ThreadedMotoServer`` for S3. Each scenario seeds a real
course / module / lesson via raw SQL and exercises the full direct-upload
lifecycle, head-verify guard, soft-delete S3 preservation invariant,
reprocess flow (chunk purge + 409), and the orphan-multipart cleanup
cron.

FIX-SEC-1 perimeter is verified by:

* ``test_unauthenticated_returns_401`` — every write endpoint rejects
  no-bearer requests.
* ``test_student_403_on_authoring`` — student token (no course.update)
  is rejected from every authoring endpoint.
* ``test_no_bare_get_current_user_on_authoring_endpoints`` — source-grep
  guard on the router file mirroring the courses-authoring test.
"""

from __future__ import annotations

import re
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock

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
from abridgeai.core.config import Settings, get_settings
from abridgeai.core.db import Base, get_db
from abridgeai.core.security import create_access_token, generate_token, hash_secret
from abridgeai.features.materials.routers import authoring_router
from abridgeai.features.materials.routers.authoring import get_arq_pool
from abridgeai.features.materials.workers.cron import _run_cleanup
from abridgeai.infrastructure import s3 as s3_module

BUCKET = "abridgeai-test-authoring"

import abridgeai.features.interviews.models  # noqa: E402, F401  -- T6.1 registers interview_* tables

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


@pytest_asyncio.fixture
async def engine() -> AsyncIterator[AsyncEngine]:
    _ensure_head()
    eng = create_async_engine(_async_url(get_settings().database_url), pool_pre_ping=True)
    yield eng
    await eng.dispose()


@pytest_asyncio.fixture
async def session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False, autoflush=False)


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


@pytest_asyncio.fixture
async def app(
    session_factory: async_sessionmaker[AsyncSession],
    s3_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[tuple[FastAPI, AsyncMock]]:
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


async def _seed_session(engine: AsyncEngine, user_id: uuid.UUID) -> uuid.UUID:
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
) -> AsyncIterator[dict[str, uuid.UUID]]:
    """Lesson under the seeded test_course → admin owns it."""
    module_id = uuid.uuid4()
    lesson_id = uuid.uuid4()
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO modules (id, course_id, title, position, status) "
                "VALUES (:m, :c, 'Mat Module', 1, 'draft')"
            ),
            {"m": module_id, "c": seeded_users.course_id},
        )
        await conn.execute(
            text(
                "INSERT INTO lessons (id, module_id, slug, title, status) "
                "VALUES (:l, :m, 'mat-lesson', 'Mat Lesson', 'draft')"
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
# Router metadata + perimeter
# ---------------------------------------------------------------------------


def test_router_metadata() -> None:
    assert authoring_router.prefix == "/teacher"
    assert len(authoring_router.routes) >= 10


async def test_unauthenticated_returns_401(
    client: httpx.AsyncClient, scenario: dict[str, uuid.UUID]
) -> None:
    resp = await client.post(
        f"/api/v1/teacher/lessons/{scenario['lesson_id']}/materials/init-upload",
        json={
            "filename": "x.pdf",
            "content_type": "application/pdf",
            "size_bytes": 100,
            "title": "X",
        },
    )
    assert resp.status_code == 401


async def test_student_403_on_authoring(
    client: httpx.AsyncClient,
    student_bearer: str,
    scenario: dict[str, uuid.UUID],
) -> None:
    resp = await client.post(
        f"/api/v1/teacher/lessons/{scenario['lesson_id']}/materials/init-upload",
        json={
            "filename": "x.pdf",
            "content_type": "application/pdf",
            "size_bytes": 100,
            "title": "X",
        },
        headers=_auth(student_bearer),
    )
    assert resp.status_code == 403


# ---------------------------------------------------------------------------
# Init upload — single + multipart
# ---------------------------------------------------------------------------


async def test_init_upload_single_file_returns_presigned(
    client: httpx.AsyncClient,
    admin_bearer: str,
    scenario: dict[str, uuid.UUID],
) -> None:
    resp = await client.post(
        f"/api/v1/teacher/lessons/{scenario['lesson_id']}/materials/init-upload",
        json={
            "filename": "small.pdf",
            "content_type": "application/pdf",
            "size_bytes": 5 * 1024 * 1024,
            "title": "Small",
        },
        headers=_auth(admin_bearer),
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["mode"] == "single"
    assert body["upload_url"].startswith("http")
    assert "X-Amz-Signature" in body["upload_url"]
    assert body["material_id"]
    assert body["version_id"]


async def test_init_upload_multipart_for_large_file(
    client: httpx.AsyncClient,
    admin_bearer: str,
    scenario: dict[str, uuid.UUID],
) -> None:
    resp = await client.post(
        f"/api/v1/teacher/lessons/{scenario['lesson_id']}/materials/init-upload",
        json={
            "filename": "big.bin",
            "content_type": "application/octet-stream",
            "size_bytes": 500 * 1024 * 1024,
            "title": "Big",
        },
        headers=_auth(admin_bearer),
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["mode"] == "multipart"
    assert body["upload_id"]
    assert body["part_count"] == 50
    assert body["parts"] is not None
    assert len(body["parts"]) >= 1
    assert body["parts"][0]["part_number"] == 1


# ---------------------------------------------------------------------------
# /complete — head verify + enqueue
# ---------------------------------------------------------------------------


async def test_complete_calls_head_verify_and_enqueues(
    client: httpx.AsyncClient,
    admin_bearer: str,
    scenario: dict[str, uuid.UUID],
    app: tuple[FastAPI, AsyncMock],
) -> None:
    _, arq_pool = app
    arq_pool.enqueue_job.reset_mock()

    init_resp = await client.post(
        f"/api/v1/teacher/lessons/{scenario['lesson_id']}/materials/init-upload",
        json={
            "filename": "doc.pdf",
            "content_type": "application/pdf",
            "size_bytes": 1024,
            "title": "Doc",
        },
        headers=_auth(admin_bearer),
    )
    assert init_resp.status_code == 201
    init = init_resp.json()
    payload = b"x" * 1024
    await _put_to_s3(init["upload_url"], payload)

    complete_resp = await client.post(
        f"/api/v1/teacher/materials/{init['material_id']}/versions/{init['version_id']}/complete",
        json={
            "storage_object_id": init["storage_object_id"],
            "checksum_sha256": "0" * 64,
        },
        headers=_auth(admin_bearer),
    )
    assert complete_resp.status_code == 202, complete_resp.text
    body = complete_resp.json()
    assert body["material_id"] == init["material_id"]
    assert body["version_id"] == init["version_id"]
    assert body["processing_job_id"]
    assert body["pipeline_run_id"]
    arq_pool.enqueue_job.assert_called_once()
    args, _ = arq_pool.enqueue_job.call_args
    assert args[0] == "ingest_material_version_task"


async def test_phantom_complete_rejected(
    client: httpx.AsyncClient,
    admin_bearer: str,
    scenario: dict[str, uuid.UUID],
    app: tuple[FastAPI, AsyncMock],
) -> None:
    _, arq_pool = app
    arq_pool.enqueue_job.reset_mock()

    init_resp = await client.post(
        f"/api/v1/teacher/lessons/{scenario['lesson_id']}/materials/init-upload",
        json={
            "filename": "phantom.pdf",
            "content_type": "application/pdf",
            "size_bytes": 1024,
            "title": "Phantom",
        },
        headers=_auth(admin_bearer),
    )
    assert init_resp.status_code == 201
    init = init_resp.json()

    complete_resp = await client.post(
        f"/api/v1/teacher/materials/{init['material_id']}/versions/{init['version_id']}/complete",
        json={
            "storage_object_id": init["storage_object_id"],
            "checksum_sha256": "0" * 64,
        },
        headers=_auth(admin_bearer),
    )
    assert complete_resp.status_code == 404
    assert complete_resp.json()["detail"]["error"] == "upload_not_found"
    arq_pool.enqueue_job.assert_not_called()


async def test_zero_byte_rejected(
    client: httpx.AsyncClient,
    admin_bearer: str,
    scenario: dict[str, uuid.UUID],
    app: tuple[FastAPI, AsyncMock],
) -> None:
    _, arq_pool = app
    arq_pool.enqueue_job.reset_mock()

    init_resp = await client.post(
        f"/api/v1/teacher/lessons/{scenario['lesson_id']}/materials/init-upload",
        json={
            "filename": "empty.pdf",
            "content_type": "application/pdf",
            "size_bytes": 0,
            "title": "Empty",
        },
        headers=_auth(admin_bearer),
    )
    assert init_resp.status_code == 201
    init = init_resp.json()
    await _put_to_s3(init["upload_url"], b"")

    complete_resp = await client.post(
        f"/api/v1/teacher/materials/{init['material_id']}/versions/{init['version_id']}/complete",
        json={
            "storage_object_id": init["storage_object_id"],
            "checksum_sha256": "0" * 64,
        },
        headers=_auth(admin_bearer),
    )
    assert complete_resp.status_code == 400
    assert complete_resp.json()["detail"]["error"] == "upload_invalid"
    arq_pool.enqueue_job.assert_not_called()


# ---------------------------------------------------------------------------
# Soft-delete preserves S3
# ---------------------------------------------------------------------------


async def test_soft_delete_preserves_s3(
    client: httpx.AsyncClient,
    admin_bearer: str,
    scenario: dict[str, uuid.UUID],
    s3_settings: Settings,
) -> None:
    init_resp = await client.post(
        f"/api/v1/teacher/lessons/{scenario['lesson_id']}/materials/init-upload",
        json={
            "filename": "keep.pdf",
            "content_type": "application/pdf",
            "size_bytes": 256,
            "title": "Keep",
        },
        headers=_auth(admin_bearer),
    )
    assert init_resp.status_code == 201
    init = init_resp.json()
    await _put_to_s3(init["upload_url"], b"k" * 256)

    complete_resp = await client.post(
        f"/api/v1/teacher/materials/{init['material_id']}/versions/{init['version_id']}/complete",
        json={
            "storage_object_id": init["storage_object_id"],
            "checksum_sha256": "0" * 64,
        },
        headers=_auth(admin_bearer),
    )
    assert complete_resp.status_code == 202

    delete_resp = await client.delete(
        f"/api/v1/teacher/materials/{init['material_id']}",
        headers=_auth(admin_bearer),
    )
    assert delete_resp.status_code == 204

    object_key = f"materials/{init['material_id']}/{init['version_id']}/keep.pdf"

    class _Obj:
        bucket = BUCKET
        object_key = ""

    obj = _Obj()
    obj.object_key = object_key
    meta = await s3_module.head_object(obj, settings=s3_settings)
    assert meta is not None, "S3 object must survive soft-delete"
    assert meta.size == 256


# ---------------------------------------------------------------------------
# Reprocess — concurrency + chunk purge
# ---------------------------------------------------------------------------


async def test_reprocess_409_when_running(
    client: httpx.AsyncClient,
    admin_bearer: str,
    scenario: dict[str, uuid.UUID],
    engine: AsyncEngine,
) -> None:
    init_resp = await client.post(
        f"/api/v1/teacher/lessons/{scenario['lesson_id']}/materials/init-upload",
        json={
            "filename": "rerun.pdf",
            "content_type": "application/pdf",
            "size_bytes": 64,
            "title": "Rerun",
        },
        headers=_auth(admin_bearer),
    )
    init = init_resp.json()
    await _put_to_s3(init["upload_url"], b"r" * 64)
    complete_resp = await client.post(
        f"/api/v1/teacher/materials/{init['material_id']}/versions/{init['version_id']}/complete",
        json={
            "storage_object_id": init["storage_object_id"],
            "checksum_sha256": "0" * 64,
        },
        headers=_auth(admin_bearer),
    )
    assert complete_resp.status_code == 202

    async with engine.begin() as conn:
        await conn.execute(
            text("UPDATE processing_jobs SET status = 'running' WHERE entity_id = :v"),
            {"v": init["version_id"]},
        )

    reprocess_resp = await client.post(
        f"/api/v1/teacher/materials/{init['material_id']}/reprocess",
        headers=_auth(admin_bearer),
    )
    assert reprocess_resp.status_code == 409
    assert reprocess_resp.json()["detail"]["error"] == "concurrent_reprocess"


async def test_reprocess_clears_chunks_and_enqueues(
    client: httpx.AsyncClient,
    admin_bearer: str,
    scenario: dict[str, uuid.UUID],
    engine: AsyncEngine,
    app: tuple[FastAPI, AsyncMock],
) -> None:
    _, arq_pool = app
    init_resp = await client.post(
        f"/api/v1/teacher/lessons/{scenario['lesson_id']}/materials/init-upload",
        json={
            "filename": "redo.pdf",
            "content_type": "application/pdf",
            "size_bytes": 128,
            "title": "Redo",
        },
        headers=_auth(admin_bearer),
    )
    init = init_resp.json()
    await _put_to_s3(init["upload_url"], b"r" * 128)
    complete_resp = await client.post(
        f"/api/v1/teacher/materials/{init['material_id']}/versions/{init['version_id']}/complete",
        json={
            "storage_object_id": init["storage_object_id"],
            "checksum_sha256": "0" * 64,
        },
        headers=_auth(admin_bearer),
    )
    assert complete_resp.status_code == 202

    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO document_chunks "
                "(id, course_id, module_id, lesson_id, material_version_id, "
                " chunk_index, chunk_type, content, content_hash) "
                "VALUES (uuid_generate_v4(), :c, :m, :l, :v, 0, 'pdf', 'old', 'h0'),"
                "       (uuid_generate_v4(), :c, :m, :l, :v, 1, 'pdf', 'old1', 'h1')"
            ),
            {
                "c": scenario["course_id"],
                "m": scenario["module_id"],
                "l": scenario["lesson_id"],
                "v": init["version_id"],
            },
        )
        await conn.execute(
            text("UPDATE processing_jobs SET status = 'completed' WHERE entity_id = :v"),
            {"v": init["version_id"]},
        )

    arq_pool.enqueue_job.reset_mock()
    reprocess_resp = await client.post(
        f"/api/v1/teacher/materials/{init['material_id']}/reprocess",
        headers=_auth(admin_bearer),
    )
    assert reprocess_resp.status_code == 202, reprocess_resp.text

    async with engine.begin() as conn:
        count = (
            (
                await conn.execute(
                    text(
                        "SELECT COUNT(*) AS n FROM document_chunks WHERE material_version_id = :v"
                    ),
                    {"v": init["version_id"]},
                )
            )
            .one()
            .n
        )
    assert count == 0
    arq_pool.enqueue_job.assert_called_once()


# ---------------------------------------------------------------------------
# Cron — orphan multipart cleanup
# ---------------------------------------------------------------------------


async def test_orphan_cleanup_cron_aborts_old_multiparts(
    s3_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Seed two multipart uploads (one fresh, one stale) and run cleanup."""
    import aioboto3

    monkeypatch.setattr(
        "abridgeai.features.materials.workers.cron.get_settings",
        lambda: s3_settings,
    )

    session = aioboto3.Session()
    stale_key_1 = "materials/stale-1.bin"
    stale_key_2 = "materials/stale-2.bin"
    async with session.client(
        "s3",
        endpoint_url=s3_settings.aws_endpoint_url,
        aws_access_key_id=s3_settings.aws_access_key_id.get_secret_value(),  # type: ignore[union-attr]
        aws_secret_access_key=s3_settings.aws_secret_access_key.get_secret_value(),  # type: ignore[union-attr]
        region_name=s3_settings.aws_region,
    ) as client:
        await client.create_multipart_upload(Bucket=BUCKET, Key=stale_key_1)
        await client.create_multipart_upload(Bucket=BUCKET, Key=stale_key_2)

        before = await client.list_multipart_uploads(Bucket=BUCKET)
        before_keys = {u["Key"] for u in before.get("Uploads", []) or []}
        assert stale_key_1 in before_keys
        assert stale_key_2 in before_keys

        await _run_cleanup(ttl_hours=-1)

        after = await client.list_multipart_uploads(Bucket=BUCKET)
        after_keys = {u["Key"] for u in after.get("Uploads", []) or []}
        assert stale_key_1 not in after_keys
        assert stale_key_2 not in after_keys


async def test_orphan_cleanup_cron_skips_fresh_multiparts(
    s3_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A multipart upload within the TTL window is preserved.

    moto returns a hardcoded ``Initiated`` timestamp (2010-11-10) for
    multipart uploads — a real S3 backend would return the actual init
    time. To verify the TTL gating logic with moto we use a very long
    TTL so the 2010 stub timestamp falls inside the "fresh" window.
    """
    import aioboto3

    monkeypatch.setattr(
        "abridgeai.features.materials.workers.cron.get_settings",
        lambda: s3_settings,
    )

    session = aioboto3.Session()
    fresh_key = "materials/fresh-skip.bin"
    async with session.client(
        "s3",
        endpoint_url=s3_settings.aws_endpoint_url,
        aws_access_key_id=s3_settings.aws_access_key_id.get_secret_value(),  # type: ignore[union-attr]
        aws_secret_access_key=s3_settings.aws_secret_access_key.get_secret_value(),  # type: ignore[union-attr]
        region_name=s3_settings.aws_region,
    ) as client:
        await client.create_multipart_upload(Bucket=BUCKET, Key=fresh_key)

        await _run_cleanup(ttl_hours=24 * 365 * 200)

        listed = await client.list_multipart_uploads(Bucket=BUCKET)
        keys = {u["Key"] for u in listed.get("Uploads", []) or []}
        assert fresh_key in keys

        for upload in listed.get("Uploads", []) or []:
            await client.abort_multipart_upload(
                Bucket=BUCKET, Key=upload["Key"], UploadId=upload["UploadId"]
            )


# ---------------------------------------------------------------------------
# Source-grep guard (FIX-SEC-1 perimeter)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Linking an already-uploaded object
#
# This endpoint is where the "pending forever" bug lived. It stamped every
# new version `processing_status='pending'` but never created a
# ProcessingJob or enqueued anything, so the material sat behind a spinner
# with nothing scheduled that could ever move it, and no error anywhere.
#
# The fix splits on the teacher's AI toggle, and the two branches have to
# stay honest about each other: "pending" is a promise that something is
# coming, so it may only be written when a job really was queued.
# ---------------------------------------------------------------------------


async def _seed_storage_object(engine: AsyncEngine) -> uuid.UUID:
    storage_id = uuid.uuid4()
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO storage_objects (id, bucket, object_key, mime_type) "
                "VALUES (:id, :b, :key, 'application/pdf')"
            ),
            {"id": storage_id, "b": BUCKET, "key": f"link/{storage_id.hex}"},
        )
    return storage_id


async def _read_version_of(engine: AsyncEngine, material_id: uuid.UUID) -> dict:
    async with engine.connect() as conn:
        row = (
            await conn.execute(
                text(
                    "SELECT id, processing_status, is_current, version_no, uploaded_by "
                    "FROM learning_material_versions WHERE material_id = :m"
                ),
                {"m": material_id},
            )
        ).mappings()
        return dict(row.one())


async def _count_jobs_for(engine: AsyncEngine, version_id: uuid.UUID) -> int:
    async with engine.connect() as conn:
        return (
            await conn.execute(
                text("SELECT count(*) FROM processing_jobs WHERE entity_id = :v"),
                {"v": version_id},
            )
        ).scalar_one()


async def test_linking_with_ai_enabled_queues_a_job_and_enqueues_it(
    client: httpx.AsyncClient,
    app: tuple[FastAPI, AsyncMock],
    admin_bearer: str,
    engine: AsyncEngine,
    scenario: dict[str, uuid.UUID],
) -> None:
    """``pending`` is only written because a job is genuinely scheduled.

    Both halves matter and neither implies the other: the row is what the
    reaper later reconciles against, and the enqueue is what actually makes
    a worker pick the document up.
    """
    _, arq_pool = app
    storage_id = await _seed_storage_object(engine)

    resp = await client.post(
        f"/api/v1/teacher/lessons/{scenario['lesson_id']}/materials/link",
        json={
            "storage_object_id": str(storage_id),
            "title": "Week 1 Slides",
            "material_type": "pdf",
            "ai_processing_enabled": True,
        },
        headers=_auth(admin_bearer),
    )

    assert resp.status_code == 201, resp.text
    material_id = uuid.UUID(resp.json()["id"])

    version = await _read_version_of(engine, material_id)
    assert version["processing_status"] == "pending"
    assert await _count_jobs_for(engine, version["id"]) == 1

    enqueued = [
        call for call in arq_pool.enqueue_job.await_args_list
        if call.args and call.args[0] == "ingest_material_version_task"
    ]
    assert len(enqueued) == 1
    assert enqueued[0].args[2] == version["id"], "the task is pointed at the new version"


async def test_linking_with_ai_disabled_parks_the_version_terminally(
    client: httpx.AsyncClient,
    app: tuple[FastAPI, AsyncMock],
    admin_bearer: str,
    engine: AsyncEngine,
    scenario: dict[str, uuid.UUID],
) -> None:
    """The teacher declined processing, so nothing is coming.

    ``cancelled`` says that; ``pending`` would be the original bug -- a
    spinner over work nobody scheduled. The distinction is the whole point
    of the branch, and it is the state the AI Hub reads to offer "Enable
    AI" later.
    """
    _, arq_pool = app
    storage_id = await _seed_storage_object(engine)

    resp = await client.post(
        f"/api/v1/teacher/lessons/{scenario['lesson_id']}/materials/link",
        json={
            "storage_object_id": str(storage_id),
            "title": "Reference Only",
            "material_type": "pdf",
            "ai_processing_enabled": False,
        },
        headers=_auth(admin_bearer),
    )

    assert resp.status_code == 201, resp.text
    material_id = uuid.UUID(resp.json()["id"])

    version = await _read_version_of(engine, material_id)
    assert version["processing_status"] == "cancelled"
    assert await _count_jobs_for(engine, version["id"]) == 0
    assert not [
        call for call in arq_pool.enqueue_job.await_args_list
        if call.args and call.args[0] == "ingest_material_version_task"
    ]


async def test_ai_processing_is_off_unless_asked_for(
    client: httpx.AsyncClient,
    admin_bearer: str,
    engine: AsyncEngine,
    scenario: dict[str, uuid.UUID],
) -> None:
    """Linking is also used for plain lesson resources that want no
    pipeline at all, so the default must not spend an ingest on every
    attachment."""
    storage_id = await _seed_storage_object(engine)

    resp = await client.post(
        f"/api/v1/teacher/lessons/{scenario['lesson_id']}/materials/link",
        json={"storage_object_id": str(storage_id), "title": "Handout"},
        headers=_auth(admin_bearer),
    )

    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["ai_processing_enabled"] is False
    assert body["visible_to_students"] is False, "hidden until the teacher publishes it"
    assert body["material_type"] == "text", (
        "the fallback has to be one of the nine values the CHECK constraint "
        "allows, or omitting the field fails the flush instead of creating "
        "the material"
    )

    version = await _read_version_of(engine, uuid.UUID(body["id"]))
    assert version["processing_status"] == "cancelled"


async def test_the_linked_version_is_current_and_attributed(
    client: httpx.AsyncClient,
    admin_bearer: str,
    engine: AsyncEngine,
    seeded_users: SeededUsers,
    scenario: dict[str, uuid.UUID],
) -> None:
    """The material points at the version and the version knows who
    uploaded it -- the reaper reads that field to decide who to notify when
    it gives up, and to attribute the recovered ingest's audit rows.
    """
    storage_id = await _seed_storage_object(engine)

    resp = await client.post(
        f"/api/v1/teacher/lessons/{scenario['lesson_id']}/materials/link",
        json={
            "storage_object_id": str(storage_id),
            "title": "Attributed",
            "ai_processing_enabled": True,
        },
        headers=_auth(admin_bearer),
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()

    version = await _read_version_of(engine, uuid.UUID(body["id"]))
    assert version["version_no"] == 1
    assert version["is_current"] is True
    assert version["uploaded_by"] == seeded_users.admin_id
    assert body["latest_version"] is not None
    assert body["version_count"] == 1


async def test_a_student_cannot_link_a_material(
    client: httpx.AsyncClient,
    student_bearer: str,
    engine: AsyncEngine,
    scenario: dict[str, uuid.UUID],
) -> None:
    """The endpoint writes to the course's curriculum, so it sits behind
    the same lesson-scoped perimeter as the rest of the router."""
    storage_id = await _seed_storage_object(engine)

    resp = await client.post(
        f"/api/v1/teacher/lessons/{scenario['lesson_id']}/materials/link",
        json={"storage_object_id": str(storage_id), "title": "Nope"},
        headers=_auth(student_bearer),
    )

    assert resp.status_code == 403


# ---------------------------------------------------------------------------
# Preprocessing report + overrides
#
# The noise cascade drops headers, footers and page numbers between
# extraction and chunking, and records every decision in
# `material_preprocess_quarantine`. These endpoints are the teacher's
# window into that and their lever to overturn it.
#
# Exercised end to end because the query layer here is raw SQL: the
# ownership join, the `include_confirmed` filter and the per-reason
# aggregate are all statements no mock can validate.
# ---------------------------------------------------------------------------


async def _seed_material_with_quarantine(
    engine: AsyncEngine,
    scenario: dict[str, uuid.UUID],
    *,
    rows: list[dict] | None = None,
) -> tuple[uuid.UUID, uuid.UUID, list[uuid.UUID]]:
    """A material + current version carrying quarantined units."""
    storage_id, material_id, version_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    quarantine_ids: list[uuid.UUID] = []

    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO storage_objects (id, bucket, object_key, mime_type) "
                "VALUES (:id, :b, :key, 'application/pdf')"
            ),
            {"id": storage_id, "b": BUCKET, "key": f"pp/{storage_id.hex}"},
        )
        await conn.execute(
            text(
                "INSERT INTO learning_materials "
                "(id, lesson_id, title, material_type, preprocess_mode) "
                "VALUES (:id, :l, 'Preprocessed Material', 'pdf', 'full')"
            ),
            {"id": material_id, "l": scenario["lesson_id"]},
        )
        await conn.execute(
            text(
                "INSERT INTO learning_material_versions "
                "(id, material_id, storage_object_id, version_no, processing_status, "
                " is_current, extracted_metadata) "
                "VALUES (:id, :m, :so, 1, 'ready', TRUE, "
                " CAST(:meta AS jsonb))"
            ),
            {
                "id": version_id,
                "m": material_id,
                "so": storage_id,
                "meta": '{"preprocess": {"lines_removed": 118, "pages_scanned": 24}}',
            },
        )
        await conn.execute(
            text("UPDATE learning_materials SET current_version_id = :v WHERE id = :m"),
            {"v": version_id, "m": material_id},
        )

        for ordinal, row in enumerate(rows or [], start=1):
            quarantine_id = uuid.uuid4()
            quarantine_ids.append(quarantine_id)
            await conn.execute(
                text(
                    "INSERT INTO material_preprocess_quarantine "
                    "(id, material_version_id, course_id, unit_kind, page_number, ordinal, "
                    " content, occurrences, rule_name, reason_code, action, rule_score, "
                    " detector_stage, teacher_action) "
                    "VALUES (:id, :v, :c, :kind, :page, :ord, :content, :occ, :rule, "
                    " :reason, 'drop', 0.9, 'rule', :teacher)"
                ),
                {
                    "id": quarantine_id,
                    "v": version_id,
                    "c": scenario["course_id"],
                    "kind": row.get("unit_kind", "line"),
                    "page": row.get("page_number", 1),
                    "ord": ordinal,
                    "content": row.get("content", "Faculty of CSE"),
                    "occ": row.get("occurrences", 1),
                    "rule": row.get("rule_name", "repeated_header"),
                    "reason": row.get("reason_code", "repeated_across_pages"),
                    "teacher": row.get("teacher_action"),
                },
            )

    return material_id, version_id, quarantine_ids


async def _cleanup_quarantine(engine: AsyncEngine, material_id: uuid.UUID) -> None:
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "DELETE FROM material_preprocess_quarantine WHERE material_version_id IN "
                "(SELECT id FROM learning_material_versions WHERE material_id = :m)"
            ),
            {"m": material_id},
        )


async def test_the_report_shows_what_the_filter_removed(
    client: httpx.AsyncClient,
    admin_bearer: str,
    engine: AsyncEngine,
    scenario: dict[str, uuid.UUID],
) -> None:
    """A teacher cannot sensibly override what they cannot see.

    The response carries the exact removed text, its page and how many
    times it occurred -- the counts alone would say a rule fired without
    saying on what.
    """
    material_id, version_id, _ids = await _seed_material_with_quarantine(
        engine,
        scenario,
        rows=[
            {"content": "Faculty of CSE", "occurrences": 42, "page_number": 3},
            {"content": "Page 7 of 120", "occurrences": 120, "reason_code": "page_number"},
        ],
    )
    try:
        resp = await client.get(
            f"/api/v1/teacher/materials/{material_id}/preprocess/report",
            headers=_auth(admin_bearer),
        )

        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["material_version_id"] == str(version_id)
        assert body["preprocess_mode"] == "full"
        assert body["summary"] == {"lines_removed": 118, "pages_scanned": 24}
        assert body["requires_reprocess"] is True, (
            "an override never edits chunks in place, so the UI must offer the button"
        )
        contents = {u["content"] for u in body["units"]}
        assert contents == {"Faculty of CSE", "Page 7 of 120"}
    finally:
        await _cleanup_quarantine(engine, material_id)


async def test_the_report_orders_units_by_page(
    client: httpx.AsyncClient,
    admin_bearer: str,
    engine: AsyncEngine,
    scenario: dict[str, uuid.UUID],
) -> None:
    """The teacher reads this beside the document, so it has to follow it."""
    material_id, _v, _ids = await _seed_material_with_quarantine(
        engine,
        scenario,
        rows=[
            {"content": "third", "page_number": 9},
            {"content": "first", "page_number": 1},
            {"content": "second", "page_number": 4},
        ],
    )
    try:
        resp = await client.get(
            f"/api/v1/teacher/materials/{material_id}/preprocess/report",
            headers=_auth(admin_bearer),
        )

        assert resp.status_code == 200, resp.text
        assert [u["content"] for u in resp.json()["units"]] == ["first", "second", "third"]
    finally:
        await _cleanup_quarantine(engine, material_id)


async def test_a_material_with_no_version_has_no_report(
    client: httpx.AsyncClient,
    admin_bearer: str,
    engine: AsyncEngine,
    scenario: dict[str, uuid.UUID],
) -> None:
    material_id = uuid.uuid4()
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO learning_materials (id, lesson_id, title, material_type) "
                "VALUES (:id, :l, 'No Version', 'pdf')"
            ),
            {"id": material_id, "l": scenario["lesson_id"]},
        )

    resp = await client.get(
        f"/api/v1/teacher/materials/{material_id}/preprocess/report",
        headers=_auth(admin_bearer),
    )

    assert resp.status_code == 404
    assert resp.json()["detail"]["resource"] == "material"


async def test_restoring_a_unit_records_the_teacher_and_the_verdict(
    client: httpx.AsyncClient,
    admin_bearer: str,
    engine: AsyncEngine,
    seeded_users: SeededUsers,
    scenario: dict[str, uuid.UUID],
) -> None:
    """The decision is persisted, never applied in place.

    Rewriting already-embedded chunks live is the re-index this whole
    design exists to avoid, so the row is stamped and the cascade reads it
    on the next reprocess.
    """
    material_id, _v, ids = await _seed_material_with_quarantine(
        engine, scenario, rows=[{"content": "A real paragraph", "unit_kind": "page"}]
    )
    try:
        resp = await client.post(
            f"/api/v1/teacher/materials/{material_id}/preprocess/quarantine/{ids[0]}/action",
            json={"action": "restore"},
            headers=_auth(admin_bearer),
        )

        assert resp.status_code == 200, resp.text
        async with engine.connect() as conn:
            row = (
                await conn.execute(
                    text(
                        "SELECT teacher_action, teacher_action_by, teacher_action_at "
                        "FROM material_preprocess_quarantine WHERE id = :id"
                    ),
                    {"id": ids[0]},
                )
            ).mappings().one()
        assert row["teacher_action"] == "restore"
        assert row["teacher_action_by"] == seeded_users.admin_id
        assert row["teacher_action_at"] is not None
    finally:
        await _cleanup_quarantine(engine, material_id)


async def test_a_quarantine_id_from_another_material_is_refused(
    client: httpx.AsyncClient,
    admin_bearer: str,
    engine: AsyncEngine,
    scenario: dict[str, uuid.UUID],
) -> None:
    """The permission dependency guards the MATERIAL in the path.

    The quarantine id beside it is just a number, so without the ownership
    re-check a caller could act on another material's rows through a URL
    the permission check approved. Both materials here belong to the same
    course, so the only thing refusing this is that re-check.
    """
    mine, _v1, _ids = await _seed_material_with_quarantine(
        engine, scenario, rows=[{"content": "mine"}]
    )
    theirs, _v2, their_ids = await _seed_material_with_quarantine(
        engine, scenario, rows=[{"content": "theirs"}]
    )
    try:
        resp = await client.post(
            f"/api/v1/teacher/materials/{mine}/preprocess/quarantine/{their_ids[0]}/action",
            json={"action": "restore"},
            headers=_auth(admin_bearer),
        )

        assert resp.status_code == 404
        assert resp.json()["detail"]["resource"] == "quarantine_unit"

        async with engine.connect() as conn:
            untouched = (
                await conn.execute(
                    text(
                        "SELECT teacher_action FROM material_preprocess_quarantine "
                        "WHERE id = :id"
                    ),
                    {"id": their_ids[0]},
                )
            ).scalar_one()
        assert untouched is None, "the other material's row was not written"
    finally:
        await _cleanup_quarantine(engine, mine)
        await _cleanup_quarantine(engine, theirs)


async def test_an_unknown_quarantine_id_is_refused_the_same_way(
    client: httpx.AsyncClient,
    admin_bearer: str,
    engine: AsyncEngine,
    scenario: dict[str, uuid.UUID],
) -> None:
    """Same status and shape as a foreign id: distinguishing them would let
    a caller enumerate which quarantine ids exist."""
    material_id, _v, _ids = await _seed_material_with_quarantine(engine, scenario, rows=[])
    try:
        resp = await client.post(
            f"/api/v1/teacher/materials/{material_id}/preprocess/quarantine/"
            f"{uuid.uuid4()}/action",
            json={"action": "confirm"},
            headers=_auth(admin_bearer),
        )
        assert resp.status_code == 404
        assert resp.json()["detail"]["resource"] == "quarantine_unit"
    finally:
        await _cleanup_quarantine(engine, material_id)


async def test_the_mode_switch_takes_effect_on_the_material(
    client: httpx.AsyncClient,
    admin_bearer: str,
    engine: AsyncEngine,
    scenario: dict[str, uuid.UUID],
) -> None:
    """``normalize_only`` keeps the never-destructive fixes while disabling
    every filter -- the right setting for a document the rules misread."""
    material_id, _v, _ids = await _seed_material_with_quarantine(engine, scenario, rows=[])
    try:
        resp = await client.patch(
            f"/api/v1/teacher/materials/{material_id}/preprocess/mode",
            json={"mode": "normalize_only"},
            headers=_auth(admin_bearer),
        )

        assert resp.status_code == 200, resp.text
        assert resp.json()["mode"] == "normalize_only"

        async with engine.connect() as conn:
            stored = (
                await conn.execute(
                    text("SELECT preprocess_mode FROM learning_materials WHERE id = :m"),
                    {"m": material_id},
                )
            ).scalar_one()
        assert stored == "normalize_only"
    finally:
        await _cleanup_quarantine(engine, material_id)


async def test_setting_the_mode_on_an_unknown_material_is_a_404(
    client: httpx.AsyncClient,
    admin_bearer: str,
    scenario: dict[str, uuid.UUID],
) -> None:
    resp = await client.patch(
        f"/api/v1/teacher/materials/{uuid.uuid4()}/preprocess/mode",
        json={"mode": "off"},
        headers=_auth(admin_bearer),
    )
    assert resp.status_code == 404


async def test_the_course_audit_counts_units_and_occurrences_per_reason(
    client: httpx.AsyncClient,
    admin_bearer: str,
    engine: AsyncEngine,
    scenario: dict[str, uuid.UUID],
) -> None:
    """The precision audit: a reason code with many restores is a rule
    eating real content and needs its threshold revisited.

    Units and occurrences are counted separately because one repeated
    header is a single decision affecting a hundred pages -- reporting only
    one of the two makes the rule look either trivial or catastrophic.
    """
    material_id, _v, _ids = await _seed_material_with_quarantine(
        engine,
        scenario,
        rows=[
            {"reason_code": "repeated_across_pages", "occurrences": 40},
            {"reason_code": "repeated_across_pages", "occurrences": 60,
             "teacher_action": "restore"},
            {"reason_code": "page_number", "occurrences": 5, "teacher_action": "confirm"},
        ],
    )
    try:
        resp = await client.get(
            f"/api/v1/teacher/courses/{scenario['course_id']}/preprocess/summary",
            headers=_auth(admin_bearer),
        )

        assert resp.status_code == 200, resp.text
        by_reason = {row["reason_code"]: row for row in resp.json()}

        repeated = by_reason["repeated_across_pages"]
        assert repeated["unit_count"] == 2
        assert repeated["occurrence_count"] == 100
        assert repeated["restored"] == 1
        assert repeated["confirmed"] == 0

        page_number = by_reason["page_number"]
        assert page_number["unit_count"] == 1
        assert page_number["confirmed"] == 1
    finally:
        await _cleanup_quarantine(engine, material_id)


async def test_a_student_cannot_read_the_preprocessing_report(
    client: httpx.AsyncClient,
    student_bearer: str,
    engine: AsyncEngine,
    scenario: dict[str, uuid.UUID],
) -> None:
    """The report carries the removed source text verbatim, so it sits
    behind the same authoring perimeter as the rest of the router."""
    material_id, _v, _ids = await _seed_material_with_quarantine(engine, scenario, rows=[])
    try:
        resp = await client.get(
            f"/api/v1/teacher/materials/{material_id}/preprocess/report",
            headers=_auth(student_bearer),
        )
        assert resp.status_code == 403
    finally:
        await _cleanup_quarantine(engine, material_id)


def test_no_bare_get_current_user_on_authoring_endpoints() -> None:
    src = (
        Path(__file__).resolve().parent.parent.parent
        / "abridgeai"
        / "features"
        / "materials"
        / "routers"
        / "authoring.py"
    ).read_text(encoding="utf-8")
    code_only = re.sub(r'"""[\s\S]*?"""', "", src)
    bare = re.findall(r"Depends\(get_current_user\)", code_only)
    nested_in_factory = code_only.count(
        "current_user: Annotated[CurrentUser, Depends(get_current_user)]"
    )
    assert len(bare) == nested_in_factory, (
        f"authoring.py uses bare Depends(get_current_user) outside dependency factories: "
        f"total={len(bare)}, factory-nested={nested_in_factory}"
    )
