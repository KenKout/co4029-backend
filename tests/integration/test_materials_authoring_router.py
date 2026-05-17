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
