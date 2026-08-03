"""Integration tests for ``features.materials.routers.learner`` (T4.6).

Covers the 9 plan-mandated scenarios:

* Visibility-as-security: 200 for visible+ready, 404 for invisible / not-ready
  / draft (NOT 403 — never leak existence; plan §5075).
* Stream URL: presigned URL + ~1h expiry; 404 for invisible.
* Chunks preview: returns first N in chunk_index order; clamped to 20;
  404 for invisible.
* Auth: 401 without bearer.

S3 is NOT exercised here — :func:`create_stream_url` is monkey-patched in
the catalog service module to return a fake URL. Real S3 / Garage round-trip
lives in ``test_s3.py`` / ``test_s3_garage.py``.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest_asyncio
from alembic import command
from alembic.config import Config
from conftest import SeededUsers
from fastapi import FastAPI
from sqlalchemy import (
    Column,
    Table,
    text,
)
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

import abridgeai.features.access_control.models  # noqa: F401  -- register orgs/roles FK targets
import abridgeai.features.identity.models  # noqa: F401  -- register users / storage_objects
import abridgeai.features.interviews.models  # noqa: F401  -- T6.1 registers interview_* tables
import abridgeai.features.materials.models  # noqa: F401  -- register learning_* + document_chunks
from abridgeai.core.config import get_settings
from abridgeai.core.db import Base, get_db
from abridgeai.features.materials.routers import learner_router
from abridgeai.features.materials.services import catalog as catalog_service

for _stub_name in ("interview_configs",):
    if _stub_name not in Base.metadata.tables:
        Table(
            _stub_name,
            Base.metadata,
            Column("id", PGUUID(as_uuid=True), primary_key=True),
        )

FAKE_URL = "https://s3.test.local/abridgeai-materials/fake?X-Amz-Signature=stub"


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
    monkeypatch,
) -> AsyncIterator[FastAPI]:
    async def _override_get_db() -> AsyncIterator[AsyncSession]:
        async with session_factory() as session:
            yield session

    async def _fake_create_stream_url(_target, *, response_headers=None, settings=None):  # noqa: ANN001
        del response_headers, settings
        return FAKE_URL, datetime.now(tz=UTC) + timedelta(seconds=3600)

    monkeypatch.setattr(catalog_service, "create_stream_url", _fake_create_stream_url)

    fastapi_app = FastAPI()
    fastapi_app.include_router(learner_router, prefix="/api/v1")
    fastapi_app.dependency_overrides[get_db] = _override_get_db
    yield fastapi_app
    fastapi_app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def client(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as ac:
        yield ac


@pytest_asyncio.fixture
async def auth_session(engine: AsyncEngine, seeded_users: SeededUsers) -> AsyncIterator[uuid.UUID]:
    from abridgeai.core.security import generate_token, hash_secret

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
                "uid": seeded_users.student_id,
                "h": hash_secret(generate_token()),
                "exp": expires_at,
            },
        )
    try:
        yield session_id
    finally:
        async with engine.begin() as conn:
            await conn.execute(
                text("DELETE FROM auth_sessions WHERE id = :id"),
                {"id": session_id},
            )


@pytest_asyncio.fixture
async def student_bearer(auth_session: uuid.UUID, seeded_users: SeededUsers) -> str:
    from abridgeai.core.security import create_access_token

    return create_access_token(user_id=seeded_users.student_id, session_id=auth_session)


@pytest_asyncio.fixture
async def scenario(engine: AsyncEngine, seeded_users: SeededUsers) -> AsyncIterator[dict]:
    """Seed visible-ready, invisible-ready, processing materials + 10 chunks."""
    course_id = uuid.uuid4()
    module_id = uuid.uuid4()
    lesson_id = uuid.uuid4()
    storage_id = uuid.uuid4()

    mat_visible = uuid.uuid4()
    mat_invisible = uuid.uuid4()
    mat_processing = uuid.uuid4()

    ver_visible = uuid.uuid4()
    ver_invisible = uuid.uuid4()
    ver_processing = uuid.uuid4()

    chunk_ids = [uuid.uuid4() for _ in range(10)]

    suffix = course_id.hex[:8]

    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO courses (id, organization_id, owner_user_id, slug, title, status) "
                "VALUES (:id, :org, :owner, :slug, 'T4.6 Course', 'published')"
            ),
            {
                "id": course_id,
                "org": seeded_users.organization_id,
                "owner": seeded_users.teacher_id,
                "slug": f"t46-{suffix}",
            },
        )
        await conn.execute(
            text(
                "INSERT INTO modules (id, course_id, title, position, status) "
                "VALUES (:m, :c, 'Module', 1, 'published')"
            ),
            {"m": module_id, "c": course_id},
        )
        await conn.execute(
            text(
                "INSERT INTO lessons (id, module_id, slug, title, status) "
                "VALUES (:l, :m, 'lesson', 'Lesson', 'published')"
            ),
            {"l": lesson_id, "m": module_id},
        )
        await conn.execute(
            text(
                "INSERT INTO storage_objects (id, bucket, object_key, mime_type, size_bytes) "
                "VALUES (:id, 'test-bucket', :k, 'application/pdf', 1024)"
            ),
            {"id": storage_id, "k": f"materials/{suffix}.pdf"},
        )
        await conn.execute(
            text(
                "INSERT INTO learning_materials "
                "(id, lesson_id, title, material_type, ai_processing_enabled, visible_to_students) "
                "VALUES "
                "(:m1, :l, 'Visible Ready', 'pdf', TRUE, TRUE), "
                "(:m2, :l, 'Invisible Ready', 'pdf', TRUE, FALSE), "
                "(:m3, :l, 'Visible Processing', 'pdf', TRUE, TRUE)"
            ),
            {"m1": mat_visible, "m2": mat_invisible, "m3": mat_processing, "l": lesson_id},
        )
        await conn.execute(
            text(
                "INSERT INTO learning_material_versions "
                "(id, material_id, storage_object_id, version_no, is_current, processing_status) "
                "VALUES "
                "(:vv, :mv, :s, 1, TRUE, 'ready'), "
                "(:vi, :mi, :s, 1, TRUE, 'ready'), "
                "(:vp, :mp, :s, 1, TRUE, 'extracting')"
            ),
            {
                "vv": ver_visible,
                "vi": ver_invisible,
                "vp": ver_processing,
                "mv": mat_visible,
                "mi": mat_invisible,
                "mp": mat_processing,
                "s": storage_id,
            },
        )
        await conn.execute(
            text(
                "UPDATE learning_materials SET current_version_id = CASE id "
                "WHEN :mv THEN :vv WHEN :mi THEN :vi WHEN :mp THEN :vp END "
                "WHERE id IN (:mv, :mi, :mp)"
            ),
            {
                "mv": mat_visible,
                "mi": mat_invisible,
                "mp": mat_processing,
                "vv": ver_visible,
                "vi": ver_invisible,
                "vp": ver_processing,
            },
        )
        for idx, chunk_id in enumerate(chunk_ids):
            await conn.execute(
                text(
                    "INSERT INTO document_chunks "
                    "(id, course_id, module_id, lesson_id, material_version_id, "
                    "chunk_index, chunk_type, content, content_hash) VALUES "
                    "(:id, :c, :m, :l, :v, :i, 'pdf', :body, :hash)"
                ),
                {
                    "id": chunk_id,
                    "c": course_id,
                    "m": module_id,
                    "l": lesson_id,
                    "v": ver_visible,
                    "i": idx,
                    "body": f"chunk content {idx}",
                    "hash": f"hash-{idx:064d}"[:64],
                },
            )

    data = {
        "course_id": course_id,
        "module_id": module_id,
        "lesson_id": lesson_id,
        "storage_id": storage_id,
        "mat_visible": mat_visible,
        "mat_invisible": mat_invisible,
        "mat_processing": mat_processing,
        "ver_visible": ver_visible,
        "ver_invisible": ver_invisible,
        "ver_processing": ver_processing,
        "chunk_ids": chunk_ids,
    }
    yield data

    async with engine.begin() as conn:
        await conn.execute(
            text("DELETE FROM document_chunks WHERE id = ANY(:ids)"),
            {"ids": chunk_ids},
        )
        await conn.execute(
            text("UPDATE learning_materials SET current_version_id = NULL WHERE id = ANY(:ids)"),
            {"ids": [mat_visible, mat_invisible, mat_processing]},
        )
        await conn.execute(
            text("DELETE FROM learning_material_versions WHERE id = ANY(:ids)"),
            {"ids": [ver_visible, ver_invisible, ver_processing]},
        )
        await conn.execute(
            text("DELETE FROM learning_materials WHERE id = ANY(:ids)"),
            {"ids": [mat_visible, mat_invisible, mat_processing]},
        )
        await conn.execute(
            text("DELETE FROM storage_objects WHERE id = :id"),
            {"id": storage_id},
        )
        await conn.execute(text("DELETE FROM lessons WHERE id = :id"), {"id": lesson_id})
        await conn.execute(text("DELETE FROM modules WHERE id = :id"), {"id": module_id})
        await conn.execute(text("DELETE FROM courses WHERE id = :id"), {"id": course_id})


def test_router_metadata() -> None:
    assert learner_router.prefix == "/materials"
    paths = {(r.path, tuple(sorted(r.methods))) for r in learner_router.routes}  # type: ignore[attr-defined]
    assert ("/materials/{material_id}", ("GET",)) in paths
    assert ("/materials/{material_id}/stream-url", ("GET",)) in paths
    assert ("/materials/{material_id}/chunks/preview", ("GET",)) in paths


async def test_unauthenticated_returns_401(client: httpx.AsyncClient, scenario: dict) -> None:
    response = await client.get(f"/api/v1/materials/{scenario['mat_visible']}")
    assert response.status_code == 401
    response = await client.get(f"/api/v1/materials/{scenario['mat_visible']}/stream-url")
    assert response.status_code == 401
    response = await client.get(f"/api/v1/materials/{scenario['mat_visible']}/chunks/preview")
    assert response.status_code == 401


async def test_get_visible_material_returns_200(
    client: httpx.AsyncClient, student_bearer: str, scenario: dict
) -> None:
    response = await client.get(
        f"/api/v1/materials/{scenario['mat_visible']}",
        headers={"Authorization": f"Bearer {student_bearer}"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["id"] == str(scenario["mat_visible"])
    assert body["lesson_id"] == str(scenario["lesson_id"])
    assert body["title"] == "Visible Ready"
    assert body["material_type"] == "pdf"
    assert "processing_status" not in body


async def test_get_invisible_returns_404(
    client: httpx.AsyncClient, student_bearer: str, scenario: dict
) -> None:
    response = await client.get(
        f"/api/v1/materials/{scenario['mat_invisible']}",
        headers={"Authorization": f"Bearer {student_bearer}"},
    )
    assert response.status_code == 404


async def test_get_unready_returns_404(
    client: httpx.AsyncClient, student_bearer: str, scenario: dict
) -> None:
    response = await client.get(
        f"/api/v1/materials/{scenario['mat_processing']}",
        headers={"Authorization": f"Bearer {student_bearer}"},
    )
    assert response.status_code == 404


async def test_stream_url_returns_presigned_with_expiry(
    client: httpx.AsyncClient, student_bearer: str, scenario: dict
) -> None:
    before = datetime.now(tz=UTC)
    response = await client.get(
        f"/api/v1/materials/{scenario['mat_visible']}/stream-url",
        headers={"Authorization": f"Bearer {student_bearer}"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["url"] == FAKE_URL
    expires_at = datetime.fromisoformat(body["expires_at"])
    delta = expires_at - before
    # The route stamps expiry from settings.s3_url_ttl_seconds; asserting a
    # hardcoded hour silently depended on the developer's .env.
    configured = timedelta(seconds=get_settings().s3_url_ttl_seconds)
    assert configured - timedelta(minutes=5) <= delta <= configured + timedelta(minutes=5)


async def test_stream_url_invisible_returns_404(
    client: httpx.AsyncClient, student_bearer: str, scenario: dict
) -> None:
    response = await client.get(
        f"/api/v1/materials/{scenario['mat_invisible']}/stream-url",
        headers={"Authorization": f"Bearer {student_bearer}"},
    )
    assert response.status_code == 404


async def test_stream_url_unready_returns_404(
    client: httpx.AsyncClient, student_bearer: str, scenario: dict
) -> None:
    response = await client.get(
        f"/api/v1/materials/{scenario['mat_processing']}/stream-url",
        headers={"Authorization": f"Bearer {student_bearer}"},
    )
    assert response.status_code == 404


async def test_chunks_preview_returns_first_n(
    client: httpx.AsyncClient, student_bearer: str, scenario: dict
) -> None:
    response = await client.get(
        f"/api/v1/materials/{scenario['mat_visible']}/chunks/preview?limit=3",
        headers={"Authorization": f"Bearer {student_bearer}"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert len(body) == 3
    indices = [c["chunk_index"] for c in body]
    assert indices == [0, 1, 2]
    assert body[0]["content"] == "chunk content 0"
    assert body[0]["chunk_type"] == "pdf"


async def test_chunks_preview_default_limit_is_five(
    client: httpx.AsyncClient, student_bearer: str, scenario: dict
) -> None:
    response = await client.get(
        f"/api/v1/materials/{scenario['mat_visible']}/chunks/preview",
        headers={"Authorization": f"Bearer {student_bearer}"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert len(body) == 5
    assert [c["chunk_index"] for c in body] == [0, 1, 2, 3, 4]


async def test_chunks_preview_invisible_returns_404(
    client: httpx.AsyncClient, student_bearer: str, scenario: dict
) -> None:
    response = await client.get(
        f"/api/v1/materials/{scenario['mat_invisible']}/chunks/preview?limit=3",
        headers={"Authorization": f"Bearer {student_bearer}"},
    )
    assert response.status_code == 404


async def test_chunks_preview_limit_clamped(
    client: httpx.AsyncClient, student_bearer: str, scenario: dict
) -> None:
    response = await client.get(
        f"/api/v1/materials/{scenario['mat_visible']}/chunks/preview?limit=100",
        headers={"Authorization": f"Bearer {student_bearer}"},
    )
    assert response.status_code == 422


async def test_unknown_material_returns_404(
    client: httpx.AsyncClient, student_bearer: str, scenario: dict
) -> None:
    del scenario
    response = await client.get(
        f"/api/v1/materials/{uuid.uuid4()}",
        headers={"Authorization": f"Bearer {student_bearer}"},
    )
    assert response.status_code == 404


async def test_cross_org_material_not_readable(
    client: httpx.AsyncClient,
    engine: AsyncEngine,
    student_bearer: str,
    seeded_users: SeededUsers,
) -> None:
    """A visible+ready material from another organization is not reachable.

    Regression: the learner material reads (by id, stream-url, chunks)
    trusted ``visible_to_students=TRUE AND ready`` alone — ``user_id`` was
    discarded — so a student of org A could fetch org B's material by UUID,
    including a presigned stream URL (content exfiltration). All three
    surfaces must now resolve to 404 for a caller who is neither a member
    of the owning org nor a course manager (no existence leak).
    """
    org2 = uuid.uuid4()
    course2 = uuid.uuid4()
    module2 = uuid.uuid4()
    lesson2 = uuid.uuid4()
    storage2 = uuid.uuid4()
    material2 = uuid.uuid4()
    version2 = uuid.uuid4()
    suffix = org2.hex[:8]
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO organizations (id, slug, name, status) "
                "VALUES (:id, :slug, :name, 'active')"
            ),
            {"id": org2, "slug": f"xorg-{suffix}", "name": "X Org"},
        )
        await conn.execute(
            text(
                "INSERT INTO courses (id, organization_id, owner_user_id, slug, title, status) "
                "VALUES (:id, :org, :owner, :slug, 'X Course', 'published')"
            ),
            {
                "id": course2,
                "org": org2,
                "owner": seeded_users.teacher_id,
                "slug": f"xcourse-{suffix}",
            },
        )
        await conn.execute(
            text(
                "INSERT INTO modules (id, course_id, title, position, status) "
                "VALUES (:m, :c, 'X Module', 1, 'published')"
            ),
            {"m": module2, "c": course2},
        )
        await conn.execute(
            text(
                "INSERT INTO lessons (id, module_id, slug, title, status) "
                "VALUES (:l, :m, :slug, 'X Lesson', 'published')"
            ),
            {"l": lesson2, "m": module2, "slug": f"xlesson-{suffix}"},
        )
        await conn.execute(
            text(
                "INSERT INTO storage_objects (id, bucket, object_key) "
                "VALUES (:id, 'x-bucket', :k)"
            ),
            {"id": storage2, "k": f"x/{suffix}.pdf"},
        )
        await conn.execute(
            text(
                "INSERT INTO learning_materials (id, lesson_id, title, material_type, visible_to_students) "
                "VALUES (:id, :l, 'X Material', 'pdf', TRUE)"
            ),
            {"id": material2, "l": lesson2},
        )
        await conn.execute(
            text(
                "INSERT INTO learning_material_versions "
                "(id, material_id, storage_object_id, version_no, is_current, processing_status) "
                "VALUES (:id, :m, :s, 1, TRUE, 'ready')"
            ),
            {"id": version2, "m": material2, "s": storage2},
        )
        await conn.execute(
            text("UPDATE learning_materials SET current_version_id = :v WHERE id = :m"),
            {"v": version2, "m": material2},
        )

    headers = {"Authorization": f"Bearer {student_bearer}"}
    try:
        response = await client.get(f"/api/v1/materials/{material2}", headers=headers)
        assert response.status_code == 404, response.text
        response = await client.get(
            f"/api/v1/materials/{material2}/stream-url", headers=headers
        )
        assert response.status_code == 404, response.text
        response = await client.get(
            f"/api/v1/materials/{material2}/chunks/preview", headers=headers
        )
        assert response.status_code == 404, response.text
    finally:
        async with engine.begin() as conn:
            await conn.execute(
                text("DELETE FROM learning_material_versions WHERE id = :v"),
                {"v": version2},
            )
            await conn.execute(text("DELETE FROM learning_materials WHERE id = :m"), {"m": material2})
            await conn.execute(text("DELETE FROM storage_objects WHERE id = :s"), {"s": storage2})
            await conn.execute(text("DELETE FROM lessons WHERE id = :l"), {"l": lesson2})
            await conn.execute(text("DELETE FROM modules WHERE id = :m"), {"m": module2})
            await conn.execute(text("DELETE FROM courses WHERE id = :c"), {"c": course2})
            await conn.execute(text("DELETE FROM organizations WHERE id = :o"), {"o": org2})
