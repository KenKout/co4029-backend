"""Integration tests for ``POST /teacher/courses/import-syllabus``.

Scope: the endpoint's contract around the parser, not the parser itself
(``tests/unit/courses/test_syllabus_parser.py`` covers the field rules
without a database).

What matters here:

* **Both permissions are enforced.** The import writes learning outcomes,
  so it is gated on ``learning_outcome.manage`` on TOP of ``course.create``.
  A teacher holding only ``course.create`` must not get outcome authoring
  through the importer's side door.
* **A failed import still leaves a record.** The manager's failure
  notification points at a ``course_syllabus_imports`` row, so the row has
  to survive the request that raised — and no half-built course may be
  left behind.
* **A successful import is one transaction**: course + outcomes + storage
  row + attempt row, all or nothing.

``put_object_bytes`` is monkeypatched: object storage is not what these
tests are about, and a live bucket would make them fail for reasons that
have nothing to do with the import.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import httpx
import pytest
import pytest_asyncio
from conftest import SeededUsers
from fastapi import FastAPI
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

import abridgeai.features.access_control.models  # noqa: F401  -- register FK targets
import abridgeai.features.identity.models  # noqa: F401  -- users / storage_objects FK targets
import abridgeai.features.interviews.models  # noqa: F401  -- module_items FK target
import abridgeai.features.materials.models  # noqa: F401  -- lessons FK target
import abridgeai.features.quizzes.models  # noqa: F401  -- module_items FK target
from abridgeai.core.config import get_settings
from abridgeai.core.db import get_db
from abridgeai.core.security import create_access_token, generate_token, hash_secret
from abridgeai.features.courses.routers.authoring import router as authoring_router

pytestmark = pytest.mark.asyncio

_ENDPOINT = "/api/v1/teacher/courses/import-syllabus"

# The smallest input that gets PAST the upload validator (magic bytes + MIME)
# and fails inside the parser instead — which is the failure path we want:
# a real PDF that simply is not a syllabus.
_NOT_A_SYLLABUS_PDF = b"%PDF-1.4\n% minimal, no text layer\n"


def _async_url(database_url: str) -> str:
    if "+psycopg_async" in database_url:
        return database_url
    if database_url.startswith("postgresql+psycopg://"):
        return database_url.replace("postgresql+psycopg://", "postgresql+psycopg_async://", 1)
    return database_url.replace("postgresql://", "postgresql+psycopg_async://", 1)


@pytest_asyncio.fixture
async def engine() -> AsyncIterator[AsyncEngine]:
    eng = create_async_engine(_async_url(get_settings().database_url), pool_pre_ping=True)
    yield eng
    await eng.dispose()


@pytest_asyncio.fixture
async def session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False, autoflush=False)


@pytest_asyncio.fixture
async def app(session_factory: async_sessionmaker[AsyncSession]) -> AsyncIterator[FastAPI]:
    async def _override_get_db() -> AsyncIterator[AsyncSession]:
        async with session_factory() as session:
            yield session

    fastapi_app = FastAPI()
    fastapi_app.include_router(authoring_router, prefix="/api/v1")
    fastapi_app.dependency_overrides[get_db] = _override_get_db
    yield fastapi_app
    fastapi_app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def client(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as ac:
        yield ac


async def _seed_session(engine: AsyncEngine, user_id: uuid.UUID) -> uuid.UUID:
    session_id = uuid.uuid4()
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
                "exp": datetime.now(tz=UTC) + timedelta(hours=1),
            },
        )
    return session_id


@pytest_asyncio.fixture
async def manager_bearer(engine: AsyncEngine, seeded_users: SeededUsers) -> AsyncIterator[str]:
    sid = await _seed_session(engine, seeded_users.manager_id)
    yield create_access_token(user_id=seeded_users.manager_id, session_id=sid)
    async with engine.begin() as conn:
        await conn.execute(text("DELETE FROM auth_sessions WHERE id = :id"), {"id": sid})


@pytest_asyncio.fixture
async def teacher_bearer(engine: AsyncEngine, seeded_users: SeededUsers) -> AsyncIterator[str]:
    sid = await _seed_session(engine, seeded_users.teacher_id)
    yield create_access_token(user_id=seeded_users.teacher_id, session_id=sid)
    async with engine.begin() as conn:
        await conn.execute(text("DELETE FROM auth_sessions WHERE id = :id"), {"id": sid})


@pytest_asyncio.fixture
async def student_bearer(engine: AsyncEngine, seeded_users: SeededUsers) -> AsyncIterator[str]:
    sid = await _seed_session(engine, seeded_users.student_id)
    yield create_access_token(user_id=seeded_users.student_id, session_id=sid)
    async with engine.begin() as conn:
        await conn.execute(text("DELETE FROM auth_sessions WHERE id = :id"), {"id": sid})


@pytest_asyncio.fixture(autouse=True)
async def _no_object_storage(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[None]:
    """Neutralise the S3 put so these tests do not need a live bucket."""

    async def _fake_put(target: object, data: bytes, *, content_type: str) -> None:
        del target, data, content_type

    monkeypatch.setattr(
        "abridgeai.features.courses.services.syllabus_import.put_object_bytes",
        _fake_put,
    )
    yield


@pytest_asyncio.fixture
async def cleanup_imports(engine: AsyncEngine) -> AsyncIterator[None]:
    """Drop anything these tests created, children before parents."""
    yield
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "DELETE FROM course_learning_outcomes WHERE course_id IN "
                "(SELECT course_id FROM course_syllabus_imports WHERE course_id IS NOT NULL)"
            )
        )
        await conn.execute(
            text(
                "DELETE FROM courses WHERE id IN "
                "(SELECT course_id FROM course_syllabus_imports WHERE course_id IS NOT NULL)"
            )
        )
        await conn.execute(text("DELETE FROM course_syllabus_imports"))


async def _post(client: httpx.AsyncClient, bearer: str, body: bytes, language: str = "vi"):
    return await client.post(
        f"{_ENDPOINT}?language={language}&filename=syllabus.pdf",
        content=body,
        headers={
            "Authorization": f"Bearer {bearer}",
            "Content-Type": "application/pdf",
        },
    )


def test_route_is_registered() -> None:
    paths = {(r.path, tuple(sorted(r.methods))) for r in authoring_router.routes}  # type: ignore[attr-defined]
    assert ("/teacher/courses/import-syllabus", ("POST",)) in paths
    assert ("/teacher/courses/syllabus-imports", ("GET",)) in paths


async def test_unauthenticated_returns_401(client: httpx.AsyncClient) -> None:
    response = await client.post(
        f"{_ENDPOINT}?language=vi",
        content=_NOT_A_SYLLABUS_PDF,
        headers={"Content-Type": "application/pdf"},
    )
    assert response.status_code == 401


async def test_student_cannot_import(client: httpx.AsyncClient, student_bearer: str) -> None:
    response = await _post(client, student_bearer, _NOT_A_SYLLABUS_PDF)
    assert response.status_code == 403


async def test_teacher_without_outcome_manage_cannot_import(
    client: httpx.AsyncClient, teacher_bearer: str
) -> None:
    """The side door the second permission closes.

    A teacher holds ``course.create`` but not ``learning_outcome.manage``,
    and outcome authoring is manager-owned. Gating the importer on
    ``course.create`` alone would have let a teacher write outcomes through
    it — the very thing ``_REQUIRE_OUTCOME_CREATE`` blocks everywhere else.
    """
    response = await _post(client, teacher_bearer, _NOT_A_SYLLABUS_PDF)
    assert response.status_code == 403


async def test_language_is_required(client: httpx.AsyncClient, manager_bearer: str) -> None:
    response = await client.post(
        _ENDPOINT,
        content=_NOT_A_SYLLABUS_PDF,
        headers={
            "Authorization": f"Bearer {manager_bearer}",
            "Content-Type": "application/pdf",
        },
    )
    assert response.status_code == 422


async def test_unsupported_language_is_rejected(
    client: httpx.AsyncClient, manager_bearer: str
) -> None:
    """Only vi/en exist today; anything else must not fall through to a guess."""
    response = await _post(client, manager_bearer, _NOT_A_SYLLABUS_PDF, language="fr")
    assert response.status_code == 422


async def test_non_pdf_upload_is_rejected(
    client: httpx.AsyncClient,
    manager_bearer: str,
    cleanup_imports: None,
) -> None:
    response = await _post(client, manager_bearer, b"not a pdf at all")
    assert response.status_code == 422
    assert "unsupported_syllabus_type" in response.text


async def test_failed_import_records_an_attempt_and_creates_no_course(
    client: httpx.AsyncClient,
    manager_bearer: str,
    engine: AsyncEngine,
    cleanup_imports: None,
) -> None:
    """The failure row has to OUTLIVE the request that raised.

    It is what the manager's failure notification points at, and what the
    import history shows once the toast is gone — so it is committed on its
    own after the doomed transaction is rolled back.
    """
    response = await _post(client, manager_bearer, _NOT_A_SYLLABUS_PDF)
    assert response.status_code == 422

    async with engine.begin() as conn:
        row = (
            await conn.execute(
                text(
                    "SELECT status, course_id, error_message, outcome_count "
                    "FROM course_syllabus_imports ORDER BY created_at DESC LIMIT 1"
                )
            )
        ).one()
    assert row.status == "failed"
    assert row.course_id is None
    assert row.outcome_count == 0
    # The reason stored is the SAME string the API returned, so the
    # notification, the history row and the HTTP error cannot drift apart.
    assert row.error_message
    assert row.error_message.split(":")[0] in response.text
