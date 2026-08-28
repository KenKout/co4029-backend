from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest
import pytest_asyncio
from alembic import command
from alembic.config import Config
from conftest import SeededUsers
from fastapi import FastAPI
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

import abridgeai.features.access_control.models  # noqa: F401
import abridgeai.features.courses.models  # noqa: F401
import abridgeai.features.identity.models  # noqa: F401
import abridgeai.features.materials.models  # noqa: F401
import abridgeai.features.progress.models  # noqa: F401
from abridgeai.core.config import get_settings
from abridgeai.core.db import get_db
from abridgeai.core.security import create_access_token, generate_token, hash_secret
from abridgeai.features.progress.routers import authoring_router, learner_router


def _async_url(database_url: str) -> str:
    if "+psycopg_async" in database_url:
        return database_url
    if database_url.startswith("postgresql+psycopg://"):
        return database_url.replace(
            "postgresql+psycopg://", "postgresql+psycopg_async://", 1
        )
    if database_url.startswith("postgresql://"):
        return database_url.replace(
            "postgresql://", "postgresql+psycopg_async://", 1
        )
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
async def session_factory(
    engine: AsyncEngine,
) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False, autoflush=False)


@pytest_asyncio.fixture
async def app(
    session_factory: async_sessionmaker[AsyncSession],
) -> AsyncIterator[FastAPI]:
    async def _override_get_db() -> AsyncIterator[AsyncSession]:
        async with session_factory() as session:
            yield session

    fastapi_app = FastAPI()
    fastapi_app.include_router(learner_router, prefix="/api/v1")
    fastapi_app.include_router(authoring_router, prefix="/api/v1")
    fastapi_app.dependency_overrides[get_db] = _override_get_db
    yield fastapi_app
    fastapi_app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def client(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
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
async def student_bearer(
    engine: AsyncEngine, seeded_users: SeededUsers
) -> AsyncIterator[str]:
    sid = await _seed_session(engine, seeded_users.student_id)
    yield create_access_token(user_id=seeded_users.student_id, session_id=sid)
    async with engine.begin() as conn:
        await conn.execute(text("DELETE FROM auth_sessions WHERE id = :id"), {"id": sid})


@pytest_asyncio.fixture
async def teacher_bearer(
    engine: AsyncEngine, seeded_users: SeededUsers
) -> AsyncIterator[str]:
    sid = await _seed_session(engine, seeded_users.teacher_id)
    yield create_access_token(user_id=seeded_users.teacher_id, session_id=sid)
    async with engine.begin() as conn:
        await conn.execute(text("DELETE FROM auth_sessions WHERE id = :id"), {"id": sid})


@pytest_asyncio.fixture
async def admin_bearer(
    engine: AsyncEngine, seeded_users: SeededUsers
) -> AsyncIterator[str]:
    sid = await _seed_session(engine, seeded_users.admin_id)
    yield create_access_token(user_id=seeded_users.admin_id, session_id=sid)
    async with engine.begin() as conn:
        await conn.execute(text("DELETE FROM auth_sessions WHERE id = :id"), {"id": sid})


@pytest_asyncio.fixture
async def scenario(
    engine: AsyncEngine, seeded_users: SeededUsers
) -> AsyncIterator[dict[str, uuid.UUID]]:
    suffix = uuid.uuid4().hex[:8]
    module_id = uuid.uuid4()
    lesson_id = uuid.uuid4()
    storage_obj_id = uuid.uuid4()
    material_id = uuid.uuid4()
    version_id = uuid.uuid4()

    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO modules (id, course_id, title, position, status) "
                "VALUES (:m, :c, :title, 1, 'published')"
            ),
            {
                "m": module_id,
                "c": seeded_users.course_id,
                "title": f"Progress Module {suffix}",
            },
        )
        await conn.execute(
            text(
                "INSERT INTO lessons "
                "(id, module_id, slug, title, status, estimated_minutes) "
                "VALUES (:l, :m, :slug, :title, 'published', 20)"
            ),
            {
                "l": lesson_id,
                "m": module_id,
                "slug": f"progress-lesson-{suffix}",
                "title": "Progress Lesson",
            },
        )
        await conn.execute(
            text(
                "INSERT INTO storage_objects (id, bucket, object_key, mime_type) "
                "VALUES (:id, :b, :k, :m)"
            ),
            {
                "id": storage_obj_id,
                "b": "test",
                "k": f"progress/{storage_obj_id.hex}",
                "m": "application/pdf",
            },
        )
        await conn.execute(
            text(
                "INSERT INTO learning_materials "
                "(id, lesson_id, title, material_type) "
                "VALUES (:id, :lesson, 'Progress Material', 'pdf')"
            ),
            {"id": material_id, "lesson": lesson_id},
        )
        await conn.execute(
            text(
                "INSERT INTO learning_material_versions "
                "(id, material_id, storage_object_id, version_no, processing_status) "
                "VALUES (:id, :mat, :obj, 1, 'ready')"
            ),
            {"id": version_id, "mat": material_id, "obj": storage_obj_id},
        )
        await conn.execute(
            text(
                "INSERT INTO course_enrollments "
                "(course_id, student_id, status, source) "
                "VALUES (:cid, :sid, 'active', 'self_enroll') "
                "ON CONFLICT (course_id, student_id) DO NOTHING"
            ),
            {"cid": seeded_users.course_id, "sid": seeded_users.student_id},
        )

    yield {
        "course_id": seeded_users.course_id,
        "module_id": module_id,
        "lesson_id": lesson_id,
        "storage_obj_id": storage_obj_id,
        "material_id": material_id,
        "version_id": version_id,
    }

    async with engine.begin() as conn:
        await conn.execute(
            text("DELETE FROM material_engagement WHERE material_version_id = :v"),
            {"v": version_id},
        )
        await conn.execute(
            text("DELETE FROM lesson_progress WHERE lesson_id = :l"),
            {"l": lesson_id},
        )
        await conn.execute(
            text("DELETE FROM course_enrollments WHERE course_id = :cid"),
            {"cid": seeded_users.course_id},
        )
        await conn.execute(
            text("DELETE FROM learning_material_versions WHERE id = :id"),
            {"id": version_id},
        )
        await conn.execute(
            text("DELETE FROM learning_materials WHERE id = :id"),
            {"id": material_id},
        )
        await conn.execute(
            text("DELETE FROM storage_objects WHERE id = :id"),
            {"id": storage_obj_id},
        )
        await conn.execute(text("DELETE FROM lessons WHERE id = :l"), {"l": lesson_id})
        await conn.execute(text("DELETE FROM modules WHERE id = :m"), {"m": module_id})


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _engagement_payload(
    version_id: uuid.UUID,
    *,
    seconds: int,
    scroll: float | None = None,
) -> dict[str, object]:
    started = datetime.now(tz=UTC) - timedelta(seconds=seconds)
    ended = datetime.now(tz=UTC)
    payload: dict[str, object] = {
        "material_version_id": str(version_id),
        "engagement_seconds": seconds,
        "started_at": started.isoformat(),
        "ended_at": ended.isoformat(),
    }
    if scroll is not None:
        payload["scroll_position_percent"] = scroll
    return payload


@pytest.mark.asyncio
async def test_record_engagement_inserts_row(
    client: httpx.AsyncClient,
    engine: AsyncEngine,
    scenario: dict[str, uuid.UUID],
    student_bearer: str,
) -> None:
    payload = _engagement_payload(scenario["version_id"], seconds=120, scroll=25.5)

    response = await client.post(
        "/api/v1/me/progress/material-engagement",
        json=payload,
        headers=_auth(student_bearer),
    )

    assert response.status_code == 201, response.text
    body = response.json()
    assert body["engagement_seconds"] == 120
    assert body["material_version_id"] == str(scenario["version_id"])

    async with engine.begin() as conn:
        rows = (
            await conn.execute(
                text(
                    "SELECT engagement_seconds FROM material_engagement "
                    "WHERE material_version_id = :v"
                ),
                {"v": scenario["version_id"]},
            )
        ).all()
    assert [r.engagement_seconds for r in rows] == [120]


@pytest.mark.asyncio
async def test_engagement_to_progress(
    client: httpx.AsyncClient,
    scenario: dict[str, uuid.UUID],
    student_bearer: str,
) -> None:
    each = 240
    for _ in range(5):
        response = await client.post(
            "/api/v1/me/progress/material-engagement",
            json=_engagement_payload(scenario["version_id"], seconds=each),
            headers=_auth(student_bearer),
        )
        assert response.status_code == 201, response.text

    progress = await client.get(
        f"/api/v1/me/progress/lessons/{scenario['lesson_id']}",
        headers=_auth(student_bearer),
    )
    assert progress.status_code == 200, progress.text
    body = progress.json()
    assert body["status"] == "completed"
    assert float(body["completion_percent"]) == 100.0


@pytest.mark.asyncio
async def test_cross_user_progress_blocked(
    client: httpx.AsyncClient,
    scenario: dict[str, uuid.UUID],
    student_bearer: str,
    teacher_bearer: str,
) -> None:
    student_resp = await client.post(
        "/api/v1/me/progress/material-engagement",
        json=_engagement_payload(scenario["version_id"], seconds=60, scroll=10.0),
        headers=_auth(student_bearer),
    )
    assert student_resp.status_code == 201, student_resp.text

    teacher_view = await client.get(
        f"/api/v1/me/progress/lessons/{scenario['lesson_id']}",
        headers=_auth(teacher_bearer),
    )

    assert teacher_view.status_code == 404, teacher_view.text


@pytest.mark.asyncio
async def test_roster_progress_teacher_only(
    client: httpx.AsyncClient,
    scenario: dict[str, uuid.UUID],
    student_bearer: str,
    admin_bearer: str,
) -> None:
    student_resp = await client.get(
        f"/api/v1/teacher/courses/{scenario['course_id']}/progress/roster",
        headers=_auth(student_bearer),
    )
    assert student_resp.status_code == 403, student_resp.text

    admin_resp = await client.get(
        f"/api/v1/teacher/courses/{scenario['course_id']}/progress/roster",
        headers=_auth(admin_bearer),
    )
    assert admin_resp.status_code == 200, admin_resp.text
    body = admin_resp.json()
    assert body["course_id"] == str(scenario["course_id"])
    assert isinstance(body["students"], list)


@pytest.mark.asyncio
async def test_at_risk(
    client: httpx.AsyncClient,
    engine: AsyncEngine,
    scenario: dict[str, uuid.UUID],
    seeded_users: SeededUsers,
    admin_bearer: str,
) -> None:
    stale_engagement_id = uuid.uuid4()
    eight_days_ago = datetime.now(tz=UTC) - timedelta(days=8)
    async with engine.begin() as conn:
        # Push the enrolment outside the new-enrolment grace period
        # (``progress.at_risk_grace_period_days``, default 14). The fixture
        # enrols at NOW(), and a student inside grace is deliberately never
        # flagged -- without this the stale engagement below would be
        # invisible and the test would be asserting the grace rule, not the
        # inactivity rule it is named for.
        await conn.execute(
            text(
                "UPDATE course_enrollments SET enrolled_at = :t "
                "WHERE course_id = :c AND student_id = :s"
            ),
            {
                "t": datetime.now(tz=UTC) - timedelta(days=60),
                "c": seeded_users.course_id,
                "s": seeded_users.student_id,
            },
        )
        await conn.execute(
            text(
                "INSERT INTO material_engagement "
                "(id, user_id, material_version_id, engagement_seconds, "
                "scroll_position_percent, started_at, ended_at, created_at) "
                "VALUES (:id, :uid, :v, 30, 5, :s, :e, :c)"
            ),
            {
                "id": stale_engagement_id,
                "uid": seeded_users.student_id,
                "v": scenario["version_id"],
                "s": eight_days_ago - timedelta(seconds=30),
                "e": eight_days_ago,
                "c": eight_days_ago,
            },
        )

    response = await client.get(
        f"/api/v1/teacher/courses/{scenario['course_id']}/progress/at-risk",
        headers=_auth(admin_bearer),
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["course_id"] == str(scenario["course_id"])
    matched = [
        s for s in body["students"] if s["user_id"] == str(seeded_users.student_id)
    ]
    assert matched, "expected student flagged as at-risk"
    codes = {r["code"] for r in matched[0]["reasons"]}
    assert codes & {"inactive", "low_completion", "no_engagement"}


async def test_at_risk_respects_new_enrolment_grace_period(
    client: httpx.AsyncClient,
    engine: AsyncEngine,
    scenario: dict[str, uuid.UUID],
    seeded_users: SeededUsers,
    admin_bearer: str,
) -> None:
    """FR-025: a freshly enrolled student is not reported at risk.

    This student trips the completion rule outright (no lesson progress at
    all, so 0% < 30%) AND the no-engagement rule. Before the grace period
    existed they were flagged the moment they enrolled, which is the false
    positive that made the metric untrustworthy: "at risk" fired on people
    whose only failing was having just joined.

    The enrolment is left at the fixture's NOW() on purpose -- that is the
    condition under test.
    """
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "UPDATE course_enrollments SET enrolled_at = :t "
                "WHERE course_id = :c AND student_id = :s"
            ),
            {
                "t": datetime.now(tz=UTC) - timedelta(days=1),
                "c": seeded_users.course_id,
                "s": seeded_users.student_id,
            },
        )

    response = await client.get(
        f"/api/v1/teacher/courses/{scenario['course_id']}/progress/at-risk",
        headers=_auth(admin_bearer),
    )

    assert response.status_code == 200, response.text
    flagged = {s["user_id"] for s in response.json()["students"]}
    assert str(seeded_users.student_id) not in flagged


# ---------------------------------------------------------------------------
# Manual mark/unmark toggle (bug report 2026-08-04: uncomplete was a no-op)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_uncomplete_flips_back_even_when_engagement_would_auto_complete(
    client: httpx.AsyncClient,
    scenario: dict[str, uuid.UUID],
    student_bearer: str,
) -> None:
    """Regression: uncomplete used to recompute from engagement, and the
    auto-complete threshold (≥80%) re-asserted ``completed`` — a student who
    had watched enough could never un-tick the lesson (the button sent the
    request, the server silently kept ``completed``, UI showed no change).
    The manual toggle must win: status goes back to ``in_progress``."""
    # Enough engagement to auto-complete (lesson estimate is 20 min = 1200s).
    for _ in range(5):
        response = await client.post(
            "/api/v1/me/progress/material-engagement",
            json=_engagement_payload(scenario["version_id"], seconds=240),
            headers=_auth(student_bearer),
        )
        assert response.status_code == 201, response.text

    progress = await client.get(
        f"/api/v1/me/progress/lessons/{scenario['lesson_id']}",
        headers=_auth(student_bearer),
    )
    assert progress.status_code == 200, progress.text
    assert progress.json()["status"] == "completed"

    unmark = await client.post(
        f"/api/v1/me/progress/lessons/{scenario['lesson_id']}/uncomplete",
        json={},
        headers=_auth(student_bearer),
    )
    assert unmark.status_code == 200, unmark.text
    body = unmark.json()
    assert body["status"] == "in_progress", body
    # Raw engagement percent stays 100 (they did watch 20min of 20min) — the
    # toggle flips STATUS, which is what the curriculum keys on.
    assert float(body["completion_percent"]) == 100.0, body

    after = await client.get(
        f"/api/v1/me/progress/lessons/{scenario['lesson_id']}",
        headers=_auth(student_bearer),
    )
    assert after.json()["status"] == "in_progress"


@pytest.mark.asyncio
async def test_uncomplete_then_new_engagement_re_auto_completes(
    client: httpx.AsyncClient,
    scenario: dict[str, uuid.UUID],
    student_bearer: str,
) -> None:
    """After unmarking, a NEW engagement heartbeat legitimately re-applies
    auto-completion — the student keeps watching past the threshold, so the
    lesson becomes complete again. Documents the re-assert behaviour so the
    toggle isn't mistaken for a permanent lock."""
    for _ in range(5):
        response = await client.post(
            "/api/v1/me/progress/material-engagement",
            json=_engagement_payload(scenario["version_id"], seconds=240),
            headers=_auth(student_bearer),
        )
        assert response.status_code == 201, response.text

    unmark = await client.post(
        f"/api/v1/me/progress/lessons/{scenario['lesson_id']}/uncomplete",
        json={},
        headers=_auth(student_bearer),
    )
    assert unmark.status_code == 200, unmark.text
    assert unmark.json()["status"] == "in_progress", unmark.text

    # One more heartbeat crosses the threshold again → auto-complete re-applies.
    extra = await client.post(
        "/api/v1/me/progress/material-engagement",
        json=_engagement_payload(scenario["version_id"], seconds=240),
        headers=_auth(student_bearer),
    )
    assert extra.status_code == 201, extra.text

    after = await client.get(
        f"/api/v1/me/progress/lessons/{scenario['lesson_id']}",
        headers=_auth(student_bearer),
    )
    assert after.status_code == 200, after.text
    assert after.json()["status"] == "completed"
