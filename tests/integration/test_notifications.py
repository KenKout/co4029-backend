from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock
from uuid import UUID

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
import abridgeai.features.identity.models  # noqa: F401
import abridgeai.features.notifications.models  # noqa: F401
from abridgeai.core.config import get_settings
from abridgeai.core.db import get_db
from abridgeai.core.security import create_access_token, generate_token, hash_secret
from abridgeai.features.notifications.models import (
    Notification,
    NotificationPreference,
)
from abridgeai.features.notifications.routers import learner_router
from abridgeai.features.notifications.services import dispatch as dispatch_service
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
    fastapi_app.dependency_overrides[get_db] = _override_get_db
    yield fastapi_app
    fastapi_app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def client(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
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
async def student_bearer(engine: AsyncEngine, seeded_users: SeededUsers) -> AsyncIterator[str]:
    sid = await _seed_session(engine, seeded_users.student_id)
    yield create_access_token(user_id=seeded_users.student_id, session_id=sid)
    async with engine.begin() as conn:
        await conn.execute(text("DELETE FROM auth_sessions WHERE id = :id"), {"id": sid})


@pytest_asyncio.fixture
async def teacher_bearer(engine: AsyncEngine, seeded_users: SeededUsers) -> AsyncIterator[str]:
    sid = await _seed_session(engine, seeded_users.teacher_id)
    yield create_access_token(user_id=seeded_users.teacher_id, session_id=sid)
    async with engine.begin() as conn:
        await conn.execute(text("DELETE FROM auth_sessions WHERE id = :id"), {"id": sid})


@pytest_asyncio.fixture
async def clean_notifications(
    engine: AsyncEngine, seeded_users: SeededUsers
) -> AsyncIterator[None]:
    async with engine.begin() as conn:
        await conn.execute(
            text("DELETE FROM notifications WHERE user_id IN (:s, :t)"),
            {"s": seeded_users.student_id, "t": seeded_users.teacher_id},
        )
        await conn.execute(
            text("DELETE FROM notification_preferences WHERE user_id IN (:s, :t)"),
            {"s": seeded_users.student_id, "t": seeded_users.teacher_id},
        )
    yield
    async with engine.begin() as conn:
        await conn.execute(
            text("DELETE FROM notifications WHERE user_id IN (:s, :t)"),
            {"s": seeded_users.student_id, "t": seeded_users.teacher_id},
        )
        await conn.execute(
            text("DELETE FROM notification_preferences WHERE user_id IN (:s, :t)"),
            {"s": seeded_users.student_id, "t": seeded_users.teacher_id},
        )


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


@pytest.mark.asyncio
async def test_send_notification_creates_in_app_row(
    session_factory: async_sessionmaker[AsyncSession],
    seeded_users: SeededUsers,
    clean_notifications: None,
) -> None:
    async with session_factory() as session:
        notif = await dispatch_service.send_notification(
            session,
            recipient_user_id=seeded_users.student_id,
            notification_type="course_announcement",
            title="Welcome",
            body="In-app row should land here.",
            arq_pool=None,
        )
        await session.commit()
        assert notif.user_id == seeded_users.student_id
        assert notif.category == "course_announcement"
        assert notif.delivery_status == "pending"

    async with session_factory() as session:
        row = await session.get(Notification, notif.id)
        assert row is not None
        assert row.title == "Welcome"
        assert row.read_at is None


@pytest.mark.asyncio
async def test_email_pref_enabled_enqueues_arq(
    session_factory: async_sessionmaker[AsyncSession],
    seeded_users: SeededUsers,
    clean_notifications: None,
) -> None:
    pool = AsyncMock()
    pool.enqueue_job = AsyncMock(return_value=None)

    async with session_factory() as session:
        await dispatch_service.send_notification(
            session,
            recipient_user_id=seeded_users.student_id,
            notification_type="interview_result",
            title="Interview ready",
            body="Open the app to view.",
            arq_pool=pool,
        )
        await session.commit()

    pool.enqueue_job.assert_awaited_once()
    args = pool.enqueue_job.await_args
    assert args is not None
    assert args.args[0] == "send_email_notification_task"
    assert args.args[1] == seeded_users.student_id
    assert isinstance(args.args[2], UUID)


@pytest.mark.asyncio
async def test_email_pref_disabled_skips_email(
    session_factory: async_sessionmaker[AsyncSession],
    seeded_users: SeededUsers,
    clean_notifications: None,
) -> None:
    pool = AsyncMock()
    pool.enqueue_job = AsyncMock(return_value=None)

    async with session_factory() as session:
        session.add(
            NotificationPreference(
                user_id=seeded_users.student_id,
                category="lesson_unlock",
                channel="email",
                enabled=False,
            )
        )
        await session.commit()

    async with session_factory() as session:
        notif = await dispatch_service.send_notification(
            session,
            recipient_user_id=seeded_users.student_id,
            notification_type="lesson_unlock",
            title="Lesson 5 unlocked",
            body="Continue learning.",
            arq_pool=pool,
        )
        await session.commit()
        assert notif.id is not None

    pool.enqueue_job.assert_not_awaited()

    async with session_factory() as session:
        row = await session.get(Notification, notif.id)
        assert row is not None


@pytest.mark.asyncio
async def test_mark_as_read_sets_read_at(
    client: httpx.AsyncClient,
    session_factory: async_sessionmaker[AsyncSession],
    seeded_users: SeededUsers,
    student_bearer: str,
    clean_notifications: None,
) -> None:
    async with session_factory() as session:
        notif = await dispatch_service.send_notification(
            session,
            recipient_user_id=seeded_users.student_id,
            notification_type="system",
            title="Heads up",
            body="Just so you know.",
            arq_pool=None,
        )
        await session.commit()

    response = await client.patch(
        f"/api/v1/me/notifications/{notif.id}/read",
        headers=_auth(student_bearer),
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["id"] == str(notif.id)
    assert body["read_at"] is not None

    async with session_factory() as session:
        row = await session.get(Notification, notif.id)
        assert row is not None
        assert row.read_at is not None


@pytest.mark.asyncio
async def test_dismiss_soft_deletes(
    client: httpx.AsyncClient,
    session_factory: async_sessionmaker[AsyncSession],
    seeded_users: SeededUsers,
    student_bearer: str,
    clean_notifications: None,
) -> None:
    async with session_factory() as session:
        notif = await dispatch_service.send_notification(
            session,
            recipient_user_id=seeded_users.student_id,
            notification_type="system",
            title="Dismiss me",
            body="Bye.",
            arq_pool=None,
        )
        await session.commit()

    delete_resp = await client.delete(
        f"/api/v1/me/notifications/{notif.id}",
        headers=_auth(student_bearer),
    )
    assert delete_resp.status_code == 204, delete_resp.text

    async with session_factory() as session:
        row = await session.get(Notification, notif.id)
        assert row is not None
        assert row.delivery_status == "cancelled"

    list_resp = await client.get("/api/v1/me/notifications", headers=_auth(student_bearer))
    assert list_resp.status_code == 200
    ids = [item["id"] for item in list_resp.json()]
    assert str(notif.id) not in ids


@pytest.mark.asyncio
async def test_unread_count_excludes_dismissed_and_read(
    client: httpx.AsyncClient,
    session_factory: async_sessionmaker[AsyncSession],
    seeded_users: SeededUsers,
    student_bearer: str,
    clean_notifications: None,
) -> None:
    created: list[UUID] = []
    async with session_factory() as session:
        for i in range(3):
            notif = await dispatch_service.send_notification(
                session,
                recipient_user_id=seeded_users.student_id,
                notification_type="system",
                title=f"N{i}",
                body=str(i),
                arq_pool=None,
            )
            created.append(notif.id)
        await session.commit()

    initial = await client.get(
        "/api/v1/me/notifications/unread-count", headers=_auth(student_bearer)
    )
    assert initial.status_code == 200
    assert initial.json() == {"unread": 3}

    await client.patch(
        f"/api/v1/me/notifications/{created[0]}/read",
        headers=_auth(student_bearer),
    )
    await client.delete(
        f"/api/v1/me/notifications/{created[1]}",
        headers=_auth(student_bearer),
    )

    final = await client.get("/api/v1/me/notifications/unread-count", headers=_auth(student_bearer))
    assert final.status_code == 200
    assert final.json() == {"unread": 1}


@pytest.mark.asyncio
async def test_other_user_cannot_read_my_notifications(
    client: httpx.AsyncClient,
    session_factory: async_sessionmaker[AsyncSession],
    seeded_users: SeededUsers,
    student_bearer: str,
    teacher_bearer: str,
    clean_notifications: None,
) -> None:
    async with session_factory() as session:
        notif = await dispatch_service.send_notification(
            session,
            recipient_user_id=seeded_users.student_id,
            notification_type="system",
            title="Private",
            body="Only the student should see this.",
            arq_pool=None,
        )
        await session.commit()

    response = await client.patch(
        f"/api/v1/me/notifications/{notif.id}/read",
        headers=_auth(teacher_bearer),
    )
    assert response.status_code == 404, response.text

    delete_resp = await client.delete(
        f"/api/v1/me/notifications/{notif.id}",
        headers=_auth(teacher_bearer),
    )
    assert delete_resp.status_code == 404, delete_resp.text

    teacher_list = await client.get("/api/v1/me/notifications", headers=_auth(teacher_bearer))
    assert teacher_list.status_code == 200
    ids = [item["id"] for item in teacher_list.json()]
    assert str(notif.id) not in ids


@pytest.mark.asyncio
async def test_arq_app_includes_notification_jobs() -> None:
    function_names = {fn.__name__ for fn in WorkerSettings.functions}
    assert "send_email_notification_task" in function_names


@pytest.mark.asyncio
async def test_preference_toggle_endpoint(
    client: httpx.AsyncClient,
    session_factory: async_sessionmaker[AsyncSession],
    seeded_users: SeededUsers,
    student_bearer: str,
    clean_notifications: None,
) -> None:
    response = await client.patch(
        "/api/v1/me/notification-preferences/lesson_unlock/email",
        json={"enabled": False},
        headers=_auth(student_bearer),
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["enabled"] is False
    assert body["category"] == "lesson_unlock"
    assert body["channel"] == "email"

    listing = await client.get("/api/v1/me/notification-preferences", headers=_auth(student_bearer))
    assert listing.status_code == 200
    rows = listing.json()
    assert any(
        r["category"] == "lesson_unlock" and r["channel"] == "email" and r["enabled"] is False
        for r in rows
    )
