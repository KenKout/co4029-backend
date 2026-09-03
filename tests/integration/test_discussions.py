"""Integration tests for ``features.discussions`` (lesson topics + comments).

Covers the teacher/student discussion flow end to end:

* teacher-authoring perimeters (create/close/delete topic) vs
  student perimeters (read, own-comment edit/delete),
* the enrolled/manage read gate (404 for outsiders, no existence leak),
* the closed-topic lock (students can't comment, teachers can),
* the two-way comment notifications: student comment -> course teachers,
  teacher reply -> thread participants, no self-notification.
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
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from abridgeai.core.config import get_settings
from abridgeai.core.db import get_db
from abridgeai.core.security import create_access_token, generate_token, hash_secret
from abridgeai.features.discussions.router import router as discussions_router


def _async_url(database_url: str) -> str:
    if "+psycopg_async" in database_url:
        return database_url
    return database_url.replace("postgresql+psycopg://", "postgresql+psycopg_async://", 1)


@pytest_asyncio.fixture
async def engine() -> AsyncIterator[AsyncEngine]:  # noqa: ASYNC240
    cfg = Config(str(Path(__file__).resolve().parents[2] / "alembic.ini"))  # noqa: ASYNC240
    cfg.set_main_option(
        "script_location", str(Path(__file__).resolve().parents[2] / "migrations")  # noqa: ASYNC240
    )
    command.upgrade(cfg, "head")
    eng = create_async_engine(_async_url(get_settings().database_url), pool_pre_ping=True)
    yield eng
    await eng.dispose()


@pytest_asyncio.fixture
async def app(engine: AsyncEngine) -> AsyncIterator[FastAPI]:
    sm = async_sessionmaker(engine, expire_on_commit=False, autoflush=False)

    async def _override_get_db() -> AsyncIterator[AsyncSession]:
        async with sm() as session:
            yield session

    fastapi_app = FastAPI()
    fastapi_app.include_router(discussions_router, prefix="/api/v1")
    fastapi_app.dependency_overrides[get_db] = _override_get_db
    yield fastapi_app
    fastapi_app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def client(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as ac:
        yield ac


class Graph:
    """One published course + lesson, teacher-owned, one enrolled student."""

    def __init__(self) -> None:
        self.tag = uuid.uuid4().hex[:10]
        self.course_id = uuid.uuid4()
        self.module_id = uuid.uuid4()
        self.lesson_id = uuid.uuid4()
        self.item_id = uuid.uuid4()

    @property
    def course_slug(self) -> str:
        return f"disc-{self.tag}"

    @property
    def lesson_slug(self) -> str:
        return f"disc-lesson-{self.tag}"


async def _open_session(engine: AsyncEngine, user_id: uuid.UUID) -> tuple[uuid.UUID, str]:
    session_id = uuid.uuid4()
    expires_at = datetime.now(tz=UTC) + timedelta(hours=1)
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO auth_sessions (id, user_id, refresh_token_hash, expires_at) "
                "VALUES (:id, :uid, :h, :exp)"
            ),
            {"id": session_id, "uid": user_id, "h": hash_secret(generate_token()), "exp": expires_at},
        )
    return session_id, create_access_token(user_id=user_id, session_id=session_id)


async def _close_session(engine: AsyncEngine, session_id: uuid.UUID) -> None:
    async with engine.begin() as conn:
        await conn.execute(text("DELETE FROM auth_sessions WHERE id = :id"), {"id": session_id})


_CREATED: list[Graph] = []


async def _seed_graph(engine: AsyncEngine, seeded: SeededUsers) -> Graph:
    g = Graph()
    _CREATED.append(g)
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO courses (id, organization_id, owner_user_id, slug, title, status) "
                "VALUES (:id, :org, :owner, :slug, 'Discussion Course', 'published')"
            ),
            {"id": g.course_id, "org": seeded.organization_id, "owner": seeded.teacher_id, "slug": g.course_slug},
        )
        await conn.execute(
            text(
                "INSERT INTO modules (id, course_id, title, position, status) "
                "VALUES (:id, :c, 'Discussion Module', 1, 'published')"
            ),
            {"id": g.module_id, "c": g.course_id},
        )
        await conn.execute(
            text(
                "INSERT INTO lessons (id, module_id, slug, title, status) "
                "VALUES (:id, :m, :slug, 'Discussion Lesson', 'published')"
            ),
            {"id": g.lesson_id, "m": g.module_id, "slug": g.lesson_slug},
        )
        await conn.execute(
            text(
                "INSERT INTO module_items (id, module_id, item_type, lesson_id, position) "
                "VALUES (:id, :m, 'lesson', :l, 1)"
            ),
            {"id": g.item_id, "m": g.module_id, "l": g.lesson_id},
        )
        await conn.execute(
            text(
                "INSERT INTO course_enrollments (course_id, student_id, status, source) "
                "VALUES (:c, :s, 'active', 'manager_bulk')"
            ),
            {"c": g.course_id, "s": seeded.student_id},
        )
    return g


async def _drain_graphs(engine: AsyncEngine) -> None:
    while _CREATED:
        g = _CREATED.pop()
        async with engine.begin() as conn:
            await conn.execute(
                text(
                    "DELETE FROM lesson_discussion_comments "
                    "WHERE topic_id IN (SELECT id FROM lesson_discussion_topics WHERE lesson_id = :l)"
                ),
                {"l": g.lesson_id},
            )
            await conn.execute(
                text("DELETE FROM lesson_discussion_topics WHERE lesson_id = :l"), {"l": g.lesson_id}
            )
            await conn.execute(text("DELETE FROM course_enrollments WHERE course_id = :c"), {"c": g.course_id})
            await conn.execute(text("DELETE FROM module_items WHERE id = :i"), {"i": g.item_id})
            await conn.execute(text("DELETE FROM lessons WHERE id = :l"), {"l": g.lesson_id})
            await conn.execute(text("DELETE FROM modules WHERE id = :m"), {"m": g.module_id})
            await conn.execute(text("DELETE FROM courses WHERE id = :c"), {"c": g.course_id})


@pytest_asyncio.fixture(autouse=True)
async def _drain(engine: AsyncEngine) -> AsyncIterator[None]:
    yield
    await _drain_graphs(engine)


async def _notifications_for(engine: AsyncEngine, user_id: uuid.UUID) -> list[dict[str, object]]:
    async with engine.begin() as conn:
        rows = (
            await conn.execute(
                text(
                    "SELECT category, entity_type, entity_id, action_url, title, body "
                    "FROM notifications WHERE user_id = :uid ORDER BY created_at DESC"
                ),
                {"uid": user_id},
            )
        ).mappings().all()
    return [dict(r) for r in rows]


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


# ── fixtures for the three actors ─────────────────────────────────────────

@pytest_asyncio.fixture
async def teacher_auth(engine: AsyncEngine, seeded_users: SeededUsers) -> AsyncIterator[tuple[uuid.UUID, str]]:
    sid, token = await _open_session(engine, seeded_users.teacher_id)
    try:
        yield sid, token
    finally:
        await _close_session(engine, sid)


@pytest_asyncio.fixture
async def student_auth(engine: AsyncEngine, seeded_users: SeededUsers) -> AsyncIterator[tuple[uuid.UUID, str]]:
    sid, token = await _open_session(engine, seeded_users.student_id)
    try:
        yield sid, token
    finally:
        await _close_session(engine, sid)


async def test_teacher_creates_topic_and_student_posts_comment(
    client: httpx.AsyncClient,
    engine: AsyncEngine,
    seeded_users: SeededUsers,
    teacher_auth: tuple[uuid.UUID, str],
    student_auth: tuple[uuid.UUID, str],
) -> None:
    _, teacher_token = teacher_auth
    _, student_token = student_auth
    g = await _seed_graph(engine, seeded_users)

    # Teacher opens a topic.
    created = await client.post(
        f"/api/v1/lessons/{g.lesson_id}/discussion/topics",
        headers=_auth(teacher_token),
        json={"title": "What did you find hard?", "body_markdown": "Discuss the ordering proof."},
    )
    assert created.status_code == 201, created.text
    topic_id = created.json()["id"]

    # Student lists: can_manage False, one topic, zero comments.
    listed = await client.get(
        f"/api/v1/lessons/{g.lesson_id}/discussion/topics", headers=_auth(student_token)
    )
    assert listed.status_code == 200, listed.text
    assert listed.json()["can_manage"] is False
    assert len(listed.json()["topics"]) == 1
    assert listed.json()["topics"][0]["comment_count"] == 0

    # Student comments.
    commented = await client.post(
        f"/api/v1/discussion/topics/{topic_id}/comments",
        headers=_auth(student_token),
        json={"body": "The Cauchy step tripped me up."},
    )
    assert commented.status_code == 201, commented.text
    assert commented.json()["is_own"] is True

    # ── notification: the student's comment reached the course owner/teacher.
    notes = await _notifications_for(engine, seeded_users.teacher_id)
    match = next((n for n in notes if str(n["entity_id"]) == commented.json()["id"]), None)
    assert match is not None, "teacher got no notification for the student comment"
    assert match["category"] == "course_discussion"
    assert match["entity_type"] == "lesson_discussion_comment"
    assert match["action_url"] == f"/courses/{g.course_slug}/learn?item={g.lesson_slug}"
    assert "Cauchy" in str(match["body"])

    # no self-notification for the student
    student_notes = await _notifications_for(engine, seeded_users.student_id)
    assert not any(
        str(n["entity_id"]) == commented.json()["id"] for n in student_notes
    ), "student must not be notified about their own comment"


async def test_teacher_reply_notifies_thread_participants_and_closed_topic_locks_students(
    client: httpx.AsyncClient,
    engine: AsyncEngine,
    seeded_users: SeededUsers,
    teacher_auth: tuple[uuid.UUID, str],
    student_auth: tuple[uuid.UUID, str],
) -> None:
    _, teacher_token = teacher_auth
    _, student_token = student_auth
    g = await _seed_graph(engine, seeded_users)

    topic = await client.post(
        f"/api/v1/lessons/{g.lesson_id}/discussion/topics",
        headers=_auth(teacher_token),
        json={"title": "Switch budget?"},
    )
    topic_id = topic.json()["id"]
    student_comment = await client.post(
        f"/api/v1/discussion/topics/{topic_id}/comments",
        headers=_auth(student_token),
        json={"body": "Student question one"},
    )
    assert student_comment.status_code == 201

    # Teacher replies -> the prior student commenter is notified.
    reply = await client.post(
        f"/api/v1/discussion/topics/{topic_id}/comments",
        headers=_auth(teacher_token),
        json={"body": "Teacher answer"},
    )
    assert reply.status_code == 201, reply.text
    student_notes = await _notifications_for(engine, seeded_users.student_id)
    match = next((n for n in student_notes if str(n["entity_id"]) == reply.json()["id"]), None)
    assert match is not None, "student was not notified about the teacher's reply"
    assert match["category"] == "course_discussion"

    # Teacher self-notification must not exist for the reply.
    teacher_notes = await _notifications_for(engine, seeded_users.teacher_id)
    assert not any(str(n["entity_id"]) == reply.json()["id"] for n in teacher_notes)
    before_count = len(teacher_notes)

    # Close the topic: students are locked out (404, no leak), teachers may post.
    closed = await client.patch(
        f"/api/v1/discussion/topics/{topic_id}",
        headers=_auth(teacher_token),
        json={"status": "closed"},
    )
    assert closed.status_code == 200, closed.text

    blocked = await client.post(
        f"/api/v1/discussion/topics/{topic_id}/comments",
        headers=_auth(student_token),
        json={"body": "Should be refused"},
    )
    assert blocked.status_code == 404, blocked.text

    after_count = len(await _notifications_for(engine, seeded_users.teacher_id))
    assert after_count == before_count, "blocked comment must not notify anyone"

    teacher_after = await client.post(
        f"/api/v1/discussion/topics/{topic_id}/comments",
        headers=_auth(teacher_token),
        json={"body": "Teachers always can post"},
    )
    assert teacher_after.status_code == 201, teacher_after.text


async def test_comment_edit_delete_and_moderation_perimeters(
    client: httpx.AsyncClient,
    engine: AsyncEngine,
    seeded_users: SeededUsers,
    teacher_auth: tuple[uuid.UUID, str],
    student_auth: tuple[uuid.UUID, str],
) -> None:
    _, teacher_token = teacher_auth
    _, student_token = student_auth
    g = await _seed_graph(engine, seeded_users)

    topic = await client.post(
        f"/api/v1/lessons/{g.lesson_id}/discussion/topics",
        headers=_auth(teacher_token),
        json={"title": "Edit me"},
    )
    topic_id = topic.json()["id"]
    mine = await client.post(
        f"/api/v1/discussion/topics/{topic_id}/comments",
        headers=_auth(student_token),
        json={"body": "Original"},
    )
    mine_id = mine.json()["id"]
    teacher_comment = await client.post(
        f"/api/v1/discussion/topics/{topic_id}/comments",
        headers=_auth(teacher_token),
        json={"body": "Teacher's own"},
    )
    teacher_comment_id = teacher_comment.json()["id"]

    # Author edits own comment; cannot edit someone else's.
    edited = await client.patch(
        f"/api/v1/discussion/comments/{mine_id}",
        headers=_auth(student_token),
        json={"body": "Edited by author"},
    )
    assert edited.status_code == 200, edited.text
    assert edited.json()["body"] == "Edited by author"

    cross_edit = await client.patch(
        f"/api/v1/discussion/comments/{teacher_comment_id}",
        headers=_auth(student_token),
        json={"body": "hijack"},
    )
    assert cross_edit.status_code == 404, cross_edit.text

    # Teacher moderates: deletes the student comment.
    deleted = await client.delete(
        f"/api/v1/discussion/comments/{mine_id}", headers=_auth(teacher_token)
    )
    assert deleted.status_code == 204, deleted.text
    comments = await client.get(
        f"/api/v1/discussion/topics/{topic_id}/comments", headers=_auth(student_token)
    )
    assert comments.status_code == 200
    assert all(c["id"] != mine_id for c in comments.json())


async def test_outsiders_are_invisible_and_topic_delete_cascades(
    client: httpx.AsyncClient,
    engine: AsyncEngine,
    seeded_users: SeededUsers,
    teacher_auth: tuple[uuid.UUID, str],
) -> None:
    _, teacher_token = teacher_auth
    g = await _seed_graph(engine, seeded_users)

    # An org member who is NOT enrolled and NOT a teacher gets 404 on reads.
    outsider_uid = uuid.uuid4()
    async with engine.begin() as conn:
        await conn.execute(
            text("INSERT INTO users (id, primary_email, status) VALUES (:id, :e, 'active')"),
            {"id": outsider_uid, "e": f"outsider-{g.tag}@abridgeai.local"},
        )
        await conn.execute(
            text(
                "INSERT INTO organization_memberships (id, user_id, organization_id, status) "
                "VALUES (gen_random_uuid(), :uid, :org, 'active')"
            ),
            {"uid": outsider_uid, "org": seeded_users.organization_id},
        )
    sid, outsider_token = await _open_session(engine, outsider_uid)
    try:
        hidden = await client.get(
            f"/api/v1/lessons/{g.lesson_id}/discussion/topics", headers=_auth(outsider_token)
        )
        assert hidden.status_code == 404, hidden.text
    finally:
        async with engine.begin() as conn:
            await conn.execute(text("DELETE FROM auth_sessions WHERE id = :s"), {"s": sid})
            await conn.execute(
                text("DELETE FROM organization_memberships WHERE user_id = :u"), {"u": outsider_uid}
            )
            await conn.execute(text("DELETE FROM users WHERE id = :u"), {"u": outsider_uid})

    # Delete the topic (soft) -> list is empty.
    topic = await client.post(
        f"/api/v1/lessons/{g.lesson_id}/discussion/topics",
        headers=_auth(teacher_token),
        json={"title": "Doomed topic"},
    )
    topic_id = topic.json()["id"]
    await client.post(
        f"/api/v1/discussion/topics/{topic_id}/comments",
        headers=_auth(teacher_token),
        json={"body": "Doomed comment"},
    )
    gone = await client.delete(f"/api/v1/discussion/topics/{topic_id}", headers=_auth(teacher_token))
    assert gone.status_code == 204, gone.text
    listed = await client.get(
        f"/api/v1/lessons/{g.lesson_id}/discussion/topics", headers=_auth(teacher_token)
    )
    assert listed.status_code == 200
    assert listed.json()["topics"] == []
