"""Integration tests for ``features.courses.routers.learner`` (T3.6).

Covers the 12 plan-mandated scenarios:

* Visibility — drafts excluded from list, slug lookup, content tree,
  module / lesson detail, lesson resources.
* Authoring leakage — ``?owned=true`` ignored, no authoring import in
  learner.py, ``Literal["published"]`` narrowing forces 404 for drafts.
* Auth — every endpoint returns 401 without a bearer token.
* Pagination — cursor round-trip walks all pages without overlap.
* Download URL — 404 path never leaks invisible resource existence.
* Outcomes — sorted by position (§A12).
"""

from __future__ import annotations

import re
import uuid
from collections.abc import AsyncIterator
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
import abridgeai.features.identity.models  # noqa: F401  -- register users FK target
from abridgeai.core.config import get_settings
from abridgeai.core.db import Base, get_db
from abridgeai.features.courses.routers.learner import (
    me_courses_router,
)
from abridgeai.features.courses.routers.learner import (
    router as learner_router,
)

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


@pytest_asyncio.fixture
async def app(
    session_factory: async_sessionmaker[AsyncSession],
) -> AsyncIterator[FastAPI]:
    async def _override_get_db() -> AsyncIterator[AsyncSession]:
        async with session_factory() as session:
            yield session

    fastapi_app = FastAPI()
    fastapi_app.include_router(learner_router, prefix="/api/v1")
    fastapi_app.include_router(me_courses_router, prefix="/api/v1")
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
    """Seed an ``auth_sessions`` row backing :func:`student_token`.

    ``conftest._token`` mints a JWT with ``session_id=uuid4()``; the
    JWT-decode branch in :func:`get_current_user` joins ``auth_sessions``
    and rejects unknown ids. Tests cannot reuse the JWT fixture without
    a matching session row, so we re-mint here.
    """
    from datetime import UTC, datetime, timedelta

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
    """Seed published + draft courses under the seeded test organization.

    The student token resolves to ``seeded_users.organization_id`` via
    ``user_role_assignments``, so courses must live there for the
    ``GET /courses`` and ``GET /courses/by-slug/...`` endpoints.
    """
    pub_course = uuid.uuid4()
    draft_course = uuid.uuid4()
    pub_module = uuid.uuid4()
    draft_module = uuid.uuid4()
    pub_lesson = uuid.uuid4()
    draft_lesson = uuid.uuid4()
    item_visible = uuid.uuid4()
    item_hidden = uuid.uuid4()
    resource_visible = uuid.uuid4()
    resource_hidden = uuid.uuid4()
    resource_no_storage = uuid.uuid4()
    suffix = pub_course.hex[:8]
    pub_slug = f"pub-{suffix}"
    draft_slug = f"draft-{suffix}"

    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO courses "
                "(id, organization_id, owner_user_id, slug, title, status) VALUES "
                "(:c1, :org, :owner, :s1, 'Published Course', 'published'), "
                "(:c2, :org, :owner, :s2, 'Draft Course', 'draft')"
            ),
            {
                "c1": pub_course,
                "c2": draft_course,
                "s1": pub_slug,
                "s2": draft_slug,
                "org": seeded_users.organization_id,
                "owner": seeded_users.teacher_id,
            },
        )
        await conn.execute(
            text(
                "INSERT INTO course_learning_outcomes "
                "(course_id, position, outcome_text) VALUES "
                "(:c, 3, 'Third'), (:c, 1, 'First'), (:c, 2, 'Second')"
            ),
            {"c": pub_course},
        )
        await conn.execute(
            text(
                "INSERT INTO modules (id, course_id, title, position, status) VALUES "
                "(:m1, :c, 'Pub Module', 1, 'published'), "
                "(:m2, :c, 'Draft Module', 2, 'draft')"
            ),
            {"m1": pub_module, "m2": draft_module, "c": pub_course},
        )
        await conn.execute(
            text(
                "INSERT INTO lessons (id, module_id, slug, title, status) VALUES "
                "(:l1, :m, 'pub-lesson', 'Pub Lesson', 'published'), "
                "(:l2, :m, 'draft-lesson', 'Draft Lesson', 'draft')"
            ),
            {"l1": pub_lesson, "l2": draft_lesson, "m": pub_module},
        )
        await conn.execute(
            text(
                "INSERT INTO module_items "
                "(id, module_id, item_type, lesson_id, position) VALUES "
                "(:i1, :m, 'lesson', :l1, 1), "
                "(:i2, :m, 'lesson', :l2, 2)"
            ),
            {
                "i1": item_visible,
                "i2": item_hidden,
                "m": pub_module,
                "l1": pub_lesson,
                "l2": draft_lesson,
            },
        )
        storage_obj_id = uuid.uuid4()
        await conn.execute(
            text("INSERT INTO storage_objects (id, bucket, object_key) VALUES (:id, :b, :k)"),
            {"id": storage_obj_id, "b": "test-bucket", "k": f"materials/{suffix}.pdf"},
        )
        await conn.execute(
            text(
                "INSERT INTO lesson_resources "
                "(id, lesson_id, title, resource_type, position, "
                "visible_to_students, storage_object_id) VALUES "
                "(:r1, :l, 'Visible PDF', 'pdf', 1, TRUE, :so), "
                "(:r2, :l, 'Hidden PDF', 'pdf', 2, FALSE, :so), "
                "(:r3, :l, 'Visible No Storage', 'link', 3, TRUE, NULL)"
            ),
            {
                "r1": resource_visible,
                "r2": resource_hidden,
                "r3": resource_no_storage,
                "l": pub_lesson,
                "so": storage_obj_id,
            },
        )

    data = {
        "pub_course": pub_course,
        "draft_course": draft_course,
        "pub_module": pub_module,
        "draft_module": draft_module,
        "pub_lesson": pub_lesson,
        "draft_lesson": draft_lesson,
        "item_visible": item_visible,
        "item_hidden": item_hidden,
        "resource_visible": resource_visible,
        "resource_hidden": resource_hidden,
        "resource_no_storage": resource_no_storage,
        "storage_object_id": storage_obj_id,
        "pub_slug": pub_slug,
        "draft_slug": draft_slug,
    }
    yield data

    async with engine.begin() as conn:
        await conn.execute(
            text("DELETE FROM lesson_resources WHERE lesson_id = :l"),
            {"l": pub_lesson},
        )
        await conn.execute(
            text("DELETE FROM storage_objects WHERE id = :id"),
            {"id": storage_obj_id},
        )
        await conn.execute(
            text("DELETE FROM module_items WHERE module_id = ANY(:ids)"),
            {"ids": [pub_module, draft_module]},
        )
        await conn.execute(
            text("DELETE FROM lessons WHERE id = ANY(:ids)"),
            {"ids": [pub_lesson, draft_lesson]},
        )
        await conn.execute(
            text("DELETE FROM modules WHERE id = ANY(:ids)"),
            {"ids": [pub_module, draft_module]},
        )
        await conn.execute(
            text("DELETE FROM course_learning_outcomes WHERE course_id = :c"),
            {"c": pub_course},
        )
        await conn.execute(
            text("DELETE FROM courses WHERE id = ANY(:ids)"),
            {"ids": [pub_course, draft_course]},
        )


def test_router_metadata() -> None:
    paths = {(r.path, tuple(sorted(r.methods))) for r in learner_router.routes}  # type: ignore[attr-defined]
    assert ("/courses", ("GET",)) in paths
    assert ("/courses/by-slug/{slug}", ("GET",)) in paths
    assert ("/courses/{course_id}", ("GET",)) in paths
    assert ("/courses/{course_id}/content", ("GET",)) in paths
    assert ("/courses/{course_id}/tags", ("GET",)) in paths
    assert ("/courses/{course_id}/outcomes", ("GET",)) in paths
    assert ("/courses/{course_id}/modules", ("GET",)) in paths
    assert ("/modules/{module_id}", ("GET",)) in paths
    assert ("/modules/{module_id}/items", ("GET",)) in paths
    assert ("/modules/{module_id}/lessons", ("GET",)) in paths
    assert ("/lessons/{lesson_id}", ("GET",)) in paths
    assert ("/lessons/{lesson_id}/resources", ("GET",)) in paths
    assert (
        "/lesson-resources/{resource_id}/download-url",
        ("GET",),
    ) in paths
    me_paths = {(r.path, tuple(sorted(r.methods))) for r in me_courses_router.routes}  # type: ignore[attr-defined]
    assert ("/me/courses", ("GET",)) in me_paths


async def test_unauthenticated_returns_401(client: httpx.AsyncClient, scenario: dict) -> None:
    response = await client.get("/api/v1/courses")
    assert response.status_code == 401
    response = await client.get(f"/api/v1/courses/{scenario['pub_course']}")
    assert response.status_code == 401
    response = await client.get(
        f"/api/v1/lesson-resources/{scenario['resource_visible']}/download-url"
    )
    assert response.status_code == 401


async def test_list_courses_excludes_drafts(
    client: httpx.AsyncClient, student_bearer: str, scenario: dict
) -> None:
    response = await client.get(
        "/api/v1/courses", headers={"Authorization": f"Bearer {student_bearer}"}
    )
    assert response.status_code == 200, response.text
    body = response.json()
    statuses = {item["status"] for item in body["items"]}
    assert statuses == {"published"}
    ids = {item["id"] for item in body["items"]}
    assert str(scenario["pub_course"]) in ids
    assert str(scenario["draft_course"]) not in ids


async def test_owned_query_param_does_not_leak_authoring(
    client: httpx.AsyncClient, student_bearer: str, scenario: dict
) -> None:
    response = await client.get(
        "/api/v1/courses?owned=true",
        headers={"Authorization": f"Bearer {student_bearer}"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    statuses = {item["status"] for item in body["items"]}
    assert statuses == {"published"}
    assert str(scenario["draft_course"]) not in {it["id"] for it in body["items"]}


async def test_get_by_slug_404_for_draft(
    client: httpx.AsyncClient, student_bearer: str, scenario: dict
) -> None:
    response = await client.get(
        f"/api/v1/courses/by-slug/{scenario['draft_slug']}",
        headers={"Authorization": f"Bearer {student_bearer}"},
    )
    assert response.status_code == 404


async def test_get_by_slug_200_for_published(
    client: httpx.AsyncClient, student_bearer: str, scenario: dict
) -> None:
    response = await client.get(
        f"/api/v1/courses/by-slug/{scenario['pub_slug']}",
        headers={"Authorization": f"Bearer {student_bearer}"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["id"] == str(scenario["pub_course"])
    assert body["status"] == "published"
    assert body["slug"] == scenario["pub_slug"]


async def test_get_course_by_uuid_404_for_draft(
    client: httpx.AsyncClient, student_bearer: str, scenario: dict
) -> None:
    response = await client.get(
        f"/api/v1/courses/{scenario['draft_course']}",
        headers={"Authorization": f"Bearer {student_bearer}"},
    )
    assert response.status_code == 404


async def test_content_tree_excludes_draft_items_pointing_to_draft_targets(
    client: httpx.AsyncClient, student_bearer: str, scenario: dict
) -> None:
    response = await client.get(
        f"/api/v1/courses/{scenario['pub_course']}/content",
        headers={"Authorization": f"Bearer {student_bearer}"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    item_ids = {item["id"] for item in body.get("modules", [])}
    assert str(scenario["pub_module"]) in item_ids
    assert str(scenario["draft_module"]) not in item_ids


async def test_module_404_when_draft(
    client: httpx.AsyncClient, student_bearer: str, scenario: dict
) -> None:
    response = await client.get(
        f"/api/v1/modules/{scenario['draft_module']}",
        headers={"Authorization": f"Bearer {student_bearer}"},
    )
    assert response.status_code == 404


async def test_module_items_excludes_draft_targets(
    client: httpx.AsyncClient, student_bearer: str, scenario: dict
) -> None:
    response = await client.get(
        f"/api/v1/modules/{scenario['pub_module']}/items",
        headers={"Authorization": f"Bearer {student_bearer}"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    item_ids = {item["id"] for item in body}
    assert str(scenario["item_visible"]) in item_ids
    assert str(scenario["item_hidden"]) not in item_ids


async def test_lesson_404_when_draft(
    client: httpx.AsyncClient, student_bearer: str, scenario: dict
) -> None:
    response = await client.get(
        f"/api/v1/lessons/{scenario['draft_lesson']}",
        headers={"Authorization": f"Bearer {student_bearer}"},
    )
    assert response.status_code == 404


async def test_lesson_resources_excludes_invisible(
    client: httpx.AsyncClient, student_bearer: str, scenario: dict
) -> None:
    response = await client.get(
        f"/api/v1/lessons/{scenario['pub_lesson']}/resources",
        headers={"Authorization": f"Bearer {student_bearer}"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    ids = {r["id"] for r in body}
    assert str(scenario["resource_visible"]) in ids
    assert str(scenario["resource_hidden"]) not in ids


async def test_outcomes_returns_ordered_by_position(
    client: httpx.AsyncClient, student_bearer: str, scenario: dict
) -> None:
    response = await client.get(
        f"/api/v1/courses/{scenario['pub_course']}/outcomes",
        headers={"Authorization": f"Bearer {student_bearer}"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert [o["position"] for o in body] == [1, 2, 3]
    assert [o["outcome_text"] for o in body] == ["First", "Second", "Third"]


async def test_download_url_404_for_invisible_resource(
    client: httpx.AsyncClient, student_bearer: str, scenario: dict
) -> None:
    response = await client.get(
        f"/api/v1/lesson-resources/{scenario['resource_hidden']}/download-url",
        headers={"Authorization": f"Bearer {student_bearer}"},
    )
    assert response.status_code == 404


async def test_download_url_404_for_resource_without_storage(
    client: httpx.AsyncClient, student_bearer: str, scenario: dict
) -> None:
    response = await client.get(
        f"/api/v1/lesson-resources/{scenario['resource_no_storage']}/download-url",
        headers={"Authorization": f"Bearer {student_bearer}"},
    )
    assert response.status_code == 404


async def test_cursor_pagination_round_trip(
    client: httpx.AsyncClient,
    student_bearer: str,
    scenario: dict,
    engine: AsyncEngine,
    seeded_users: SeededUsers,
) -> None:
    extra = [uuid.uuid4() for _ in range(3)]
    async with engine.begin() as conn:
        for i, cid in enumerate(extra):
            await conn.execute(
                text(
                    "INSERT INTO courses "
                    "(id, organization_id, owner_user_id, slug, title, status) "
                    "VALUES (:id, :org, :owner, :slug, :title, 'published')"
                ),
                {
                    "id": cid,
                    "org": seeded_users.organization_id,
                    "owner": seeded_users.teacher_id,
                    "slug": f"page-{i}-{cid.hex[:6]}",
                    "title": f"Extra {i}",
                },
            )
    try:
        response = await client.get(
            "/api/v1/courses?limit=2",
            headers={"Authorization": f"Bearer {student_bearer}"},
        )
        assert response.status_code == 200, response.text
        page1 = response.json()
        assert len(page1["items"]) == 2
        assert page1["next_cursor"] is not None
        response = await client.get(
            f"/api/v1/courses?limit=2&cursor={page1['next_cursor']}",
            headers={"Authorization": f"Bearer {student_bearer}"},
        )
        assert response.status_code == 200, response.text
        page2 = response.json()
        ids_a = {it["id"] for it in page1["items"]}
        ids_b = {it["id"] for it in page2["items"]}
        assert ids_a.isdisjoint(ids_b)
    finally:
        async with engine.begin() as conn:
            await conn.execute(text("DELETE FROM courses WHERE id = ANY(:ids)"), {"ids": extra})


def test_no_authoring_imports_in_learner() -> None:
    learner_path = (
        Path(__file__).resolve().parent.parent.parent
        / "abridgeai"
        / "features"
        / "courses"
        / "routers"
        / "learner.py"
    )
    source = learner_path.read_text(encoding="utf-8")
    forbidden = re.compile(
        r"services\.(authoring|assignment|administration)|queries\.(authoring|assignment|administration)"
    )
    matches = forbidden.findall(source)
    assert matches == [], f"learner.py imports forbidden modules: {matches}"
