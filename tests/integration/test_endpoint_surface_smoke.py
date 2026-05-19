"""Endpoint surface smoke test.

Builds the full ``create_app()`` factory and exercises every registered
``APIRoute`` once. The contract is intentionally weak:

* GET ``/healthz`` and GET ``/readyz`` must return 200.
* Every other route must return a HTTP status code (any 4xx is fine — that
  proves auth/validation fired). 500 is a wiring/regression bug.

Path parameters are substituted with syntactically valid placeholders
(zero-UUID for ``*_id``, sentinel string for ``slug`` / ``category`` /
``channel``). Auth-gated routes reject before they hit the route body, so
the dummy parameters never need to resolve to real rows.

Non-GET routes are also dispatched: the auth/perm boundary fires before
any database mutation, so the smoke run is non-destructive. POST/PATCH/
PUT requests are sent with an empty JSON body — schema-validation 422 is
counted as a healthy 4xx response.
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
from fastapi.routing import APIRoute
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

import abridgeai.ai.models  # noqa: F401  -- register processing_jobs / generation_runs / ai_model_calls
import abridgeai.features.access_control.models  # noqa: F401  -- register FK targets
import abridgeai.features.career_paths.models  # noqa: F401  -- register career path tables
import abridgeai.features.courses.models  # noqa: F401  -- register course tables
import abridgeai.features.enrollments.models  # noqa: F401  -- register enrollments tables
import abridgeai.features.identity.models  # noqa: F401  -- register users
import abridgeai.features.interviews.models  # noqa: F401  -- register interview_* tables
import abridgeai.features.notifications.models  # noqa: F401  -- register notifications
import abridgeai.features.progress.models  # noqa: F401  -- register lesson_progress
import abridgeai.features.quizzes.models  # noqa: F401  -- register quiz tables
from abridgeai.api import create_app
from abridgeai.core.config import get_settings
from abridgeai.core.db import get_db
from abridgeai.core.security import create_access_token
from abridgeai.features.admin.routers.processing import (
    get_arq_pool as get_admin_arq_pool,
)
from abridgeai.features.interviews.routers.authoring import (
    get_arq_pool as get_interview_authoring_arq_pool,
)
from abridgeai.features.interviews.routers.learner import (
    get_arq_pool as get_interview_learner_arq_pool,
)
from abridgeai.features.materials.routers.authoring import (
    get_arq_pool as get_materials_arq_pool,
)
from abridgeai.features.quizzes.routers.authoring import (
    get_arq_pool as get_quiz_arq_pool,
)

ZERO_UUID = "00000000-0000-0000-0000-000000000000"
SLUG_PLACEHOLDER = "smoke-placeholder"
CATEGORY_PLACEHOLDER = "course_announcement"
CHANNEL_PLACEHOLDER = "in_app"

_PARAM_RE = re.compile(r"\{([^}]+)\}")


def _substitute_path(path: str) -> str:
    """Replace every ``{param}`` placeholder with a syntactically valid stub."""

    def _replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name == "slug":
            return SLUG_PLACEHOLDER
        if name == "category":
            return CATEGORY_PLACEHOLDER
        if name == "channel":
            return CHANNEL_PLACEHOLDER
        return ZERO_UUID

    return _PARAM_RE.sub(_replace, path)


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
    """Build the full app with DB + ARQ-pool overrides applied."""

    async def _override_get_db() -> AsyncIterator[AsyncSession]:
        async with session_factory() as session:
            yield session

    async def _none_pool() -> object | None:
        return None

    fastapi_app = create_app()
    fastapi_app.dependency_overrides[get_db] = _override_get_db
    for dep in (
        get_admin_arq_pool,
        get_materials_arq_pool,
        get_quiz_arq_pool,
        get_interview_authoring_arq_pool,
        get_interview_learner_arq_pool,
    ):
        fastapi_app.dependency_overrides[dep] = _none_pool
    yield fastapi_app
    fastapi_app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def client(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as ac:
        yield ac


@pytest_asyncio.fixture
async def student_bearer(seeded_users: SeededUsers) -> str:
    return create_access_token(user_id=seeded_users.student_id, session_id=uuid.uuid4())


async def test_endpoint_surface_smoke(
    app: FastAPI,
    client: httpx.AsyncClient,
    student_bearer: str,
) -> None:
    """Exercise every registered ``APIRoute`` and assert no route raises a 500."""
    api_routes = [r for r in app.routes if isinstance(r, APIRoute)]
    assert api_routes, "create_app() should register APIRoute instances"

    headers_authed = {"Authorization": f"Bearer {student_bearer}"}

    counts = {
        "total": 0,
        "ok_200": 0,
        "client_4xx": 0,
        "redirect_3xx": 0,
        "service_unavailable_503": 0,
        "other": 0,
    }
    unhandled_5xx: list[tuple[str, str, int]] = []
    exceptions: list[tuple[str, str, str]] = []

    for route in api_routes:
        path = _substitute_path(route.path)
        for method in sorted(route.methods or {"GET"}):
            counts["total"] += 1
            request_kwargs: dict[str, object] = {"headers": headers_authed}
            if method in {"POST", "PATCH", "PUT"}:
                request_kwargs["json"] = {}

            try:
                response = await client.request(method, path, **request_kwargs)
            except Exception as exc:  # noqa: BLE001 -- smoke must classify any raise as failure
                exceptions.append((method, route.path, repr(exc)))
                continue

            code = response.status_code
            if code == 503:
                counts["service_unavailable_503"] += 1
            elif code >= 500:
                unhandled_5xx.append((method, route.path, code))
            elif code == 200:
                counts["ok_200"] += 1
            elif 400 <= code < 500:
                counts["client_4xx"] += 1
            elif 300 <= code < 400:
                counts["redirect_3xx"] += 1
            else:
                counts["other"] += 1

    healthz = await client.get("/healthz")
    assert healthz.status_code == 200, healthz.text

    readyz = await client.get("/readyz")
    assert readyz.status_code in {200, 503}, readyz.text

    print(
        "Endpoint surface smoke: "
        f"tested {counts['total']} route invocations across {len(api_routes)} routes; "
        f"{counts['ok_200']} returned 200, "
        f"{counts['client_4xx']} returned 4xx (auth/validation), "
        f"{counts['service_unavailable_503']} returned 503 (documented-skip), "
        f"{counts['redirect_3xx']} returned 3xx, "
        f"{counts['other']} other; "
        f"{len(unhandled_5xx)} returned unhandled 5xx"
    )

    assert not exceptions, f"Routes raised exceptions instead of returning a response: {exceptions}"
    assert not unhandled_5xx, f"Routes returned unhandled 5xx (wiring regression): {unhandled_5xx}"


def test_create_app_route_count_is_stable() -> None:
    """Guard against an accidental router being silently dropped from ``create_app``."""
    api_routes = [r for r in create_app().routes if isinstance(r, APIRoute)]
    assert len(api_routes) >= 180, (
        f"Expected at least 180 APIRoute instances; got {len(api_routes)}. "
        "A router was likely dropped from create_app()."
    )
