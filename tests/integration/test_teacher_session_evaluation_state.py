"""Teacher session projections must carry the server-derived evaluation state.

The teacher surfaces (per-config attempts list, cross-config teacher read,
authoring session detail) used to return only ``status`` + ``pass_verdict``,
which cannot answer "is a verdict still coming?". The frontend answered by
guessing — treating ``status='failed'`` as final — and:

* an ``abandoned`` attempt (never enqueued for evaluation) drove the GAP report
  page into a 60-retry / 3-minute 404 poll behind a full-screen spinner;
* a recoverable ``failed`` evaluation froze as "Evaluation failed" on teacher
  lists even though the recovery sweep re-drives exactly those rows.

Same contract the learner DTOs already pin (see
``tests/unit/interviews/test_public_evaluation_state.py``): the label is derived
server-side by ``derive_evaluation_state`` and the frontend must not re-derive
it. Detail parity: the authoring detail endpoint also returns
``assessment_started_at`` / onboarding / language — the fields the teacher GAP
page needs to classify report availability.
"""

from __future__ import annotations

import itertools
import json
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
from sqlalchemy import Column, Table, text
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

import abridgeai.core.db.all_models  # noqa: F401  -- full ORM registry (conftest parity)
import abridgeai.features.interviews.models  # noqa: F401  -- register interview tables
from abridgeai.core.config import get_settings
from abridgeai.core.db import Base, get_db
from abridgeai.core.security import create_access_token, generate_token, hash_secret
from abridgeai.features.interviews.routers import (
    authoring_router,
    authoring_sessions_router,
    learner_sessions_router,
)
from tests.support.db_graph import hard_delete_graph

for _stub_name in ("learning_materials", "learning_material_versions"):
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
async def admin_bearer(engine: AsyncEngine, seeded_users: SeededUsers) -> AsyncIterator[str]:
    """Same per-file fixture the sibling interview suite defines — an admin
    auth session whose token doubles as the bearer; torn down after."""
    sid = await _seed_auth_session(engine, seeded_users.admin_id)
    yield create_access_token(user_id=seeded_users.admin_id, session_id=sid)
    async with engine.begin() as conn:
        await conn.execute(text("DELETE FROM auth_sessions WHERE id = :id"), {"id": sid})


@pytest_asyncio.fixture
async def client(
    session_factory: async_sessionmaker[AsyncSession],
) -> AsyncIterator[httpx.AsyncClient]:
    arq_pool = AsyncMock()
    arq_pool.enqueue_job = AsyncMock()

    async def _override_get_db() -> AsyncIterator[AsyncSession]:
        async with session_factory() as session:
            yield session

    fastapi_app = FastAPI()
    fastapi_app.include_router(authoring_router, prefix="/api/v1")
    fastapi_app.include_router(authoring_sessions_router, prefix="/api/v1")
    fastapi_app.include_router(learner_sessions_router, prefix="/api/v1")

    async def _override_arq_pool() -> object:
        return arq_pool

    from abridgeai.features.interviews.routers.learner_sessions import (
        get_arq_pool as get_learner_sessions_arq_pool,
    )

    fastapi_app.dependency_overrides[get_db] = _override_get_db
    fastapi_app.dependency_overrides[get_learner_sessions_arq_pool] = _override_arq_pool
    http_client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=fastapi_app), base_url="http://test"
    )
    yield http_client
    await http_client.aclose()
    fastapi_app.dependency_overrides.clear()


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


# Distinct position pool — the seeded course's (course_id, position) unique key
# is shared with every other integration suite (sibling starts at 9000).
_MODULE_POSITIONS = itertools.count(9500)


async def _seed_auth_session(engine: AsyncEngine, user_id: uuid.UUID) -> uuid.UUID:
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


async def _seed_published_config(
    engine: AsyncEngine,
    *,
    course_id: uuid.UUID,
    module_id: uuid.UUID,
    actor_id: uuid.UUID,
) -> uuid.UUID:
    """Minimal published interview config (the sibling suite's proven shape)."""
    config_id = uuid.uuid4()
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO interview_configs "
                "(id, course_id, module_id, title, status, created_by, slug) "
                "VALUES (:id, :c, :m, 'Eval State Config', 'published', :u, "
                "'slug-' || uuid_generate_v4()::text)"
            ),
            {
                "id": config_id,
                "c": course_id,
                "m": module_id,
                "u": actor_id,
            },
        )
        # start_session refuses an outcome-less config (thesis §4.3).
        await conn.execute(
            text(
                "INSERT INTO interview_outcomes "
                "(id, interview_config_id, position, outcome_text, outcome_type, "
                " importance_weight, created_by) "
                "VALUES (:id, :cfg, 1, 'Understand recursion', 'knowledge', 3, :u)"
            ),
            {"id": uuid.uuid4(), "cfg": config_id, "u": actor_id},
        )
    return config_id


async def _seed_gradable_session(
    engine: AsyncEngine,
    *,
    session_id: uuid.UUID,
    status: str,
    attempts: int,
    pass_verdict: bool | None = None,
) -> None:
    """A terminal session whose recovery budget is at a known point.

    ``last_attempt_at`` is long ago so a session at the attempt ceiling
    describes a SETTLED final attempt — the ``exhausted`` shape. The claim
    columns stay NULL: a live lease would keep the row ``pending`` (that
    branch is pinned in the unit tests for ``derive_evaluation_state``).
    """
    summary = {
        "evaluation_recovery": {
            "attempts": attempts,
            "last_attempt_at": "2020-01-01T00:00:00+00:00",
        }
    }
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "UPDATE interview_sessions "
                "SET status = :status, "
                "    pass_verdict = :pass_verdict, "
                "    onboarding_stage = 'completed', "
                "    assessment_started_at = NOW(), "
                "    ended_at = NOW(), "
                "    internal_summary_json = CAST(:summary AS jsonb) "
                "WHERE id = :id"
            ),
            {
                "status": status,
                "pass_verdict": pass_verdict,
                "summary": json.dumps(summary),
                "id": str(session_id),
            },
        )


class TeacherSessionScenario:
    """One started session + the tokens to read it from both sides."""

    def __init__(
        self,
        *,
        client: httpx.AsyncClient,
        engine: AsyncEngine,
        admin_bearer: str,
        student_token: str,
        config_id: uuid.UUID,
        session_id: str,
        auth_session_id: uuid.UUID,
    ) -> None:
        self.client = client
        self.engine = engine
        self.admin_bearer = admin_bearer
        self.student_token = student_token
        self.config_id = config_id
        self.session_id = session_id
        self.auth_session_id = auth_session_id


@pytest_asyncio.fixture
async def teacher_session(
    client: httpx.AsyncClient,
    engine: AsyncEngine,
    admin_bearer: str,
    seeded_users: SeededUsers,
) -> AsyncIterator[TeacherSessionScenario]:
    """Published config + one in-progress student attempt, torn down after."""
    module_id = uuid.uuid4()
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO modules (id, course_id, title, position, status) "
                "VALUES (:m, :c, 'Eval State Module', :pos, 'draft')"
            ),
            {"m": module_id, "c": seeded_users.course_id, "pos": next(_MODULE_POSITIONS)},
        )
        await conn.execute(
            text(
                "INSERT INTO course_enrollments (course_id, student_id, status, source) "
                "VALUES (:c, :s, 'active', 'manager_bulk') "
                "ON CONFLICT (course_id, student_id) DO NOTHING"
            ),
            {"c": seeded_users.course_id, "s": seeded_users.student_id},
        )

    config_id = await _seed_published_config(
        engine,
        course_id=seeded_users.course_id,
        module_id=module_id,
        actor_id=seeded_users.admin_id,
    )
    auth_sid = await _seed_auth_session(engine, seeded_users.student_id)
    student_token = create_access_token(user_id=seeded_users.student_id, session_id=auth_sid)
    start = await client.post(
        f"/api/v1/interview-configs/{config_id}/sessions",
        json={"input_mode": "text"},
        headers=_auth(student_token),
    )
    assert start.status_code == 201, start.text

    scenario = TeacherSessionScenario(
        client=client,
        engine=engine,
        admin_bearer=admin_bearer,
        student_token=student_token,
        config_id=config_id,
        session_id=start.json()["session_id"],
        auth_session_id=auth_sid,
    )
    yield scenario

    async with engine.begin() as conn:
        await conn.execute(
            text("DELETE FROM interview_sessions WHERE id = :id"),
            {"id": scenario.session_id},
        )
        await conn.execute(
            text("DELETE FROM auth_sessions WHERE id = :id"),
            {"id": auth_sid},
        )
        await hard_delete_graph(conn, "modules", [str(module_id)])


@pytest.mark.asyncio
async def test_failed_session_with_budget_left_is_pending_on_teacher_list(
    teacher_session: TeacherSessionScenario,
    seeded_users: SeededUsers,
) -> None:
    """``status='failed'`` + recovery budget left → ``pending``, NOT a verdict
    that never came: the sweep re-drives exactly these rows."""
    await _seed_gradable_session(
        teacher_session.engine,
        session_id=uuid.UUID(teacher_session.session_id),
        status="failed",
        attempts=0,
    )
    resp = await teacher_session.client.get(
        f"/api/v1/teacher/interview-configs/{teacher_session.config_id}/sessions",
        headers=_auth(teacher_session.admin_bearer),
    )
    assert resp.status_code == 200, resp.text
    row = next(r for r in resp.json() if r["session_id"] == teacher_session.session_id)
    assert row["evaluation_state"] == "pending"


@pytest.mark.asyncio
async def test_failed_session_with_spent_budget_is_exhausted_on_teacher_detail(
    teacher_session: TeacherSessionScenario,
) -> None:
    """The terminal no-report shape: budget spent AND a terminal phase record —
    ``exhausted``. The GAP page reads this and stops immediately."""
    session_id = uuid.UUID(teacher_session.session_id)
    await _seed_gradable_session(
        teacher_session.engine,
        session_id=session_id,
        status="failed",
        attempts=3,
    )
    async with teacher_session.engine.begin() as conn:
        await conn.execute(
            text(
                "UPDATE interview_sessions "
                "SET internal_summary_json = jsonb_set("
                "    internal_summary_json, "
                "    '{evaluation_recovery,current}', "
                "    CAST(:phase AS jsonb)) "
                "WHERE id = :id"
            ),
            {
                "phase": json.dumps(
                    {
                        "attempt": 3,
                        "job_id": "j-final",
                        "phase": "missing",
                        "dispatched_at": "2020-01-01T00:00:00+00:00",
                    }
                ),
                "id": str(session_id),
            },
        )
    resp = await teacher_session.client.get(
        f"/api/v1/teacher/interview-sessions/{teacher_session.session_id}",
        headers=_auth(teacher_session.admin_bearer),
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["evaluation_state"] == "exhausted"
    # Availability fields the GAP page needs alongside the state label.
    assert body["assessment_started_at"] is not None
    assert body["onboarding_stage"] == "completed"
    assert body["interview_language"] in ("en", "vi")


@pytest.mark.asyncio
async def test_abandoned_session_is_not_required_on_teacher_detail(
    teacher_session: TeacherSessionScenario,
) -> None:
    """The production shape behind the infinite spinner: status='abandoned',
    never assessed → ``not_required``. The GAP page can render its empty state
    immediately instead of polling a report that will never exist."""
    async with teacher_session.engine.begin() as conn:
        await conn.execute(
            text(
                "UPDATE interview_sessions "
                "SET status = 'abandoned', ended_at = NOW() "
                "WHERE id = :id"
            ),
            {"id": teacher_session.session_id},
        )
    resp = await teacher_session.client.get(
        f"/api/v1/teacher/interview-sessions/{teacher_session.session_id}",
        headers=_auth(teacher_session.admin_bearer),
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["evaluation_state"] == "not_required"
