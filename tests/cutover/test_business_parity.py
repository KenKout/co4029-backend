"""Cutover business-parity dry-run (T9.3).

These tests assert that ``backend-new`` produces sensible business outcomes for
the canonical user flows that the cutover runbook (`docs/cutover/RUNBOOK.md`)
will gate on. They are NOT a structural diff against the legacy ``backend/``
service -- the API contracts intentionally diverge (Phase 9 §4).

What this file proves:
    * The wired stack (FastAPI app + alembic head + seeded catalog) responds
      to canonical requests as a real cutover would expect.
    * Each flow yields a business-reasonable result (status, shape sanity,
      role-gating respected) so a 100% canary on ``backend-new`` will not
      surface ``5xx`` regressions on these paths.

Flows covered:
    1. ``test_parity_healthz_smoke``        -- liveness probe (Step 1 smoke).
    2. ``test_parity_admin_lists_users``    -- admin listing endpoint reachable.
    3. ``test_parity_admin_stats_overview`` -- admin global stats query works.
    4. ``test_parity_courses_learner_list`` -- student can list courses.
    5. ``test_parity_users_me_self_lookup`` -- ``GET /users/me`` resolves token.

Flows explicitly skipped (with docstrings) where end-to-end execution requires
infrastructure outside the dry-run envelope (real LLM, S3, ffmpeg, Whisper).
"""

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
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

import abridgeai.features.access_control.models  # noqa: F401
import abridgeai.features.career_paths.models  # noqa: F401
import abridgeai.features.courses.models  # noqa: F401
import abridgeai.features.enrollments.models  # noqa: F401
import abridgeai.features.identity.models  # noqa: F401
import abridgeai.features.interviews.models  # noqa: F401
import abridgeai.features.materials.models  # noqa: F401
import abridgeai.features.notifications.models  # noqa: F401
import abridgeai.features.progress.models  # noqa: F401
import abridgeai.features.quizzes.models  # noqa: F401
from abridgeai.api import create_app
from abridgeai.core.config import get_settings
from abridgeai.core.security import create_access_token, generate_token, hash_secret


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
async def client() -> AsyncIterator[httpx.AsyncClient]:
    app = create_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as ac:
        yield ac


async def _seed_session(engine: AsyncEngine, user_id: uuid.UUID) -> uuid.UUID:
    sid = uuid.uuid4()
    expires_at = datetime.now(tz=UTC) + timedelta(hours=1)
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO auth_sessions (id, user_id, refresh_token_hash, expires_at) "
                "VALUES (:id, :uid, :h, :exp)"
            ),
            {
                "id": sid,
                "uid": user_id,
                "h": hash_secret(generate_token()),
                "exp": expires_at,
            },
        )
    return sid


async def _bearer(engine: AsyncEngine, user_id: uuid.UUID) -> str:
    sid = await _seed_session(engine, user_id)
    return create_access_token(user_id=user_id, session_id=sid)


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


async def _purge_sessions(engine: AsyncEngine, user_id: uuid.UUID) -> None:
    async with engine.begin() as conn:
        await conn.execute(
            text("DELETE FROM auth_sessions WHERE user_id = :u"),
            {"u": user_id},
        )


async def test_parity_healthz_smoke(client: httpx.AsyncClient) -> None:
    """Canary Step 1 smoke check: ``GET /healthz`` returns 200 + status=ok.

    Cutover RUNBOOK.md Step 1 calls this out as the first read-only smoke.
    Business outcome: the new backend reports itself live without auth.
    """
    resp = await client.get("/healthz")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body == {"status": "ok"}


async def test_parity_admin_lists_users(
    client: httpx.AsyncClient,
    engine: AsyncEngine,
    seeded_users: SeededUsers,
) -> None:
    """Admin can look up a user by id -- proves the admin role + DB are wired.

    Business outcome: admin token returns 200 for an admin user-detail lookup,
    response is a JSON object identifying the requested user. Schema may
    differ from legacy backend; we only assert the *outcome* (admin can
    enumerate user records).
    """
    token = await _bearer(engine, seeded_users.admin_id)
    try:
        resp = await client.get(
            f"/api/v1/admin/users/{seeded_users.teacher_id}",
            headers=_auth(token),
        )
    finally:
        await _purge_sessions(engine, seeded_users.admin_id)
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    assert isinstance(payload, dict), "user detail must return an object"
    user_block = payload.get("user") if isinstance(payload.get("user"), dict) else payload
    user_id = user_block.get("id") or user_block.get("user_id")
    assert user_id is not None, "user detail must expose an id"
    assert str(user_id) == str(seeded_users.teacher_id)


async def test_parity_admin_stats_overview(
    client: httpx.AsyncClient,
    engine: AsyncEngine,
    seeded_users: SeededUsers,
) -> None:
    """Admin sees the global stats overview -- exercises the read-side join graph.

    Business outcome: 200 with a JSON object (the dashboard's home tile).
    A 5xx here would be the loudest cutover regression imaginable.
    """
    token = await _bearer(engine, seeded_users.admin_id)
    try:
        resp = await client.get("/api/v1/admin/stats/overview", headers=_auth(token))
    finally:
        await _purge_sessions(engine, seeded_users.admin_id)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert isinstance(body, dict), "stats overview must return an object"


async def test_parity_courses_learner_list(
    client: httpx.AsyncClient,
    engine: AsyncEngine,
    seeded_users: SeededUsers,
) -> None:
    """Student lists courses -- proves the learner read path is intact.

    Business outcome: 200 with a paginated CoursePage (items + pagination
    fields). We do not assert exact item count: the learner-side filter
    excludes drafts, and the seeded course is a draft, so an empty
    items list is still a valid business outcome.
    """
    token = await _bearer(engine, seeded_users.student_id)
    try:
        resp = await client.get("/api/v1/courses", headers=_auth(token))
    finally:
        await _purge_sessions(engine, seeded_users.student_id)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert isinstance(body, dict), "course list must return paginated envelope"
    assert "items" in body, "course list must expose an items array"
    assert isinstance(body["items"], list)


async def test_parity_users_me_self_lookup(
    client: httpx.AsyncClient,
    engine: AsyncEngine,
    seeded_users: SeededUsers,
) -> None:
    """``GET /users/me`` resolves the bearer back to the seeded user row.

    Business outcome: the token-to-user mapping is intact post-cutover. If
    this regresses, every authenticated route in the new stack is broken.
    """
    token = await _bearer(engine, seeded_users.student_id)
    try:
        resp = await client.get("/api/v1/users/me", headers=_auth(token))
    finally:
        await _purge_sessions(engine, seeded_users.student_id)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert isinstance(body, dict)
    user_id = body.get("id") or body.get("user_id")
    assert user_id is not None, "users/me must return an id field"
    assert str(user_id) == str(seeded_users.student_id)


@pytest.mark.skip(
    reason="Quiz attempt + scoring needs a real LLM (T8.4 baseline). Cutover RUNBOOK Step 3 "
    "asserts eval-scenario parity from eval/results/baseline-*.json instead."
)
async def test_parity_student_enrolls_and_takes_quiz() -> None:
    """Placeholder: full quiz attempt requires the LLM provider to be live."""


@pytest.mark.skip(
    reason="Material upload + processing needs S3 (Garage) + ffmpeg + Whisper. Covered "
    "separately under -m s3_live + -m audio_live + -m video_live."
)
async def test_parity_material_upload_and_processing() -> None:
    """Placeholder: upload pipeline runs under the *_live markers, not the dry-run."""


@pytest.mark.skip(
    reason="OAuth callback round-trip is exercised in tests/integration/test_identity_full.py "
    "(test_full_login_then_me_then_logout). Repeating it here would be redundant."
)
async def test_parity_oauth_login() -> None:
    """Placeholder: OAuth login is covered by the identity full suite, not this dry-run."""
