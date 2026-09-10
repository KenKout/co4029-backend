"""Integrity-event ingest: scoring, one-shot warning, idempotent retries.

These tests drive the REAL route (``create_app``) against the real test DB so
the response contract cannot drift from what the server actually computes:

* scored signals accumulate the session's FROZEN policy weights;
* the first crossing returns ``warning_issued=true`` exactly once, flags the
  session, and appends a server-generated ``warning_issued`` evidence event
  with the score/threshold metadata;
* a replayed batch (same ``client_event_id``) is idempotent: no double score,
  no second warning;
* reconnect/disconnect never score;
* a terminal session drops the batch (accepted=0, score untouched);
* the server's decision values come from the snapshot, not from anything the
  client can forge (client ``metadata`` and ``severity`` never influence it).

The scenario fixture hangs each module off a fresh position (``_MODULE_POSITIONS``)
on the shared seeded course — see the fixture comments for why.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock

import httpx
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

from abridgeai.core.config import get_settings
from abridgeai.core.security import create_access_token
from abridgeai.features.interviews.routers.learner import router as learner_router
from abridgeai.features.interviews.routers.learner_sessions import (
    router as learner_sessions_router,
)
from tests.support.db_graph import hard_delete_graph


def _async_url(database_url: str) -> str:
    if database_url.startswith("postgresql+psycopg://"):
        return database_url.replace("postgresql+psycopg://", "postgresql+psycopg_async://", 1)
    if database_url.startswith("postgresql://"):
        return database_url.replace("postgresql://", "postgresql+psycopg_async://", 1)
    return database_url


@pytest_asyncio.fixture
async def engine() -> AsyncIterator[AsyncEngine]:
    eng = create_async_engine(_async_url(get_settings().database_url), pool_pre_ping=True)
    yield eng
    await eng.dispose()


@pytest_asyncio.fixture
async def session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False, autoflush=False)


@pytest_asyncio.fixture
async def app(
    session_factory: async_sessionmaker[AsyncSession],
) -> AsyncIterator[tuple[FastAPI, AsyncMock]]:
    """Learner-session routers only — the integrity endpoint's slice."""
    from abridgeai.core.db import get_db

    arq_pool = AsyncMock()
    arq_pool.enqueue_job = AsyncMock()

    async def _override_get_db() -> AsyncIterator[AsyncSession]:
        async with session_factory() as session:
            yield session

    fastapi_app = FastAPI()
    fastapi_app.include_router(learner_router, prefix="/api/v1")
    fastapi_app.include_router(learner_sessions_router, prefix="/api/v1")
    fastapi_app.dependency_overrides[get_db] = _override_get_db
    yield fastapi_app, arq_pool
    fastapi_app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def client(app: tuple[FastAPI, AsyncMock]) -> AsyncIterator[httpx.AsyncClient]:
    fastapi_app, _ = app
    transport = httpx.ASGITransport(app=fastapi_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as ac:
        yield ac


async def _seed_integrity_config(
    engine: AsyncEngine,
    *,
    course_id: uuid.UUID,
    module_id: uuid.UUID,
    actor_id: uuid.UUID,
    weights: tuple[int, int, int] = (3, 1, 2),
    threshold: int = 3,
    custom_refusal_en: str | None = None,
) -> uuid.UUID:
    """Published config carrying an explicit integrity policy + one question."""
    config_id = uuid.uuid4()
    question_id = uuid.uuid4()
    outcome_id = uuid.uuid4()
    w_tab, w_focus, w_full = weights
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO interview_configs (id, course_id, module_id, title, status, "
                " created_by, slug, security_custom_refusal_en, "
                " integrity_weight_tab_switch, integrity_weight_focus_lost, "
                " integrity_weight_fullscreen_exit, integrity_score_threshold) "
                "VALUES (:id, :c, :m, 'Integrity policy', 'published', :u, "
                " 'slug-' || uuid_generate_v4()::text, :refusal, :wt, :wf, :wo, :th)"
            ),
            {
                "id": config_id,
                "c": course_id,
                "m": module_id,
                "u": actor_id,
                "refusal": custom_refusal_en,
                "wt": w_tab,
                "wf": w_focus,
                "wo": w_full,
                "th": threshold,
            },
        )
        await conn.execute(
            text(
                "INSERT INTO interview_questions "
                "(id, interview_config_id, position, question_type, prompt_text, "
                " review_status, ai_generated, source_refs_json, created_by) "
                "VALUES (:id, :cfg, 1, 'conceptual', 'What is normalization?', "
                "        'approved', false, '[]'::jsonb, :u)"
            ),
            {"id": question_id, "cfg": config_id, "u": actor_id},
        )
        await conn.execute(
            text(
                "INSERT INTO interview_outcomes "
                "(id, interview_config_id, position, outcome_text, outcome_type, "
                " importance_weight, created_by) "
                "VALUES (:id, :cfg, 1, 'Explain normalization', 'knowledge', 3, :u)"
            ),
            {"id": outcome_id, "cfg": config_id, "u": actor_id},
        )
    return config_id


async def _start_session(
    client: httpx.AsyncClient,
    config_id: uuid.UUID,
    token: str,
) -> tuple[uuid.UUID, dict]:
    resp = await client.post(
        f"/api/v1/interview-configs/{config_id}/sessions",
        json={"input_mode": "text"},
        headers=_auth(token),
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    return uuid.UUID(body["session_id"]), body


async def _post_batch(
    client: httpx.AsyncClient,
    session_id: uuid.UUID,
    token: str,
    events: list[dict],
) -> httpx.Response:
    return await client.post(
        f"/api/v1/interview-sessions/{session_id}/integrity-events",
        json={"events": events},
        headers=_auth(token),
    )


async def _seed_auth_session(engine: AsyncEngine, user_id: uuid.UUID) -> uuid.UUID:
    """One live auth_sessions row so signed tokens pass the security dep."""
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
            {"id": session_id, "uid": user_id, "h": hash_secret(generate_token()), "exp": expires_at},
        )
    return session_id


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


@pytest_asyncio.fixture
async def integrity_scenario(
    engine: AsyncEngine,
    seeded_users: SeededUsers,
) -> AsyncIterator[dict]:
    """A published config + a session-started student token, torn down safely."""
    module_id = uuid.uuid4()
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO modules (id, course_id, title, position, status) "
                "VALUES (:m, :c, 'Integrity Module', :pos, 'draft')"
            ),
            {"m": module_id, "c": seeded_users.course_id, "pos": 9400 + uuid.uuid4().int % 100},
        )
        await conn.execute(
            text(
                "INSERT INTO course_enrollments (course_id, student_id, status, source) "
                "VALUES (:c, :s, 'active', 'manager_bulk') "
                "ON CONFLICT (course_id, student_id) DO NOTHING"
            ),
            {"c": seeded_users.course_id, "s": seeded_users.student_id},
        )
    config_id = await _seed_integrity_config(
        engine,
        course_id=seeded_users.course_id,
        module_id=module_id,
        actor_id=seeded_users.admin_id,
    )
    yield {
        "config_id": config_id,
        "module_id": module_id,
        "course_id": seeded_users.course_id,
    }
    async with engine.begin() as conn:
        await hard_delete_graph(conn, "modules", [str(module_id)])


@pytest_asyncio.fixture
async def student_token(
    engine: AsyncEngine, seeded_users: SeededUsers
) -> AsyncIterator[str]:
    sid = await _seed_auth_session(engine, seeded_users.student_id)
    yield create_access_token(user_id=seeded_users.student_id, session_id=sid)
    async with engine.begin() as conn:
        await conn.execute(text("DELETE FROM auth_sessions WHERE id = :id"), {"id": sid})


async def test_tab_switch_weight_three_crosses_default_threshold(
    client: httpx.AsyncClient,
    engine: AsyncEngine,
    integrity_scenario: dict,
    student_token: str,
) -> None:
    token = student_token
    session_id, _ = await _start_session(client, integrity_scenario["config_id"], token)

    first = await _post_batch(
        client,
        session_id,
        token,
        [{"event_type": "focus_lost", "severity": "info"}],
    )
    assert first.status_code == 202, first.text
    body = first.json()
    assert body["accepted"] == 1
    assert body["integrity_score"] == 1
    assert body["integrity_score_threshold"] == 3
    assert body["warning_issued"] is False

    crossing = await _post_batch(
        client,
        session_id,
        token,
        [
            {"event_type": "focus_lost", "severity": "info"},
            {"event_type": "focus_lost", "severity": "info"},
        ],
    )
    assert crossing.status_code == 202
    body = crossing.json()
    assert body["integrity_score"] == 3
    assert body["warning_issued"] is True

    # Session row carries the flag + score durably.
    async with engine.begin() as conn:
        row = (
            await conn.execute(
                text(
                    "SELECT integrity_score, integrity_warning_issued, "
                    " session_security_flagged, integrity_threshold_flagged_at "
                    "FROM interview_sessions WHERE id = :sid"
                ),
                {"sid": session_id},
            )
        ).mappings().one()
    assert row["integrity_score"] == 3
    assert row["integrity_warning_issued"] is True
    assert row["session_security_flagged"] is True
    assert row["integrity_threshold_flagged_at"] is not None

    # Exactly one server warning_issued evidence event exists.
    async with engine.begin() as conn:
        warnings = (
            await conn.execute(
                text(
                    "SELECT metadata_json FROM assessment_integrity_events "
                    "WHERE interview_session_id = :sid AND event_type = 'warning_issued'"
                ),
                {"sid": session_id},
            )
        ).scalars().all()
    assert len(warnings) == 1
    meta = dict(warnings[0])
    assert meta["integrity_score_after"] == 3
    assert meta["integrity_score_threshold"] == 3
    assert meta["integrity_weight_tab_switch"] == 3
    assert meta["integrity_weight_focus_lost"] == 1
    assert meta["integrity_weight_fullscreen_exit"] == 2


async def test_single_tab_switch_crosses_and_later_batches_never_re_warn(
    client: httpx.AsyncClient,
    engine: AsyncEngine,
    integrity_scenario: dict,
    student_token: str,
) -> None:
    token = student_token
    session_id, _ = await _start_session(client, integrity_scenario["config_id"], token)

    crossing = await _post_batch(
        client, session_id, token, [{"event_type": "tab_switch", "severity": "warning"}]
    )
    body = crossing.json()
    assert body["integrity_score"] == 3
    assert body["warning_issued"] is True

    later = await _post_batch(
        client, session_id, token, [{"event_type": "tab_switch", "severity": "warning"}]
    )
    body = later.json()
    assert body["integrity_score"] == 6
    assert body["warning_issued"] is False  # exactly-once contract

    async with engine.begin() as conn:
        count = (
            await conn.execute(
                text(
                    "SELECT count(*) FROM assessment_integrity_events "
                    "WHERE interview_session_id = :sid AND event_type = 'warning_issued'"
                ),
                {"sid": session_id},
            )
        ).scalar_one()
    assert count == 1


async def test_duplicate_client_event_id_is_idempotent(
    client: httpx.AsyncClient,
    engine: AsyncEngine,
    integrity_scenario: dict,
    student_token: str,
) -> None:
    token = student_token
    session_id, _ = await _start_session(client, integrity_scenario["config_id"], token)
    event_key = str(uuid.uuid4())

    first = await _post_batch(
        client,
        session_id,
        token,
        [{"event_type": "tab_switch", "severity": "warning", "metadata": {"client_event_id": event_key}}],
    )
    assert first.json()["integrity_score"] == 3

    # The reporter retries after a network loss: same batch, same client key.
    retry = await _post_batch(
        client,
        session_id,
        token,
        [{"event_type": "tab_switch", "severity": "warning", "metadata": {"client_event_id": event_key}}],
    )
    assert retry.status_code == 202
    body = retry.json()
    assert body["integrity_score"] == 3  # NOT 6 — the retry was absorbed

    async with engine.begin() as conn:
        count = (
            await conn.execute(
                text(
                    "SELECT count(*) FROM assessment_integrity_events "
                    "WHERE interview_session_id = :sid AND event_type = 'tab_switch'"
                ),
                {"sid": session_id},
            )
        ).scalar_one()
    assert count == 1


async def test_connectivity_events_are_recorded_but_never_score(
    client: httpx.AsyncClient,
    integrity_scenario: dict,
    student_token: str,
) -> None:
    token = student_token
    session_id, _ = await _start_session(client, integrity_scenario["config_id"], token)

    resp = await _post_batch(
        client,
        session_id,
        token,
        [
            {"event_type": "reconnect", "severity": "info"},
            {"event_type": "disconnect", "severity": "info"},
        ],
    )
    assert resp.status_code == 202
    body = resp.json()
    assert body["accepted"] == 2
    assert body["integrity_score"] == 0
    assert body["warning_issued"] is False


async def test_client_cannot_forge_the_warning_contract(
    client: httpx.AsyncClient,
    engine: AsyncEngine,
    integrity_scenario: dict,
    student_token: str,
) -> None:
    """Client-side ``warning_issued`` events / metadata never move the score."""
    token = student_token
    session_id, _ = await _start_session(client, integrity_scenario["config_id"], token)

    resp = await _post_batch(
        client,
        session_id,
        token,
        [
            {
                "event_type": "warning_issued",  # server-only type, sent by client
                "severity": "critical",
                "metadata": {"forged": True},
            }
        ],
    )
    assert resp.status_code == 202
    body = resp.json()
    assert body["warning_issued"] is False
    assert body["integrity_score"] == 0

    async with engine.begin() as conn:
        flagged = (
            await conn.execute(
                text("SELECT integrity_warning_issued FROM interview_sessions WHERE id = :sid"),
                {"sid": session_id},
            )
        ).scalar_one()
    assert flagged is False


async def test_terminal_session_drops_the_batch(
    client: httpx.AsyncClient,
    engine: AsyncEngine,
    integrity_scenario: dict,
    student_token: str,
) -> None:
    token = student_token
    session_id, _ = await _start_session(client, integrity_scenario["config_id"], token)
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "UPDATE interview_sessions SET status = 'completed', ended_at = NOW() "
                "WHERE id = :sid"
            ),
            {"sid": session_id},
        )

    resp = await _post_batch(
        client, session_id, token, [{"event_type": "tab_switch", "severity": "warning"}]
    )
    assert resp.status_code == 202
    body = resp.json()
    assert body["accepted"] == 0
    assert body["integrity_score"] == 0
    assert body["warning_issued"] is False


async def test_session_snapshot_freezes_the_policy(
    client: httpx.AsyncClient,
    integrity_scenario: dict,
    engine: AsyncEngine,
    student_token: str,
) -> None:
    """The session is scored under the weights frozen at start, not live config."""
    token = student_token
    config_id = integrity_scenario["config_id"]
    session_id, _ = await _start_session(client, config_id, token)

    # Teacher edits the (published ⇒ frozen, but prove the snapshot regardless)
    # config AFTER the session started; drift must not re-score the session.
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "UPDATE interview_configs SET integrity_weight_focus_lost = 5 "
                "WHERE id = :cid"
            ),
            {"cid": config_id},
        )

    resp = await _post_batch(
        client, session_id, token, [{"event_type": "focus_lost", "severity": "info"}]
    )
    # Snapshot weight 1 applies, not the edited 5.
    assert resp.json()["integrity_score"] == 1


async def test_start_session_persists_the_snapshot(
    client: httpx.AsyncClient,
    integrity_scenario: dict,
    engine: AsyncEngine,
    student_token: str,
) -> None:
    session_id, _ = await _start_session(
        client, integrity_scenario["config_id"], student_token
    )
    async with engine.begin() as conn:
        row = (
            await conn.execute(
                text("SELECT integrity_policy_snapshot FROM interview_sessions WHERE id = :sid"),
                {"sid": session_id},
            )
        ).scalar_one()
    assert dict(row) == {
        "tab_switch": 3,
        "focus_lost": 1,
        "fullscreen_exit": 2,
        "score_threshold": 3,
    }


async def test_custom_refusal_contract_is_english_only_at_the_orm(
    engine: AsyncEngine,
    integrity_scenario: dict,
    seeded_users: SeededUsers,
) -> None:
    """EN custom text is the single source; the vi column no longer exists."""
    from abridgeai.features.interviews.services.taking import (  # noqa: PLC0415
        integrity_policy_snapshot_from_config,
    )

    module_id = uuid.uuid4()
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO modules (id, course_id, title, position, status) "
                "VALUES (:m, :c, 'Refusal Module', :pos, 'draft')"
            ),
            {"m": module_id, "c": seeded_users.course_id, "pos": 9500},
        )
    config_id = await _seed_integrity_config(
        engine,
        course_id=seeded_users.course_id,
        module_id=module_id,
        actor_id=seeded_users.admin_id,
        custom_refusal_en="Custom English refusal wording.",
    )
    try:
        async with engine.begin() as conn:
            cols = (
                await conn.execute(
                    text(
                        "SELECT count(*) FROM information_schema.columns "
                        "WHERE table_name = 'interview_configs' "
                        "AND column_name = 'security_custom_refusal_vi'"
                    )
                )
            ).scalar_one()
        assert cols == 0

        from sqlalchemy.ext.asyncio import async_sessionmaker  # noqa: PLC0415

        from abridgeai.features.interviews.api.public import (  # noqa: PLC0415
            deep_clone_interview_config,
        )
        from abridgeai.features.interviews.models import (  # noqa: PLC0415
            InterviewConfig as _Cfg,
        )

        # ORM round-trip: the row reads back the EN text, the snapshot builder
        # maps all four policy numbers, and the clone path (used by course
        # copies) carries the policy to the copy.
        maker = async_sessionmaker(bind=engine, expire_on_commit=False)
        async with maker() as session:
            source = await session.get(_Cfg, config_id)
            assert source.security_custom_refusal_en == "Custom English refusal wording."
            snapshot = integrity_policy_snapshot_from_config(source)
            assert snapshot == {
                "tab_switch": 3,
                "focus_lost": 1,
                "fullscreen_exit": 2,
                "score_threshold": 3,
            }
            clone_id = await deep_clone_interview_config(
                session,
                source_config_id=config_id,
                target_module_id=module_id,
                actor_id=seeded_users.admin_id,
            )
            await session.commit()
            clone = await session.get(_Cfg, clone_id)
            assert clone.security_custom_refusal_en == "Custom English refusal wording."
            assert clone.integrity_weight_tab_switch == 3
            assert clone.integrity_score_threshold == 3
            assert not hasattr(clone, "security_custom_refusal_vi")
    finally:
        async with engine.begin() as conn:
            await hard_delete_graph(conn, "modules", [str(module_id)])
