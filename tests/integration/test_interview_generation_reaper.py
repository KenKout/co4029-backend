"""Integration tests for the interview generation-run orphan reaper.

``features.materials.workers.reaper._run_reconcile_interview`` is the only
thing that rescues an interview ``generation_runs`` row left ``running`` by a
worker that died mid-pipeline. Without it the row keeps its
``dedup_key``, the partial unique index ``uq_generation_runs_active_dedup``
(``WHERE status IN ('pending','running')``) rejects every later Generate as
409 ``interview_generation_in_progress``, and the teacher's config is wedged
forever. It shipped with zero tests; these pin the contract:

* a stale ``running`` run with no live ARQ job is re-enqueued (``pending`` +
  ``reap_count`` bumped) so the dedup index frees up;
* a run whose ARQ job is still live in Redis is left alone (no double-spend on
  the LLM endpoint, no racing the original);
* a fresh ``running`` run inside the grace window is left alone;
* budget exhaustion terminalizes the run as ``failed`` with a readable message
  instead of spinning forever;
* a Redis-less tick is a no-op (an empty liveness scan must not be read as
  "everything is orphaned");
* the pipeline's own ``config_json`` writes MERGE, so a progress checkpoint
  cannot erase the reaper's durable ``reap_count`` budget.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from arq.jobs import serialize_job
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

import abridgeai.core.db.all_models  # noqa: F401  -- complete the ORM registry
from abridgeai.core.config import get_settings
from abridgeai.features.materials.workers import reaper as reaper_mod

_INTERVIEW_TASK = "run_interview_generation_task"


def _async_url(database_url: str) -> str:
    if "+psycopg_async" in database_url:
        return database_url
    if database_url.startswith("postgresql+psycopg://"):
        return database_url.replace("postgresql+psycopg://", "postgresql+psycopg_async://", 1)
    if database_url.startswith("postgresql://"):
        return database_url.replace("postgresql://", "postgresql+psycopg_async://", 1)
    return database_url


class _FakeRedis:
    """Minimal stand-in for the ARQ pool the reaper reads liveness from.

    Only the two calls the reaper makes are implemented (``keys`` + ``get``),
    plus ``enqueue_job`` recording. Jobs are stored under real ``arq:job:<id>``
    keys with ARQ's own serializer so ``deserialize_job_raw`` in the reaper
    parses exactly what a real worker would have written.
    """

    def __init__(self) -> None:
        self._jobs: dict[bytes, bytes] = {}
        self.enqueued: list[tuple[Any, ...]] = []

    def add_live_job(
        self,
        run_id: UUID,
        *,
        actor_id: UUID | None = None,
        function_name: str = _INTERVIEW_TASK,
    ) -> None:
        payload = serialize_job(function_name, (actor_id or uuid4(), run_id), {}, 1, 0)
        self._jobs[f"arq:job:{uuid4().hex}".encode()] = payload

    async def keys(self, _pattern: str) -> list[bytes]:
        return list(self._jobs.keys())

    async def get(self, key: bytes) -> bytes | None:
        return self._jobs.get(key)

    async def enqueue_job(self, *args: Any) -> None:  # noqa: ANN401 -- arq's own signature
        self.enqueued.append(args)


@pytest_asyncio.fixture
async def engine() -> AsyncIterator[AsyncEngine]:
    eng = create_async_engine(_async_url(get_settings().database_url), pool_pre_ping=True)
    yield eng
    await eng.dispose()


@pytest_asyncio.fixture(autouse=True)
def _point_reaper_at_test_db(
    engine: AsyncEngine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The reaper opens its OWN session via ``get_sessionmaker()``.

    Under pytest ``Settings`` already rewrites ``database_url`` to the test DB,
    but ``get_sessionmaker`` memoizes a module-global engine that another suite
    may have built first. Bind it explicitly to this fixture's engine so the
    reaper reads and writes the rows these tests seeded.
    """
    factory = async_sessionmaker(engine, expire_on_commit=False, autoflush=False)
    monkeypatch.setattr(reaper_mod, "get_sessionmaker", lambda: factory)


@pytest_asyncio.fixture
async def seeded(engine: AsyncEngine) -> AsyncIterator[dict[str, UUID]]:
    """One course/module/interview-config plus a generation run to reap."""
    org, teacher, course = uuid4(), uuid4(), uuid4()
    module_id, cfg_id, run_id = uuid4(), uuid4(), uuid4()

    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO organizations (id, slug, name, status) "
                "VALUES (:id, :slug, 'Reaper Org', 'active')"
            ),
            {"id": org, "slug": f"reap-org-{org.hex[:8]}"},
        )
        await conn.execute(
            text("INSERT INTO users (id, primary_email, status) VALUES (:t, :te, 'active')"),
            {"t": teacher, "te": f"reap-{teacher.hex[:8]}@test.local"},
        )
        await conn.execute(
            text(
                "INSERT INTO courses (id, organization_id, owner_user_id, slug, title, status) "
                "VALUES (:id, :org, :u, :slug, 'Reaper Course', 'published')"
            ),
            {"id": course, "org": org, "u": teacher, "slug": f"reap-course-{course.hex[:8]}"},
        )
        await conn.execute(
            text(
                "INSERT INTO modules (id, course_id, title, position, status) "
                "VALUES (:m, :c, 'Reaper Module', 1, 'published')"
            ),
            {"m": module_id, "c": course},
        )
        await conn.execute(
            text(
                "INSERT INTO interview_configs (id, course_id, module_id, title, status, slug) "
                "VALUES (:id, :c, :m, 'Reaper Interview', 'draft', "
                "'slug-' || uuid_generate_v4()::text)"
            ),
            {"id": cfg_id, "c": course, "m": module_id},
        )

    yield {
        "org": org,
        "teacher": teacher,
        "course": course,
        "module_id": module_id,
        "cfg_id": cfg_id,
        "run_id": run_id,
    }

    async with engine.begin() as conn:
        await conn.execute(
            text("DELETE FROM notifications WHERE entity_id = :id"), {"id": cfg_id}
        )
        await conn.execute(
            text("DELETE FROM generation_runs WHERE course_id = :c"), {"c": course}
        )
        await conn.execute(text("DELETE FROM interview_configs WHERE id = :id"), {"id": cfg_id})
        await conn.execute(text("DELETE FROM modules WHERE id = :id"), {"id": module_id})
        await conn.execute(text("DELETE FROM courses WHERE id = :id"), {"id": course})
        await conn.execute(text("DELETE FROM users WHERE id = :id"), {"id": teacher})
        await conn.execute(text("DELETE FROM organizations WHERE id = :id"), {"id": org})


async def _insert_run(
    engine: AsyncEngine,
    seeded: dict[str, UUID],
    *,
    status: str = "running",
    age_seconds: int = 900,
    reap_count: int | None = None,
    dedup: bool = True,
    generation_type: str = "interview",
) -> UUID:
    """Insert one generation run, backdating started_at/updated_at by ``age_seconds``."""
    run_id = uuid4()
    config: dict[str, Any] = {
        "interview_config_id": str(seeded["cfg_id"]),
        "question_count": 2,
    }
    if reap_count is not None:
        config["reap_count"] = reap_count
    started = datetime.now(tz=UTC) - timedelta(seconds=age_seconds)
    import json as _json

    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO generation_runs (id, generation_type, source_scope_kind, "
                "course_id, module_id, requested_by, status, config_json, dedup_key, "
                "started_at, created_at, updated_at) "
                "VALUES (:id, :gt, 'module', :c, :m, :u, :st, CAST(:cfg AS jsonb), :dk, "
                ":started, :started, :started)"
            ),
            {
                "id": run_id,
                "gt": generation_type,
                "c": seeded["course"],
                "m": seeded["module_id"],
                "u": seeded["teacher"],
                "st": status,
                "cfg": _json.dumps(config),
                "dk": f"interview-generation:{seeded['cfg_id']}" if dedup else None,
                "started": started,
            },
        )
    return run_id


async def _read_run(engine: AsyncEngine, run_id: UUID) -> dict[str, Any]:
    async with engine.connect() as conn:
        row = (
            await conn.execute(
                text(
                    "SELECT status, config_json, finished_at, started_at "
                    "FROM generation_runs WHERE id = :id"
                ),
                {"id": run_id},
            )
        ).mappings()
        return dict(row.one())


@pytest.mark.asyncio
async def test_stale_running_run_is_requeued_and_frees_the_dedup_key(
    engine: AsyncEngine,
    seeded: dict[str, UUID],
) -> None:
    """The core rescue: a dead ``running`` run goes back to ``pending``.

    That is what releases ``uq_generation_runs_active_dedup``'s hold on the
    config — while the row sits ``running`` the teacher's next Generate 409s.
    """
    run_id = await _insert_run(engine, seeded)
    redis = _FakeRedis()  # no live job for this run

    await reaper_mod._run_reconcile_interview(arq_pool=redis)

    after = await _read_run(engine, run_id)
    assert after["status"] == "pending"
    assert after["config_json"]["reap_count"] == 1
    # started_at is cleared so the NEXT tick measures staleness from the new
    # attempt, not the dead one.
    assert after["started_at"] is None
    assert redis.enqueued == [(_INTERVIEW_TASK, seeded["teacher"], run_id)]


@pytest.mark.asyncio
async def test_run_with_a_live_arq_job_is_left_alone(
    engine: AsyncEngine,
    seeded: dict[str, UUID],
) -> None:
    """A slow-but-alive run must never be reaped.

    Re-enqueueing it would double LLM spend and race the original writer.
    """
    run_id = await _insert_run(engine, seeded)
    redis = _FakeRedis()
    redis.add_live_job(run_id)

    await reaper_mod._run_reconcile_interview(arq_pool=redis)

    after = await _read_run(engine, run_id)
    assert after["status"] == "running"
    assert "reap_count" not in after["config_json"]
    assert redis.enqueued == []


@pytest.mark.asyncio
async def test_a_live_job_for_another_task_does_not_protect_the_run(
    engine: AsyncEngine,
    seeded: dict[str, UUID],
) -> None:
    """Liveness is per-function: only ``run_interview_generation_task`` counts.

    A quiz job carrying the same run id must not shield an interview run.
    """
    run_id = await _insert_run(engine, seeded)
    redis = _FakeRedis()
    redis.add_live_job(run_id, function_name="run_quiz_generation_task")

    await reaper_mod._run_reconcile_interview(arq_pool=redis)

    assert (await _read_run(engine, run_id))["status"] == "pending"


@pytest.mark.asyncio
async def test_fresh_running_run_inside_the_grace_window_is_left_alone(
    engine: AsyncEngine,
    seeded: dict[str, UUID],
) -> None:
    """A run that started seconds ago is not an orphan even with no ARQ key.

    This covers the gap between enqueue and the ``arq:job:*`` key appearing.
    """
    run_id = await _insert_run(engine, seeded, age_seconds=30)
    redis = _FakeRedis()

    await reaper_mod._run_reconcile_interview(arq_pool=redis)

    after = await _read_run(engine, run_id)
    assert after["status"] == "running"
    assert redis.enqueued == []


@pytest.mark.asyncio
async def test_exhausted_budget_terminalizes_the_run_as_failed(
    engine: AsyncEngine,
    seeded: dict[str, UUID],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Once the re-enqueue budget is spent the run must FAIL, not spin.

    A permanently-``running`` row keeps the dedup index locked; failing it is
    what lets the teacher retry (and surfaces a readable reason).
    """
    notify = AsyncMock()
    monkeypatch.setattr(reaper_mod, "notify_interview_generation_outcome", notify)
    run_id = await _insert_run(
        engine, seeded, reap_count=reaper_mod._INTERVIEW_MAX_REQUEUE_ATTEMPTS
    )
    redis = _FakeRedis()

    await reaper_mod._run_reconcile_interview(arq_pool=redis)

    after = await _read_run(engine, run_id)
    assert after["status"] == "failed"
    assert after["finished_at"] is not None
    assert "could not be recovered" in after["config_json"]["failure"]["message"]
    assert redis.enqueued == []
    # The initiating teacher is told, with the config id for the deep link.
    notify.assert_awaited_once()
    kwargs = notify.await_args.kwargs
    assert kwargs["recipient_user_id"] == seeded["teacher"]
    assert kwargs["config_id"] == seeded["cfg_id"]
    assert kwargs["succeeded"] is False


@pytest.mark.asyncio
async def test_no_arq_pool_is_a_no_op(
    engine: AsyncEngine,
    seeded: dict[str, UUID],
) -> None:
    """Without Redis the liveness scan is blind — reaping would be guessing.

    An empty scan makes EVERY run look orphaned, so a Redis-less tick used to
    spend the durable budget on live runs and fail them two ticks later.
    """
    run_id = await _insert_run(engine, seeded)

    await reaper_mod._run_reconcile_interview(arq_pool=None)

    after = await _read_run(engine, run_id)
    assert after["status"] == "running"
    assert "reap_count" not in after["config_json"]


@pytest.mark.asyncio
async def test_a_quiz_run_is_not_touched_by_the_interview_reaper(
    engine: AsyncEngine,
    seeded: dict[str, UUID],
) -> None:
    """Scoping guard: the interview sweep filters on ``generation_type``."""
    run_id = await _insert_run(engine, seeded, generation_type="quiz", dedup=False)
    redis = _FakeRedis()

    await reaper_mod._run_reconcile_interview(arq_pool=redis)

    assert (await _read_run(engine, run_id))["status"] == "running"


@pytest.mark.asyncio
async def test_pipeline_progress_write_preserves_the_reapers_budget(
    engine: AsyncEngine,
    seeded: dict[str, UUID],
) -> None:
    """``_update_run`` must MERGE ``config_json``, never replace it.

    The pipeline holds an in-memory ``config_json`` snapshot taken at run start
    and re-commits it on every progress tick. A blind overwrite erased the
    reaper's ``reap_count``, resetting the recovery budget to 0 — so an
    unrecoverable run could be re-enqueued forever, burning LLM spend each
    cycle instead of failing after two attempts.
    """
    from abridgeai.features.interviews.ai.pipelines.generation import (
        _RunState,
        _write_progress,
    )

    run_id = await _insert_run(engine, seeded, status="running", reap_count=1)
    factory = async_sessionmaker(engine, expire_on_commit=False, autoflush=False)

    # The pipeline's snapshot predates the reaper's write: no reap_count in it.
    state = _RunState(
        id=run_id,
        course_id=seeded["course"],
        config_json={"interview_config_id": str(seeded["cfg_id"]), "question_count": 2},
        requested_by=seeded["teacher"],
    )
    async with factory() as db:
        await _write_progress(db, state, phase="generating", accepted=1, target=2)

    after = await _read_run(engine, run_id)
    assert after["config_json"]["reap_count"] == 1, "progress write clobbered the reaper budget"
    # ...and the progress the teacher's SPA polls still landed.
    assert after["config_json"]["progress"] == {
        "phase": "generating",
        "accepted": 1,
        "target": 2,
    }


@pytest.mark.asyncio
async def test_requeued_run_can_be_claimed_by_the_pipeline_again(
    engine: AsyncEngine,
    seeded: dict[str, UUID],
) -> None:
    """End-to-end of the rescue: after reaping, the pipeline's claim succeeds.

    ``_claim_pending_run`` only claims ``status='pending'``, so this is the
    assertion that the reaper's reset actually makes the run runnable rather
    than merely looking recovered.
    """
    from abridgeai.features.interviews.ai.pipelines.generation import _claim_pending_run

    run_id = await _insert_run(engine, seeded)
    redis = _FakeRedis()
    await reaper_mod._run_reconcile_interview(arq_pool=redis)

    factory = async_sessionmaker(engine, expire_on_commit=False, autoflush=False)
    async with factory() as db:
        claimed = await _claim_pending_run(db, run_id)
        await db.commit()

    assert claimed is True
    assert (await _read_run(engine, run_id))["status"] == "running"
