"""Integration tests for the material-ingest orphan reaper.

``features.materials.workers.reaper._run_reconcile`` is what rescues a
material version whose ``ProcessingJob`` row committed but whose ARQ job
never ran, or died with the worker that was running it. The two live in
different stores and are committed separately, so they drift: the row says
``pending``, Redis holds nothing, and the teacher watches a spinner that
will never resolve. That is the failure this module was written for, and it
had no tests.

What the tests pin, in rough order of what it costs to get wrong:

* a Redis-less liveness scan aborts the tick -- an empty live-set means
  *every* row looks orphaned, so the alternative is re-enqueueing the whole
  backlog at once;
* a version with a live ARQ job is never touched, however old the row is;
* a row inside the grace window is never touched, so a job being enqueued
  at the moment the reaper runs is not reaped out from under itself;
* a genuine orphan is re-enqueued with its durable budget bumped, and the
  DB write is committed before the enqueue so the worker cannot pick the
  job up before the row is visible;
* an exhausted budget terminalizes the job with a message a teacher can act
  on, and tells them, instead of spinning forever;
* a job whose version has been deleted is closed out rather than retried.

The DB is real. Redis is a fake holding ARQ-serialized payloads, so the
reaper's own ``deserialize_job_raw`` reads exactly what a worker writes.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from arq.jobs import serialize_job
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    async_sessionmaker,
    create_async_engine,
)

import abridgeai.core.db.all_models  # noqa: F401  -- complete the ORM registry
from abridgeai.core.config import get_settings
from abridgeai.features.materials.workers import reaper as reaper_mod

_INGEST_TASK = "ingest_material_version_task"
_QUIZ_TASK = "run_quiz_generation_task"


def _async_url(database_url: str) -> str:
    if "+psycopg_async" in database_url:
        return database_url
    if database_url.startswith("postgresql+psycopg://"):
        return database_url.replace("postgresql+psycopg://", "postgresql+psycopg_async://", 1)
    if database_url.startswith("postgresql://"):
        return database_url.replace("postgresql://", "postgresql+psycopg_async://", 1)
    return database_url


class _FakeRedis:
    """Stand-in for the ARQ pool: liveness reads plus enqueue recording."""

    def __init__(self, *, fail_on: Exception | None = None) -> None:
        self._jobs: dict[bytes, bytes] = {}
        self._fail_on = fail_on
        self.enqueued: list[tuple[Any, ...]] = []

    def add_live_ingest(self, version_id: UUID) -> None:
        payload = serialize_job(_INGEST_TASK, (uuid4(), version_id, uuid4()), {}, 1, 0)
        self._jobs[f"arq:job:{uuid4().hex}".encode()] = payload

    def add_live_quiz_run(self, run_id: UUID) -> None:
        payload = serialize_job(_QUIZ_TASK, (uuid4(), run_id), {}, 1, 0)
        self._jobs[f"arq:job:{uuid4().hex}".encode()] = payload

    async def keys(self, _pattern: str) -> list[bytes]:
        if self._fail_on:
            raise self._fail_on
        return list(self._jobs)

    async def get(self, key: bytes) -> bytes | None:
        if self._fail_on:
            raise self._fail_on
        return self._jobs.get(key)

    async def enqueue_job(self, *args: Any) -> None:  # noqa: ANN401 -- arq's own signature
        self.enqueued.append(args)


def _ingests_enqueued(redis: _FakeRedis, version_id: UUID) -> list[tuple[Any, ...]]:
    """Ingest re-enqueues for one version.

    Filtered twice over, for two different reasons. The give-up path also
    notifies the teacher, and notification dispatch may enqueue an email job
    on the same pool -- so "nothing at all was enqueued" would assert
    something these tests do not mean. And the reaper sweeps the whole
    ``processing_jobs`` table rather than one course, so a stale row another
    suite left behind would show up here as an extra call; scoping to this
    test's own version keeps the assertions about this test's own work.
    """
    return [
        call
        for call in redis.enqueued
        if len(call) >= 3 and call[0] == _INGEST_TASK and call[2] == version_id
    ]


def _quiz_runs_enqueued(redis: _FakeRedis, run_id: UUID) -> list[tuple[Any, ...]]:
    """Quiz re-enqueues for one run; same two reasons as above."""
    return [
        call
        for call in redis.enqueued
        if len(call) >= 3 and call[0] == _QUIZ_TASK and call[2] == run_id
    ]


@pytest_asyncio.fixture
async def engine() -> AsyncIterator[AsyncEngine]:
    eng = create_async_engine(_async_url(get_settings().database_url), pool_pre_ping=True)
    yield eng
    await eng.dispose()


@pytest_asyncio.fixture(autouse=True)
def _point_reaper_at_test_db(
    engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reaper opens its own session via ``get_sessionmaker()``.

    That helper memoizes a module-global engine another suite may have built
    first, so bind it explicitly to this fixture's engine -- otherwise the
    reaper reads a different database than the one these tests seeded.
    """
    factory = async_sessionmaker(engine, expire_on_commit=False, autoflush=False)
    monkeypatch.setattr(reaper_mod, "get_sessionmaker", lambda: factory)


@pytest_asyncio.fixture(autouse=True)
def _no_live_progress_writes(monkeypatch: pytest.MonkeyPatch) -> None:
    """The reaper publishes progress to Redis on both recovery paths.

    It is best-effort and swallows its own errors, so leaving it live would
    not fail a test -- but it would reach for a real cache client. Stub it
    out so these tests touch exactly two things: Postgres and the fake pool.
    """

    async def _noop(*_args: Any, **_kwargs: Any) -> None:
        return None

    monkeypatch.setattr(reaper_mod, "publish_progress", _noop)


@pytest_asyncio.fixture
async def scope(engine: AsyncEngine) -> AsyncIterator[dict[str, UUID]]:
    """One course/module/lesson/material plus a version to reap."""
    ids = {
        key: uuid4()
        for key in (
            "org",
            "teacher",
            "course",
            "module",
            "lesson",
            "storage",
            "material",
            "version",
        )
    }

    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO organizations (id, slug, name, status) "
                "VALUES (:id, :s, 'Reaper Org', 'active')"
            ),
            {"id": ids["org"], "s": f"reap-{ids['org'].hex[:8]}"},
        )
        await conn.execute(
            text("INSERT INTO users (id, primary_email, status) VALUES (:id, :e, 'active')"),
            {"id": ids["teacher"], "e": f"reap-{ids['teacher'].hex[:8]}@test.local"},
        )
        await conn.execute(
            text(
                "INSERT INTO courses (id, organization_id, owner_user_id, slug, title, status) "
                "VALUES (:id, :o, :u, :s, 'Reaper Course', 'draft')"
            ),
            {
                "id": ids["course"],
                "o": ids["org"],
                "u": ids["teacher"],
                "s": f"reap-c-{ids['course'].hex[:6]}",
            },
        )
        await conn.execute(
            text(
                "INSERT INTO modules (id, course_id, title, position) "
                "VALUES (:id, :c, 'Mod', 1)"
            ),
            {"id": ids["module"], "c": ids["course"]},
        )
        await conn.execute(
            text("INSERT INTO lessons (id, module_id, slug, title) VALUES (:id, :m, :s, 'Lsn')"),
            {"id": ids["lesson"], "m": ids["module"], "s": f"reap-l-{ids['lesson'].hex[:6]}"},
        )
        await conn.execute(
            text(
                "INSERT INTO storage_objects (id, bucket, object_key, mime_type) "
                "VALUES (:id, 'test-bucket', :key, 'application/pdf')"
            ),
            {"id": ids["storage"], "key": f"reap/{ids['storage'].hex}"},
        )
        await conn.execute(
            text(
                "INSERT INTO learning_materials (id, lesson_id, title, material_type) "
                "VALUES (:id, :l, 'Reaper Material', 'pdf')"
            ),
            {"id": ids["material"], "l": ids["lesson"]},
        )
        await conn.execute(
            text(
                "INSERT INTO learning_material_versions "
                "(id, material_id, storage_object_id, version_no, processing_status, "
                "uploaded_by) VALUES (:id, :m, :so, 1, 'pending', :u)"
            ),
            {
                "id": ids["version"],
                "m": ids["material"],
                "so": ids["storage"],
                "u": ids["teacher"],
            },
        )

    yield ids

    async with engine.begin() as conn:
        await conn.execute(
            text("DELETE FROM notifications WHERE user_id = :u"), {"u": ids["teacher"]}
        )
        await conn.execute(
            text("DELETE FROM processing_jobs WHERE entity_id = :v"), {"v": ids["version"]}
        )
        await conn.execute(
            text("DELETE FROM generation_runs WHERE course_id = :c"), {"c": ids["course"]}
        )
        await conn.execute(
            text("DELETE FROM learning_material_versions WHERE id = :id"),
            {"id": ids["version"]},
        )
        await conn.execute(
            text("DELETE FROM learning_materials WHERE id = :id"), {"id": ids["material"]}
        )
        await conn.execute(
            text("DELETE FROM storage_objects WHERE id = :id"), {"id": ids["storage"]}
        )
        await conn.execute(text("DELETE FROM lessons WHERE id = :id"), {"id": ids["lesson"]})
        await conn.execute(text("DELETE FROM modules WHERE id = :id"), {"id": ids["module"]})
        await conn.execute(text("DELETE FROM courses WHERE id = :id"), {"id": ids["course"]})
        await conn.execute(text("DELETE FROM users WHERE id = :id"), {"id": ids["teacher"]})
        await conn.execute(
            text("DELETE FROM organizations WHERE id = :id"), {"id": ids["org"]}
        )


async def _insert_job(
    engine: AsyncEngine,
    scope: dict[str, UUID],
    *,
    status: str = "pending",
    age_seconds: int = 600,
    retry_count: int = 0,
    entity_type: str = "material_version",
    entity_id: UUID | None = None,
) -> UUID:
    """One ``ProcessingJob``, backdated so the grace period has passed."""
    job_id = uuid4()
    created = datetime.now(tz=UTC) - timedelta(seconds=age_seconds)
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO processing_jobs "
                "(id, entity_type, entity_id, job_type, status, retry_count, "
                "progress_percent, created_at, updated_at) "
                "VALUES (:id, :et, :eid, 'full_pipeline', :st, :rc, 10, :ts, :ts)"
            ),
            {
                "id": job_id,
                "et": entity_type,
                "eid": entity_id or scope["version"],
                "st": status,
                "rc": retry_count,
                "ts": created,
            },
        )
    return job_id


async def _read_job(engine: AsyncEngine, job_id: UUID) -> dict[str, Any]:
    async with engine.connect() as conn:
        row = (
            await conn.execute(
                text(
                    "SELECT status, retry_count, error_message, started_at, finished_at "
                    "FROM processing_jobs WHERE id = :id"
                ),
                {"id": job_id},
            )
        ).mappings()
        return dict(row.one())


async def _read_version(engine: AsyncEngine, version_id: UUID) -> dict[str, Any]:
    async with engine.connect() as conn:
        row = (
            await conn.execute(
                text(
                    "SELECT processing_status, processing_error "
                    "FROM learning_material_versions WHERE id = :id"
                ),
                {"id": version_id},
            )
        ).mappings()
        return dict(row.one())


# --- Rows the reaper must not touch -----------------------------------------


@pytest.mark.asyncio
async def test_a_redis_failure_aborts_the_tick_and_reaps_nothing(
    engine: AsyncEngine, scope: dict[str, UUID]
) -> None:
    """The scan is the only evidence of liveness, so losing it is disabling.

    A scan that failed open would report an empty live-set, every stuck row
    would look orphaned, and one Redis blip would re-enqueue the entire
    backlog simultaneously -- the outage turned into a thundering herd.
    """
    job_id = await _insert_job(engine, scope)
    redis = _FakeRedis(fail_on=ConnectionError("redis is down"))

    await reaper_mod._run_reconcile(arq_pool=redis)

    job = await _read_job(engine, job_id)
    assert job["status"] == "pending"
    assert job["retry_count"] == 0
    assert _ingests_enqueued(redis, scope["version"]) == []


@pytest.mark.asyncio
async def test_a_version_with_a_live_job_is_left_alone(
    engine: AsyncEngine, scope: dict[str, UUID]
) -> None:
    """A job can sit in a busy queue far longer than the grace period.

    Age is not evidence of death; absence from Redis is. Re-enqueueing a
    queued job would run the same ingest twice against one version.
    """
    job_id = await _insert_job(engine, scope, age_seconds=86_400)
    redis = _FakeRedis()
    redis.add_live_ingest(scope["version"])

    await reaper_mod._run_reconcile(arq_pool=redis)

    job = await _read_job(engine, job_id)
    assert job["status"] == "pending"
    assert job["retry_count"] == 0
    assert _ingests_enqueued(redis, scope["version"]) == []


@pytest.mark.asyncio
async def test_a_row_inside_the_grace_window_is_left_alone(
    engine: AsyncEngine, scope: dict[str, UUID]
) -> None:
    """The DB row is committed a moment before the ARQ job is written.

    A reaper tick landing in that gap would see a row with no live job and
    reap an ingest that was about to start perfectly normally.
    """
    job_id = await _insert_job(engine, scope, age_seconds=5)

    await reaper_mod._run_reconcile(arq_pool=_FakeRedis())

    job = await _read_job(engine, job_id)
    assert job["status"] == "pending"
    assert job["retry_count"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal", ["completed", "failed", "cancelled"])
async def test_a_finished_job_is_never_reconsidered(
    engine: AsyncEngine, scope: dict[str, UUID], terminal: str
) -> None:
    """Only ``pending`` and ``running`` are in flight. Re-enqueueing a
    completed ingest would redo the extraction and re-embed the chunks."""
    job_id = await _insert_job(engine, scope, status=terminal)

    await reaper_mod._run_reconcile(arq_pool=_FakeRedis())

    assert (await _read_job(engine, job_id))["status"] == terminal


@pytest.mark.asyncio
async def test_a_job_of_another_entity_type_is_left_to_its_owner(
    engine: AsyncEngine, scope: dict[str, UUID]
) -> None:
    """``processing_jobs`` is polymorphic. This reaper only knows how to
    re-enqueue a material ingest, so anything else is not its to recover --
    and would be re-enqueued as the wrong task if it tried.
    """
    job_id = await _insert_job(engine, scope, entity_type="lesson")

    await reaper_mod._run_reconcile(arq_pool=_FakeRedis())

    assert (await _read_job(engine, job_id))["status"] == "pending"


# --- Recovery ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_an_orphan_is_requeued_with_its_budget_bumped(
    engine: AsyncEngine, scope: dict[str, UUID]
) -> None:
    """The recovery path, end to end.

    ``retry_count`` is the durable budget: ARQ's own retry counter lived in
    Redis and is exactly what was lost, so the column is what survives a
    worker restart and stops an unrecoverable job cycling forever.
    """
    job_id = await _insert_job(engine, scope, status="running")
    redis = _FakeRedis()

    await reaper_mod._run_reconcile(arq_pool=redis)

    job = await _read_job(engine, job_id)
    assert job["status"] == "pending", "reset to a clean pending state for the new attempt"
    assert job["retry_count"] == 1
    assert job["started_at"] is None, "the dead attempt's start time would misreport duration"
    assert job["error_message"] is None

    version = await _read_version(engine, scope["version"])
    assert version["processing_status"] == "pending"
    assert version["processing_error"] is None


@pytest.mark.asyncio
async def test_the_requeued_job_names_the_version_and_its_uploader(
    engine: AsyncEngine, scope: dict[str, UUID]
) -> None:
    """The replacement job is enqueued through the resilient helper, so the
    arguments have to match the task the worker actually registers.

    The actor is the version's uploader: the recovered ingest's audit rows
    should name the teacher who uploaded the material, not the cron.
    """
    await _insert_job(engine, scope)
    redis = _FakeRedis()

    await reaper_mod._run_reconcile(arq_pool=redis)

    calls = _ingests_enqueued(redis, scope["version"])
    assert len(calls) == 1
    task_name, actor_id, version_id, pipeline_run_id = calls[0]
    assert task_name == _INGEST_TASK
    assert actor_id == scope["teacher"]
    assert version_id == scope["version"]
    assert isinstance(pipeline_run_id, uuid.UUID)


@pytest.mark.asyncio
async def test_the_row_is_committed_before_the_job_is_enqueued(
    engine: AsyncEngine, scope: dict[str, UUID]
) -> None:
    """A worker can claim the job microseconds after the enqueue returns.

    If the reset were still uncommitted at that moment, the worker would
    read the old ``running`` row -- or nothing at all -- and the recovery
    would fail in a way that looks like the bug it was fixing. The ordering
    is asserted by reading the row from a *separate* connection at the
    instant of the enqueue.
    """
    seen: dict[str, Any] = {}
    job_id = await _insert_job(engine, scope, status="running")

    redis = _FakeRedis()
    original_enqueue = redis.enqueue_job

    async def _spy(*args: Any) -> None:
        async with engine.connect() as conn:
            row = (
                await conn.execute(
                    text("SELECT status, retry_count FROM processing_jobs WHERE id = :id"),
                    {"id": job_id},
                )
            ).mappings()
            seen.update(dict(row.one()))
        await original_enqueue(*args)

    redis.enqueue_job = _spy  # type: ignore[method-assign]

    await reaper_mod._run_reconcile(arq_pool=redis)

    assert seen["status"] == "pending", "another connection can already see the reset"
    assert seen["retry_count"] == 1


@pytest.mark.asyncio
async def test_a_failed_enqueue_leaves_the_row_ready_for_the_next_tick(
    engine: AsyncEngine, scope: dict[str, UUID]
) -> None:
    """The commit has already happened when the enqueue fails.

    That is the right way round: the row is left clean and pending, so the
    next tick simply tries again, and the budget it spent is recorded so the
    retries remain finite.
    """
    job_id = await _insert_job(engine, scope, status="running")
    redis = _FakeRedis()

    async def _boom(*_args: Any) -> None:
        raise ConnectionError("redis went away between scan and enqueue")

    redis.enqueue_job = _boom  # type: ignore[method-assign]

    await reaper_mod._run_reconcile(arq_pool=redis)

    job = await _read_job(engine, job_id)
    assert job["status"] == "pending"
    assert job["retry_count"] == 1


@pytest.mark.asyncio
async def test_each_tick_spends_one_attempt(
    engine: AsyncEngine, scope: dict[str, UUID]
) -> None:
    """Consecutive ticks must not spend the budget faster than one per run,
    or a job would burn through three attempts in three minutes."""
    job_id = await _insert_job(engine, scope)

    for expected in (1, 2, 3):
        await reaper_mod._run_reconcile(arq_pool=_FakeRedis())
        assert (await _read_job(engine, job_id))["retry_count"] == expected


# --- Giving up ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_an_exhausted_budget_fails_the_job_with_an_actionable_message(
    engine: AsyncEngine, scope: dict[str, UUID]
) -> None:
    """The point of the whole module: stop pretending to be in flight.

    An eternal spinner tells the teacher nothing and offers them nothing to
    do. A failed status with a message naming the remedy is worse news and a
    better outcome.
    """
    job_id = await _insert_job(
        engine, scope, retry_count=reaper_mod._MAX_REQUEUE_ATTEMPTS
    )
    redis = _FakeRedis()

    await reaper_mod._run_reconcile(arq_pool=redis)

    job = await _read_job(engine, job_id)
    assert job["status"] == "failed"
    assert job["finished_at"] is not None
    assert "reprocess manually" in job["error_message"]

    version = await _read_version(engine, scope["version"])
    assert version["processing_status"] == "failed", (
        "the version drives the badge the teacher is watching"
    )
    assert version["processing_error"] == job["error_message"]
    assert _ingests_enqueued(redis, scope["version"]) == [], (
        "a job that has given up is not enqueued again"
    )


@pytest.mark.asyncio
async def test_giving_up_notifies_the_teacher_who_uploaded_it(
    engine: AsyncEngine, scope: dict[str, UUID]
) -> None:
    """Nobody is watching the spinner by the time the reaper gives up.

    The recovery is automatic and silent while it is working; the one moment
    that needs a person is the moment it stops, so the notification is the
    only thing that reaches them.
    """
    await _insert_job(engine, scope, retry_count=reaper_mod._MAX_REQUEUE_ATTEMPTS)

    await reaper_mod._run_reconcile(arq_pool=_FakeRedis())

    async with engine.connect() as conn:
        rows = (
            await conn.execute(
                text(
                    "SELECT category, user_id FROM notifications "
                    "WHERE user_id = :u AND category = 'material_processing'"
                ),
                {"u": scope["teacher"]},
            )
        ).all()
    assert len(rows) == 1


@pytest.mark.asyncio
async def test_a_job_whose_version_is_gone_is_closed_not_retried(
    engine: AsyncEngine, scope: dict[str, UUID]
) -> None:
    """The material was deleted while its ingest was still queued.

    Re-enqueueing would hand the worker a version id that no longer
    resolves, and the job would fail its way through the whole budget before
    reaching the same place. Closing it immediately keeps the counts honest.
    """
    orphan_version = uuid4()
    job_id = await _insert_job(engine, scope, entity_id=orphan_version)
    redis = _FakeRedis()
    try:
        await reaper_mod._run_reconcile(arq_pool=redis)

        job = await _read_job(engine, job_id)
        assert job["status"] == "failed"
        assert "no longer exists" in job["error_message"]
        assert job["retry_count"] == 0, "no budget is spent on something unrecoverable"
        assert _ingests_enqueued(redis, orphan_version) == []
    finally:
        # This row points at a version id the `scope` teardown never sees, so
        # it would outlive the fixture and be re-reaped by every later run.
        async with engine.begin() as conn:
            await conn.execute(
                text("DELETE FROM processing_jobs WHERE id = :id"), {"id": job_id}
            )


# --- The quiz reconciler in the same tick ------------------------------------


@pytest.mark.asyncio
async def test_the_quiz_reconciler_does_nothing_without_a_pool(
    engine: AsyncEngine, scope: dict[str, UUID]
) -> None:
    """Without Redis it can neither prove a run is dead nor replace it.

    The liveness scan would return an empty set, making every run look
    orphaned, and the budget would be spent terminalizing runs that are
    still generating. Doing nothing is the only safe option.
    """
    run_id = await _insert_quiz_run(engine, scope)

    await reaper_mod._run_reconcile_quiz(arq_pool=None)

    run = await _read_run(engine, run_id)
    assert run["status"] == "running"
    assert "reap_count" not in (run["config_json"] or {})


@pytest.mark.asyncio
async def test_a_stale_quiz_run_is_requeued_and_keeps_its_config(
    engine: AsyncEngine, scope: dict[str, UUID]
) -> None:
    """``generation_runs`` has no ``retry_count`` column, so the budget lives
    inside ``config_json`` -- beside the run's real settings.

    It therefore has to be merged in rather than written over: replacing the
    blob would discard the quiz id and question count the run needs to
    resume, and the re-enqueued job would generate the wrong thing.
    """
    run_id = await _insert_quiz_run(engine, scope)
    redis = _FakeRedis()

    await reaper_mod._run_reconcile_quiz(arq_pool=redis)

    run = await _read_run(engine, run_id)
    assert run["status"] == "pending"
    assert run["config_json"]["reap_count"] == 1
    assert run["config_json"]["question_count"] == 5, "the run's own settings survive"
    assert _quiz_runs_enqueued(redis, run_id) == [(_QUIZ_TASK, scope["teacher"], run_id)]


@pytest.mark.asyncio
async def test_a_quiz_run_with_a_live_job_is_left_alone(
    engine: AsyncEngine, scope: dict[str, UUID]
) -> None:
    """A generation run is quiet for minutes between LLM calls; silence is
    not death. A duplicate run would double-spend the teacher's LLM budget
    and race the original for the same row."""
    run_id = await _insert_quiz_run(engine, scope)
    redis = _FakeRedis()
    redis.add_live_quiz_run(run_id)

    await reaper_mod._run_reconcile_quiz(arq_pool=redis)

    run = await _read_run(engine, run_id)
    assert run["status"] == "running"
    assert _quiz_runs_enqueued(redis, run_id) == []


@pytest.mark.asyncio
async def test_a_recently_restarted_run_is_judged_on_when_work_began(
    engine: AsyncEngine, scope: dict[str, UUID]
) -> None:
    """A re-enqueued run's row is old; its current attempt is not.

    Staleness is read from ``started_at`` -- when the dispatcher marked the
    run running -- rather than from ``created_at``. Judging by row age would
    reap a healthy run the moment it was recovered, every time, and it would
    never get past its budget.
    """
    run_id = await _insert_quiz_run(engine, scope, age_seconds=7200, started_seconds_ago=30)

    await reaper_mod._run_reconcile_quiz(arq_pool=_FakeRedis())

    assert (await _read_run(engine, run_id))["status"] == "running"


@pytest.mark.asyncio
async def test_an_exhausted_quiz_run_carries_its_failure_message(
    engine: AsyncEngine, scope: dict[str, UUID]
) -> None:
    """The message is read back out of ``config_json`` by the generation-run
    projection and shown in the authoring panel, so it is the teacher's only
    account of what happened to their generation."""
    run_id = await _insert_quiz_run(
        engine, scope, reap_count=reaper_mod._QUIZ_MAX_REQUEUE_ATTEMPTS
    )
    redis = _FakeRedis()

    await reaper_mod._run_reconcile_quiz(arq_pool=redis)

    run = await _read_run(engine, run_id)
    assert run["status"] == "failed"
    assert run["finished_at"] is not None
    assert "please retry generation" in run["config_json"]["failure"]["message"]
    assert _quiz_runs_enqueued(redis, run_id) == []


@pytest.mark.asyncio
async def test_an_interview_run_is_not_touched_by_the_quiz_reconciler(
    engine: AsyncEngine, scope: dict[str, UUID]
) -> None:
    """The two share a table and a shape, and are told apart only by
    ``generation_type``. Crossing them would re-enqueue an interview run as
    a quiz task, which no worker would handle correctly."""
    run_id = await _insert_quiz_run(engine, scope, generation_type="interview")
    redis = _FakeRedis()

    await reaper_mod._run_reconcile_quiz(arq_pool=redis)

    assert (await _read_run(engine, run_id))["status"] == "running"
    assert _quiz_runs_enqueued(redis, run_id) == []


async def _insert_quiz_run(
    engine: AsyncEngine,
    scope: dict[str, UUID],
    *,
    status: str = "running",
    age_seconds: int = 900,
    started_seconds_ago: int | None = None,
    reap_count: int | None = None,
    generation_type: str = "quiz",
) -> UUID:
    """One generation run, backdated past the 5-minute generation grace."""
    import json

    run_id = uuid4()
    config: dict[str, Any] = {"quiz_id": str(uuid4()), "question_count": 5}
    if reap_count is not None:
        config["reap_count"] = reap_count
    created = datetime.now(tz=UTC) - timedelta(seconds=age_seconds)
    started = (
        created
        if started_seconds_ago is None
        else datetime.now(tz=UTC) - timedelta(seconds=started_seconds_ago)
    )

    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO generation_runs (id, generation_type, source_scope_kind, "
                "course_id, module_id, requested_by, status, config_json, "
                "started_at, created_at, updated_at) "
                "VALUES (:id, :gt, 'module', :c, :m, :u, :st, CAST(:cfg AS jsonb), "
                ":started, :created, :created)"
            ),
            {
                "id": run_id,
                "gt": generation_type,
                "c": scope["course"],
                "m": scope["module"],
                "u": scope["teacher"],
                "st": status,
                "cfg": json.dumps(config),
                "started": started,
                "created": created,
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
