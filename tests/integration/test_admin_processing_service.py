"""Coverage for ``admin/services/processing.py``.

``test_admin.py`` already covers the happy-path retry and the job listing over
HTTP. What had no test is everything an operator reaches for when a job has
actually gone wrong:

* ``job_investigation`` -- the four surfaces (row, resolved owner, per-stage
  rollup, individual calls) assembled into one payload. Its ownership walk is
  resolved at READ time because ``processing_jobs.entity_id`` is polymorphic
  with no foreign key, so nothing in the schema keeps it honest; a test is the
  only thing that does.
* the retry guard -- ``retry_failed_job`` is deliberately not idempotent in
  the loose sense: it refuses anything not in ``failed``. Retrying a RUNNING
  job would double-enqueue live work.
* ``queue_depth`` / ``status_counts_since`` -- the numbers on the operations
  page, which have to agree with the list they sit above.

Isolation: the suite shares one Postgres, so every fixture here creates its
own organization/course/quiz chain and deletes it, and the window queries are
asserted as deltas against a baseline taken in the same test rather than as
absolute counts.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import pytest_asyncio
from alembic import command
from alembic.config import Config
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from abridgeai.core.config import get_settings
from abridgeai.core.exceptions import NotFoundError
from abridgeai.features.admin.services import processing as processing_service


def _async_url(database_url: str) -> str:
    if "+psycopg_async" in database_url:
        return database_url
    if database_url.startswith("postgresql+psycopg://"):
        return database_url.replace("postgresql+psycopg://", "postgresql+psycopg_async://", 1)
    if database_url.startswith("postgresql://"):
        return database_url.replace("postgresql://", "postgresql+psycopg_async://", 1)
    return database_url


def _ensure_head() -> None:
    root = Path(__file__).resolve().parents[2]
    cfg = Config(str(root / "alembic.ini"))
    cfg.set_main_option("script_location", str(root / "migrations"))
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
async def db(
    session_factory: async_sessionmaker[AsyncSession],
) -> AsyncIterator[AsyncSession]:
    async with session_factory() as session:
        yield session


class Chain:
    """A quiz with a real course and organization behind it.

    ``job_context`` resolves ownership by walking quiz -> course ->
    organization, so a job pointed at a bare uuid proves nothing about that
    walk. This gives it something real to find.
    """

    def __init__(self) -> None:
        self.org_id = uuid.uuid4()
        self.user_id = uuid.uuid4()
        self.course_id = uuid.uuid4()
        self.module_id = uuid.uuid4()
        self.quiz_id = uuid.uuid4()
        self.job_id = uuid.uuid4()


@pytest_asyncio.fixture
async def chain(engine: AsyncEngine) -> AsyncIterator[Chain]:
    c = Chain()
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO organizations (id, slug, name, status) "
                "VALUES (:id, :slug, 'Processing Coverage Org', 'active')"
            ),
            {"id": c.org_id, "slug": f"proc-cov-{c.org_id.hex[:10]}"},
        )
        await conn.execute(
            text("INSERT INTO users (id, primary_email) VALUES (:id, :email)"),
            {"id": c.user_id, "email": f"proc-{c.user_id.hex[:8]}@test.local"},
        )
        await conn.execute(
            text(
                "INSERT INTO courses (id, organization_id, owner_user_id, slug, title) "
                "VALUES (:id, :org, :owner, :slug, 'Processing Coverage Course')"
            ),
            {
                "id": c.course_id,
                "org": c.org_id,
                "owner": c.user_id,
                "slug": f"proc-course-{c.course_id.hex[:8]}",
            },
        )
        await conn.execute(
            text(
                "INSERT INTO modules (id, course_id, title, position) VALUES (:id, :course, 'M', 1)"
            ),
            {"id": c.module_id, "course": c.course_id},
        )
        await conn.execute(
            text(
                "INSERT INTO quizzes (id, course_id, module_id, title, status, slug) "
                "VALUES (:id, :course, :module, 'Q', 'draft', "
                "        'proc-quiz-' || uuid_generate_v4()::text)"
            ),
            {"id": c.quiz_id, "course": c.course_id, "module": c.module_id},
        )
    yield c
    async with engine.begin() as conn:
        await conn.execute(
            text("DELETE FROM ai_model_calls WHERE processing_job_id = :id"), {"id": c.job_id}
        )
        await conn.execute(text("DELETE FROM processing_jobs WHERE id = :id"), {"id": c.job_id})
        await conn.execute(text("DELETE FROM quizzes WHERE id = :id"), {"id": c.quiz_id})
        await conn.execute(text("DELETE FROM modules WHERE id = :id"), {"id": c.module_id})
        await conn.execute(text("DELETE FROM courses WHERE id = :id"), {"id": c.course_id})
        await conn.execute(text("DELETE FROM users WHERE id = :id"), {"id": c.user_id})
        await conn.execute(text("DELETE FROM organizations WHERE id = :id"), {"id": c.org_id})


async def _insert_job(
    engine: AsyncEngine,
    chain: Chain,
    *,
    status: str,
    created_at: datetime | None = None,
    started_at: datetime | None = None,
    finished_at: datetime | None = None,
    retry_count: int = 0,
    error_message: str | None = None,
) -> None:
    # ``updated_at`` is pinned alongside ``created_at`` on purpose: the list
    # and summary windows filter on updated_at (it bounds the scan and tracks
    # recent activity), so a backdated job whose updated_at defaulted to NOW()
    # would sit outside the very window the test places it in.
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO processing_jobs "
                "(id, entity_type, entity_id, job_type, status, progress_percent, "
                " error_message, retry_count, created_at, updated_at, "
                " started_at, finished_at) "
                "VALUES (:id, 'quiz', :eid, 'generate_quiz', :status, 0, "
                "        :err, :retries, COALESCE(:created, NOW()), "
                "        COALESCE(:created, NOW()), :started, :finished)"
            ),
            {
                "id": chain.job_id,
                "eid": chain.quiz_id,
                "status": status,
                "err": error_message,
                "retries": retry_count,
                "created": created_at,
                "started": started_at,
                "finished": finished_at,
            },
        )


# ---------------------------------------------------------------------------
# get_job
# ---------------------------------------------------------------------------


async def test_get_job_returns_the_row(db: AsyncSession, engine: AsyncEngine, chain: Chain) -> None:
    await _insert_job(engine, chain, status="failed", error_message="boom")
    job = await processing_service.get_job(db, job_id=chain.job_id)
    assert job["id"] == chain.job_id
    assert job["status"] == "failed"
    assert job["job_type"] == "generate_quiz"


async def test_get_job_raises_for_an_unknown_id(db: AsyncSession) -> None:
    with pytest.raises(NotFoundError):
        await processing_service.get_job(db, job_id=uuid.uuid4())


# ---------------------------------------------------------------------------
# job_investigation
# ---------------------------------------------------------------------------


async def test_investigation_resolves_the_owner_by_walking_the_entity_chain(
    db: AsyncSession, engine: AsyncEngine, chain: Chain
) -> None:
    """quiz -> course -> organization, resolved at read time.

    ``entity_id`` is polymorphic with no foreign key, so nothing in the schema
    keeps this walk correct. If it breaks, the operations page shows a job
    with no owner and an operator cannot tell whose tenant is affected.
    """
    await _insert_job(engine, chain, status="failed", error_message="boom")

    payload = await processing_service.job_investigation(db, job_id=chain.job_id)

    context = payload["context"]
    assert context["course_id"] == chain.course_id
    assert context["course_title"] == "Processing Coverage Course"
    assert context["organization_id"] == chain.org_id
    assert context["organization_name"] == "Processing Coverage Org"


async def test_investigation_separates_queue_wait_from_run_duration(
    db: AsyncSession, engine: AsyncEngine, chain: Chain
) -> None:
    """The two numbers answer different questions and must not be conflated.

    A job that took ten minutes wall-clock but ran for four seconds was queued,
    not slow -- and the fix for those two situations is not the same one.
    """
    created = datetime.now(tz=UTC) - timedelta(minutes=10)
    started = created + timedelta(minutes=6)
    finished = started + timedelta(seconds=30)
    await _insert_job(
        engine,
        chain,
        status="completed",
        created_at=created,
        started_at=started,
        finished_at=finished,
    )

    context = (await processing_service.job_investigation(db, job_id=chain.job_id))["context"]
    assert context["queue_wait_seconds"] == pytest.approx(360, abs=2)
    assert context["duration_seconds"] == pytest.approx(30, abs=2)
    assert context["is_running"] is False


async def test_a_running_job_reports_elapsed_time_not_null(
    db: AsyncSession, engine: AsyncEngine, chain: Chain
) -> None:
    """An in-flight job must say how long it has been going.

    Reporting NULL until it finishes hides exactly the job an operator is
    looking for: the one that has been running far too long.
    """
    started = datetime.now(tz=UTC) - timedelta(minutes=5)
    await _insert_job(
        engine,
        chain,
        status="running",
        created_at=started - timedelta(seconds=10),
        started_at=started,
    )

    now = datetime.now(tz=UTC)
    payload = await processing_service.job_investigation(db, job_id=chain.job_id, now=now)
    context = payload["context"]
    assert context["is_running"] is True
    assert context["duration_seconds"] == pytest.approx(300, abs=5)
    assert payload["as_of"] == now


async def test_a_job_that_never_started_reports_null_not_zero(
    db: AsyncSession, engine: AsyncEngine, chain: Chain
) -> None:
    """ "Has not run" and "ran for no time" are different states."""
    await _insert_job(engine, chain, status="pending")
    context = (await processing_service.job_investigation(db, job_id=chain.job_id))["context"]
    assert context["queue_wait_seconds"] is None
    assert context["duration_seconds"] is None
    assert context["is_running"] is False


async def test_investigation_rolls_up_ai_calls_by_stage(
    db: AsyncSession, engine: AsyncEngine, chain: Chain
) -> None:
    """The rollup and the call list must describe the same calls."""
    await _insert_job(engine, chain, status="completed")
    async with engine.begin() as conn:
        for stage, tokens, cost in (
            ("draft", 100, "0.01"),
            ("draft", 200, "0.02"),
            ("validate", 50, "0.005"),
        ):
            await conn.execute(
                text(
                    "INSERT INTO ai_model_calls "
                    "(id, processing_job_id, role, stage_name, model_name, "
                    " total_tokens, estimated_cost_usd, latency_ms, status) "
                    "VALUES (gen_random_uuid(), :job, 'ingest', :stage, 'test-model', "
                    "        :tokens, CAST(:cost AS numeric), 5, 'success')"
                ),
                {"job": chain.job_id, "stage": stage, "tokens": tokens, "cost": cost},
            )

    payload = await processing_service.job_investigation(db, job_id=chain.job_id)
    assert len(payload["ai_calls"]) == 3
    # The rollup projects ``stage`` (COALESCE of stage_name/role/'unknown')
    # and ``tokens``, not the raw column names -- session-runtime calls
    # attribute by role alone and would otherwise drop out of a rollup that
    # claims to cover the whole job.
    by_stage = {s["stage"]: s for s in payload["stages"]}
    assert set(by_stage) == {"draft", "validate"}
    assert int(by_stage["draft"]["tokens"]) == 300
    assert int(by_stage["draft"]["call_count"]) == 2
    assert int(by_stage["validate"]["call_count"]) == 1


async def test_investigation_caps_the_call_list(
    db: AsyncSession, engine: AsyncEngine, chain: Chain
) -> None:
    """A pathological job must not return thousands of rows to a detail page."""
    await _insert_job(engine, chain, status="completed")
    async with engine.begin() as conn:
        for _ in range(5):
            await conn.execute(
                text(
                    "INSERT INTO ai_model_calls "
                    "(id, processing_job_id, role, stage_name, model_name, "
                    " total_tokens, estimated_cost_usd, latency_ms, status) "
                    "VALUES (gen_random_uuid(), :job, 'ingest', 'draft', 'test-model', "
                    "        10, 0.001, 5, 'success')"
                ),
                {"job": chain.job_id},
            )

    payload = await processing_service.job_investigation(db, job_id=chain.job_id, ai_call_limit=2)
    assert len(payload["ai_calls"]) == 2


async def test_investigation_of_a_missing_job_is_not_found(db: AsyncSession) -> None:
    with pytest.raises(NotFoundError):
        await processing_service.job_investigation(db, job_id=uuid.uuid4())


# ---------------------------------------------------------------------------
# retry_failed_job
# ---------------------------------------------------------------------------


async def test_retry_resets_the_job_and_enqueues_it(
    db: AsyncSession, engine: AsyncEngine, chain: Chain
) -> None:
    """The reset has to clear the previous run's traces, not just the status.

    Leaving ``error_message`` or ``finished_at`` behind makes the requeued job
    render as still-broken on the operations page.
    """
    await _insert_job(
        engine,
        chain,
        status="failed",
        error_message="boom",
        retry_count=1,
        started_at=datetime.now(tz=UTC) - timedelta(minutes=1),
        finished_at=datetime.now(tz=UTC),
    )

    enqueued: list[tuple[object, ...]] = []

    class _Pool:
        async def enqueue_job(self, *args: object) -> None:
            enqueued.append(args)

    refreshed = await processing_service.retry_failed_job(db, job_id=chain.job_id, arq_pool=_Pool())

    assert refreshed["status"] == "pending"
    assert refreshed["retry_count"] == 2
    assert refreshed["error_message"] is None
    assert refreshed["started_at"] is None
    assert refreshed["finished_at"] is None

    assert enqueued == [("generate_quiz", str(chain.job_id), str(chain.quiz_id))]


@pytest.mark.parametrize("status", ["pending", "running", "completed"])
async def test_retry_refuses_a_job_that_is_not_failed(
    db: AsyncSession, engine: AsyncEngine, chain: Chain, status: str
) -> None:
    """The guard is what stops a running job being enqueued a second time.

    Without it, a double-click on Retry duplicates live work -- and the second
    worker overwrites the first one's results.
    """
    await _insert_job(engine, chain, status=status)

    class _Pool:
        def __init__(self) -> None:
            self.calls = 0

        async def enqueue_job(self, *args: object) -> None:
            self.calls += 1

    pool = _Pool()
    with pytest.raises(NotFoundError):
        await processing_service.retry_failed_job(db, job_id=chain.job_id, arq_pool=pool)
    assert pool.calls == 0, "a refused retry must not enqueue anything"

    still = await processing_service.get_job(db, job_id=chain.job_id)
    assert still["status"] == status, "the job must be left exactly as it was"


async def test_retry_of_a_missing_job_is_not_found(db: AsyncSession) -> None:
    with pytest.raises(NotFoundError):
        await processing_service.retry_failed_job(db, job_id=uuid.uuid4(), arq_pool=None)


async def test_retry_still_resets_when_no_queue_is_available(
    db: AsyncSession, engine: AsyncEngine, chain: Chain
) -> None:
    """Re-enqueueing is best-effort; the DB reset is not.

    An operator running without a live ARQ pool (or against a pool that has no
    ``enqueue_job``) must still get the job back into ``pending`` so a worker
    picks it up on its own.
    """
    await _insert_job(engine, chain, status="failed", error_message="boom")
    refreshed = await processing_service.retry_failed_job(db, job_id=chain.job_id, arq_pool=None)
    assert refreshed["status"] == "pending"

    # A pool object that does not implement enqueue_job must not raise either.
    async with engine.begin() as conn:
        await conn.execute(
            text("UPDATE processing_jobs SET status = 'failed' WHERE id = :id"),
            {"id": chain.job_id},
        )
    refreshed = await processing_service.retry_failed_job(
        db, job_id=chain.job_id, arq_pool=object()
    )
    assert refreshed["status"] == "pending"


# ---------------------------------------------------------------------------
# queue_depth / status_counts_since / list_jobs
# ---------------------------------------------------------------------------


async def test_queue_depth_counts_the_job_it_was_given(
    db: AsyncSession, engine: AsyncEngine, chain: Chain
) -> None:
    """Asserted as a delta: the shared database has other suites' jobs in it."""
    before = await processing_service.queue_depth(db)
    await _insert_job(engine, chain, status="pending")
    after = await processing_service.queue_depth(db)

    assert after.get("pending", 0) == before.get("pending", 0) + 1


async def test_status_counts_agree_with_the_list_they_sit_above(
    db: AsyncSession, engine: AsyncEngine, chain: Chain
) -> None:
    """The summary and the list must be computed over the same window.

    They are separate queries with separate WHERE clauses, which is exactly
    how a summary drifts from the rows underneath it -- and an operator has no
    way to tell which of the two is lying.
    """
    since = datetime.now(tz=UTC) - timedelta(hours=1)
    await _insert_job(engine, chain, status="failed", error_message="boom")

    counts = await processing_service.status_counts_since(db, since=since)
    jobs = await processing_service.list_jobs(db, status=None, since=since, limit=500)

    listed = {job["id"] for job in jobs}
    assert chain.job_id in listed
    assert counts.get("failed", 0) == sum(1 for j in jobs if j["status"] == "failed")


async def test_the_until_bound_excludes_later_jobs(
    db: AsyncSession, engine: AsyncEngine, chain: Chain
) -> None:
    """``until`` is what makes a historical window closed rather than open."""
    created = datetime.now(tz=UTC) - timedelta(hours=2)
    await _insert_job(engine, chain, status="failed", created_at=created)

    inside = await processing_service.list_jobs(
        db,
        status=None,
        since=created - timedelta(minutes=5),
        until=created + timedelta(minutes=5),
        limit=500,
    )
    assert chain.job_id in {j["id"] for j in inside}

    before_it = await processing_service.list_jobs(
        db,
        status=None,
        since=created - timedelta(hours=1),
        until=created - timedelta(minutes=5),
        limit=500,
    )
    assert chain.job_id not in {j["id"] for j in before_it}


async def test_list_jobs_honours_the_status_filter(
    db: AsyncSession, engine: AsyncEngine, chain: Chain
) -> None:
    since = datetime.now(tz=UTC) - timedelta(hours=1)
    await _insert_job(engine, chain, status="failed", error_message="boom")

    failed = await processing_service.list_jobs(db, status="failed", since=since, limit=500)
    assert chain.job_id in {j["id"] for j in failed}
    assert all(j["status"] == "failed" for j in failed)

    completed = await processing_service.list_jobs(db, status="completed", since=since, limit=500)
    assert chain.job_id not in {j["id"] for j in completed}


async def test_list_jobs_honours_its_limit(
    db: AsyncSession, engine: AsyncEngine, chain: Chain
) -> None:
    await _insert_job(engine, chain, status="failed")
    jobs = await processing_service.list_jobs(
        db, status=None, since=datetime.now(tz=UTC) - timedelta(hours=1), limit=1
    )
    assert len(jobs) <= 1
