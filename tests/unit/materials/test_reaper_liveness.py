"""The liveness scan the reaper decides life and death from.

The reaper's whole job is to tell a dead ingest from a slow one, and this
scan is the only evidence it has. An ``arq:job:<id>`` key exists in Redis
for a job's entire lifecycle -- queued, running, waiting to retry -- and is
deleted when it finishes, so "absent from this set" is what the reaper
reads as "nothing is ever going to consume this row".

Both ways of getting it wrong are expensive, and they are not symmetric:

* A **false orphan** (a live job the scan fails to see) gets re-enqueued
  alongside the original. For a quiz or interview run that means a second
  set of LLM calls racing the first, on the teacher's budget, both writing
  the same row. This is why the scanners coerce a stringified UUID instead
  of skipping it, and why a Redis error aborts the tick rather than
  returning an empty set -- an empty set means *every* row looks orphaned,
  so a Redis blip would re-enqueue the entire backlog at once.
* A **missed orphan** costs one more cron interval. The row is picked up on
  the next tick.

So the scan is deliberately biased: when it cannot tell, it says "alive".
The tests below pin that bias at each point where the code chooses it.

Pure functions over a fake Redis - no database, no worker, no network.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID, uuid4

import pytest
from arq.jobs import serialize_job

from abridgeai.features.materials.workers.reaper import (
    _coerce_uuid,
    _live_interview_run_ids,
    _live_quiz_run_ids,
    _live_version_ids,
)

_INGEST_TASK = "ingest_material_version_task"
_QUIZ_TASK = "run_quiz_generation_task"
_INTERVIEW_TASK = "run_interview_generation_task"


class _FakeRedis:
    """Only the two calls the scanners make, plus a way to inject failure.

    Jobs are stored under real ``arq:job:<id>`` keys using ARQ's own
    serializer, so the reaper's ``deserialize_job_raw`` parses exactly the
    bytes a real worker would have written.
    """

    def __init__(self, *, fail_on: Exception | None = None) -> None:
        self._jobs: dict[bytes, bytes | None] = {}
        self._fail_on = fail_on

    def add_job(self, function_name: str, args: tuple[Any, ...]) -> bytes:
        key = f"arq:job:{uuid4().hex}".encode()
        self._jobs[key] = serialize_job(function_name, args, {}, 1, 0)
        return key

    def add_raw(self, payload: bytes | None) -> bytes:
        """A key holding something that is not a serialized job (or nothing)."""
        key = f"arq:job:{uuid4().hex}".encode()
        self._jobs[key] = payload
        return key

    async def keys(self, _pattern: str) -> list[bytes]:
        if self._fail_on:
            raise self._fail_on
        return list(self._jobs)

    async def get(self, key: bytes) -> bytes | None:
        if self._fail_on:
            raise self._fail_on
        return self._jobs.get(key)


class TestWithoutRedisNothingIsAlive:
    """``None`` is what the caller passes when the worker gave it no pool.

    Every scanner returns an empty set here, which on its own would make the
    whole backlog look orphaned. The quiz and interview reconcilers therefore
    refuse to run at all without a pool; only the ingest reaper proceeds,
    because its re-enqueue helper can build its own pool.
    """

    async def test_the_ingest_scan_is_empty(self) -> None:
        assert await _live_version_ids(None) == set()

    async def test_the_quiz_scan_is_empty(self) -> None:
        assert await _live_quiz_run_ids(None) == set()

    async def test_the_interview_scan_is_empty(self) -> None:
        assert await _live_interview_run_ids(None) == set()


class TestReadingLiveIngestJobs:
    async def test_a_queued_ingest_protects_its_version(self) -> None:
        """``ingest_material_version_task(ctx, actor, version, run)`` stores
        its args without ``ctx``, so the version id is ``args[1]``."""
        redis = _FakeRedis()
        version_id = uuid4()
        redis.add_job(_INGEST_TASK, (uuid4(), version_id, uuid4()))

        assert await _live_version_ids(redis) == {version_id}

    async def test_several_jobs_are_all_collected(self) -> None:
        redis = _FakeRedis()
        first, second = uuid4(), uuid4()
        redis.add_job(_INGEST_TASK, (uuid4(), first, uuid4()))
        redis.add_job(_INGEST_TASK, (uuid4(), second, uuid4()))

        assert await _live_version_ids(redis) == {first, second}

    async def test_an_empty_queue_yields_an_empty_set(self) -> None:
        assert await _live_version_ids(_FakeRedis()) == set()

    async def test_a_job_with_too_few_arguments_is_skipped(self) -> None:
        """Another task's payload can sit in the same keyspace."""
        redis = _FakeRedis()
        redis.add_job("some_other_task", (uuid4(),))
        assert await _live_version_ids(redis) == set()

    async def test_a_non_uuid_second_argument_is_skipped(self) -> None:
        redis = _FakeRedis()
        redis.add_job("some_other_task", (uuid4(), "not-a-uuid", 3))
        assert await _live_version_ids(redis) == set()

    async def test_the_ingest_scan_does_not_filter_by_task_name(self) -> None:
        """Deliberate, and safe in the direction that matters.

        Any job whose second argument is a UUID contributes to the set, so a
        quiz run's id can land in the ingest live-set. Because ids never
        collide, that can only ever *protect* a row from being reaped, never
        expose one -- the scan errs towards "alive", which is the cheap
        mistake. Pinned so a future filter is added on purpose rather than
        as a cleanup.
        """
        redis = _FakeRedis()
        foreign_id = uuid4()
        redis.add_job(_QUIZ_TASK, (uuid4(), foreign_id))

        assert await _live_version_ids(redis) == {foreign_id}

    async def test_a_malformed_payload_does_not_blind_the_sweep(self) -> None:
        """One unreadable key must not hide every other live job.

        If a bad payload aborted the scan, the remaining live jobs would be
        invisible and their rows would be reaped as orphans.
        """
        redis = _FakeRedis()
        redis.add_raw(b"this is not a serialized arq job")
        version_id = uuid4()
        redis.add_job(_INGEST_TASK, (uuid4(), version_id, uuid4()))

        assert await _live_version_ids(redis) == {version_id}

    async def test_a_key_that_vanished_mid_scan_is_skipped(self) -> None:
        """``keys`` then ``get`` is not atomic: a job can finish in between,
        and the read comes back empty."""
        redis = _FakeRedis()
        redis.add_raw(None)
        version_id = uuid4()
        redis.add_job(_INGEST_TASK, (uuid4(), version_id, uuid4()))

        assert await _live_version_ids(redis) == {version_id}

    async def test_a_redis_failure_is_raised_not_swallowed(self) -> None:
        """The most consequential line in the module.

        Returning an empty set on error would tell the reaper that nothing
        is alive, and it would re-enqueue every pending and running ingest in
        the backlog at once. Raising makes the caller abort the tick instead,
        so a Redis blip costs one cron interval and nothing else.
        """
        redis = _FakeRedis(fail_on=ConnectionError("redis is down"))
        with pytest.raises(ConnectionError):
            await _live_version_ids(redis)


class TestReadingLiveGenerationRuns:
    """The quiz and interview scans share a shape but must not share results.

    Both read ``args[1]`` of ``(actor_id, generation_run_id)``, so the task
    name is the only thing separating them. Without that filter a live quiz
    run would protect an interview run and vice versa -- and worse, an id
    absent from its own scan would be reaped while its job was still running.
    """

    @pytest.mark.parametrize(
        ("scan", "own_task", "other_task"),
        [
            (_live_quiz_run_ids, _QUIZ_TASK, _INTERVIEW_TASK),
            (_live_interview_run_ids, _INTERVIEW_TASK, _QUIZ_TASK),
        ],
    )
    async def test_only_its_own_task_counts(
        self, scan: Any, own_task: str, other_task: str
    ) -> None:
        redis = _FakeRedis()
        mine, theirs = uuid4(), uuid4()
        redis.add_job(own_task, (uuid4(), mine))
        redis.add_job(other_task, (uuid4(), theirs))
        redis.add_job(_INGEST_TASK, (uuid4(), uuid4(), uuid4()))

        assert await scan(redis) == {mine}

    @pytest.mark.parametrize(
        ("scan", "task"),
        [(_live_quiz_run_ids, _QUIZ_TASK), (_live_interview_run_ids, _INTERVIEW_TASK)],
    )
    async def test_a_run_id_serialized_as_a_string_still_counts(
        self, scan: Any, task: str
    ) -> None:
        """ARQ stores whatever the enqueue site passed, so the id arrives as
        a ``UUID`` or as its string form depending on the call site.

        Reading only the ``UUID`` form would mark a live run an orphan, and
        the reaper would start a second generation racing the first -- two
        sets of LLM calls writing the same row.
        """
        redis = _FakeRedis()
        run_id = uuid4()
        redis.add_job(task, (uuid4(), str(run_id)))

        assert await scan(redis) == {run_id}

    @pytest.mark.parametrize(
        ("scan", "task"),
        [(_live_quiz_run_ids, _QUIZ_TASK), (_live_interview_run_ids, _INTERVIEW_TASK)],
    )
    async def test_an_unparseable_run_id_is_skipped(self, scan: Any, task: str) -> None:
        redis = _FakeRedis()
        redis.add_job(task, (uuid4(), "definitely-not-a-uuid"))
        assert await scan(redis) == set()

    @pytest.mark.parametrize(
        ("scan", "task"),
        [(_live_quiz_run_ids, _QUIZ_TASK), (_live_interview_run_ids, _INTERVIEW_TASK)],
    )
    async def test_one_bad_payload_does_not_hide_the_live_runs(
        self, scan: Any, task: str
    ) -> None:
        redis = _FakeRedis()
        redis.add_raw(b"garbage")
        run_id = uuid4()
        redis.add_job(task, (uuid4(), run_id))

        assert await scan(redis) == {run_id}

    @pytest.mark.parametrize("scan", [_live_quiz_run_ids, _live_interview_run_ids])
    async def test_a_redis_failure_propagates(self, scan: Any) -> None:
        """Both reconcilers catch this and abort the tick. Same reasoning as
        the ingest scan: an empty set would reap the whole backlog."""
        redis = _FakeRedis(fail_on=ConnectionError("redis is down"))
        with pytest.raises(ConnectionError):
            await scan(redis)


class TestCoercingIdsOutOfConfigJson:
    """``config_json`` is JSON, so every id inside it is a string.

    The coerced id becomes the deep-link on the failure notification the
    teacher receives. A bad value has to become ``None`` -- a link to
    nowhere is better than a crash inside the notify loop, which would take
    the remaining teachers' notifications with it.
    """

    def test_a_uuid_passes_through(self) -> None:
        value = uuid4()
        assert _coerce_uuid(value) is value

    def test_a_string_is_parsed(self) -> None:
        value = uuid4()
        assert _coerce_uuid(str(value)) == value

    @pytest.mark.parametrize("value", ["", "not-a-uuid", "1234"])
    def test_an_unparseable_string_is_none(self, value: str) -> None:
        assert _coerce_uuid(value) is None

    @pytest.mark.parametrize("value", [None, 42, 3.5, [], {}, True])
    def test_a_non_string_non_uuid_is_none(self, value: object) -> None:
        """``config_json`` is teacher-influenced upstream, so the type is not
        guaranteed."""
        assert _coerce_uuid(value) is None

    def test_a_missing_key_reads_as_none(self) -> None:
        """The call sites pass ``config.get(...)`` straight in."""
        assert _coerce_uuid({}.get("quiz_id")) is None


def test_the_reaper_exposes_only_its_task() -> None:
    """The scanners are internals; the cron entry point is the contract."""
    from abridgeai.features.materials.workers import reaper

    assert reaper.__all__ == ["reconcile_orphaned_ingests_task"]


def test_the_ingest_budget_is_larger_than_the_generation_budget() -> None:
    """An ingest re-run is cheap and deterministic -- extract, chunk, embed.

    A generation re-run spends LLM calls on the teacher's budget, so it gets
    fewer attempts before the reaper gives up and says so.
    """
    from abridgeai.features.materials.workers import reaper

    assert reaper._MAX_REQUEUE_ATTEMPTS > reaper._QUIZ_MAX_REQUEUE_ATTEMPTS
    assert reaper._QUIZ_MAX_REQUEUE_ATTEMPTS == reaper._INTERVIEW_MAX_REQUEUE_ATTEMPTS


def test_generation_runs_get_a_longer_grace_than_ingests() -> None:
    """A generation run legitimately sits quiet for minutes between LLM
    calls; an ingest reports progress continuously. Reaping a generation run
    on the ingest's grace period would kill healthy runs mid-flight.
    """
    from abridgeai.features.materials.workers import reaper

    assert reaper._QUIZ_ORPHAN_GRACE_SECONDS > reaper._ORPHAN_GRACE_SECONDS
    assert reaper._INTERVIEW_ORPHAN_GRACE_SECONDS > reaper._ORPHAN_GRACE_SECONDS
