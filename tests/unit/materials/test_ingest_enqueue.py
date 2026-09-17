"""The enqueue that is not allowed to silently do nothing.

This helper exists because of a specific production failure: the request
path guarded the enqueue with a bare ``if arq_pool is not None``. When the
injected pool was ``None`` -- app lifespan not yet finished wiring the
override, or a Redis blip at startup -- the enqueue was skipped, the
``ProcessingJob`` row committed as ``pending``, and nothing ever consumed
it. The teacher's document sat at "pending" indefinitely and nothing in the
logs or the UI said why.

Both halves of the fix are load-bearing and neither is visible from the
call site:

* a missing pool is replaced, not treated as permission to skip;
* a failing enqueue is logged and re-raised, so the request fails in front
  of the teacher instead of committing an orphan.

No Redis: the pool is a stub and the fallback constructor is patched.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import Mock
from uuid import uuid4

import pytest

from abridgeai.features.materials.workers import enqueue as enqueue_module
from abridgeai.features.materials.workers.enqueue import enqueue_material_ingest


class _FakePool:
    def __init__(self, fail_with: Exception | None = None) -> None:
        self._fail_with = fail_with
        self.jobs: list[tuple[Any, ...]] = []

    async def enqueue_job(self, *args: Any) -> None:
        if self._fail_with:
            raise self._fail_with
        self.jobs.append(args)


@pytest.fixture(autouse=True)
def _reset_fallback_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    """The fallback pool is a module-level global, cached across calls.

    Left set by one test it would satisfy the next one's assertion without
    the code under test doing anything, so it is reset around every test.
    """
    monkeypatch.setattr(enqueue_module, "_fallback_pool", None)


@pytest.fixture
def logger(monkeypatch: pytest.MonkeyPatch) -> Mock:
    recorder = Mock()
    monkeypatch.setattr(enqueue_module, "_logger", recorder)
    return recorder


def _patch_fallback(monkeypatch: pytest.MonkeyPatch, *pools: _FakePool) -> list[_FakePool]:
    """Patch ``create_pool`` to hand out ``pools`` in order, recording calls."""
    created: list[_FakePool] = []
    queue = list(pools)

    async def _create_pool(_settings: object) -> _FakePool:
        pool = queue.pop(0) if queue else _FakePool()
        created.append(pool)
        return pool

    monkeypatch.setattr(enqueue_module, "create_pool", _create_pool)
    return created


async def test_the_injected_pool_is_used_with_the_canonical_task_name() -> None:
    """The name must match the function registered on the ARQ worker.

    A typo here enqueues a job no worker claims, which fails exactly like
    the bug this module was written to fix: silently, and visible only as a
    document that never leaves "pending".
    """
    pool = _FakePool()
    actor, version, run = uuid4(), uuid4(), uuid4()

    await enqueue_material_ingest(
        pool, actor_id=actor, material_version_id=version, pipeline_run_id=run
    )

    assert pool.jobs == [("ingest_material_version_task", actor, version, run)]


async def test_the_actor_leads_the_arguments() -> None:
    """Phase 0.8 convention: ``actor_id`` comes first after the task name, so
    the worker can bind it before touching anything auditable. Reordering
    these would attribute the ingest's audit rows to a material id.
    """
    pool = _FakePool()
    actor = uuid4()
    await enqueue_material_ingest(
        pool, actor_id=actor, material_version_id=uuid4(), pipeline_run_id=uuid4()
    )
    assert pool.jobs[0][1] == actor


async def test_a_missing_pool_is_replaced_rather_than_skipped(
    monkeypatch: pytest.MonkeyPatch, logger: Mock
) -> None:
    """The heart of the fix. ``None`` used to mean "do not enqueue"."""
    fallback = _FakePool()
    _patch_fallback(monkeypatch, fallback)

    await enqueue_material_ingest(
        None, actor_id=uuid4(), material_version_id=uuid4(), pipeline_run_id=uuid4()
    )

    assert len(fallback.jobs) == 1, "the job is enqueued on the fallback, not dropped"


async def test_the_fallback_pool_is_built_once_and_reused(
    monkeypatch: pytest.MonkeyPatch, logger: Mock
) -> None:
    """A pool per enqueue would open a Redis connection per upload."""
    created = _patch_fallback(monkeypatch)

    for _ in range(3):
        await enqueue_material_ingest(
            None, actor_id=uuid4(), material_version_id=uuid4(), pipeline_run_id=uuid4()
        )

    assert len(created) == 1
    assert len(created[0].jobs) == 3


async def test_falling_back_is_recorded_as_a_warning(
    monkeypatch: pytest.MonkeyPatch, logger: Mock
) -> None:
    """Working is not the same as healthy: the fallback means dependency
    injection did not supply a pool, which is worth finding in the logs
    before it becomes something worse.
    """
    _patch_fallback(monkeypatch)
    version = uuid4()

    await enqueue_material_ingest(
        None, actor_id=uuid4(), material_version_id=version, pipeline_run_id=uuid4()
    )

    logger.warning.assert_called_once()
    event, fields = logger.warning.call_args.args[0], logger.warning.call_args.kwargs
    assert event == "materials_ingest_enqueue_pool_fallback"
    assert fields["material_version_id"] == str(version)


async def test_a_supplied_pool_warns_about_nothing(logger: Mock) -> None:
    """The normal path is silent, or the warning stops meaning anything."""
    await enqueue_material_ingest(
        _FakePool(), actor_id=uuid4(), material_version_id=uuid4(), pipeline_run_id=uuid4()
    )
    logger.warning.assert_not_called()


async def test_a_failing_enqueue_reaches_the_caller(logger: Mock) -> None:
    """Swallowing this would recreate the original bug by another route.

    The caller is mid-request with an uncommitted ``ProcessingJob``; the
    raise is what makes the request fail and the row roll back, instead of
    committing a job nothing will ever pick up.
    """
    pool = _FakePool(fail_with=ConnectionError("redis is gone"))

    with pytest.raises(ConnectionError):
        await enqueue_material_ingest(
            pool, actor_id=uuid4(), material_version_id=uuid4(), pipeline_run_id=uuid4()
        )

    assert logger.exception.call_args.args[0] == "materials_ingest_enqueue_failed"


async def test_the_failure_log_names_the_orphaned_work(logger: Mock) -> None:
    """Both ids are needed to find the stranded job row afterwards."""
    version, run = uuid4(), uuid4()
    pool = _FakePool(fail_with=ConnectionError("down"))

    with pytest.raises(ConnectionError):
        await enqueue_material_ingest(
            pool, actor_id=uuid4(), material_version_id=version, pipeline_run_id=run
        )

    fields = logger.exception.call_args.kwargs
    assert fields["material_version_id"] == str(version)
    assert fields["pipeline_run_id"] == str(run)


async def test_a_fallback_that_also_fails_still_raises(
    monkeypatch: pytest.MonkeyPatch, logger: Mock
) -> None:
    """Redis being unreachable is precisely when both paths fail, and it is
    the case where a silent skip would be most tempting and most harmful."""
    _patch_fallback(monkeypatch, _FakePool(fail_with=ConnectionError("down")))

    with pytest.raises(ConnectionError):
        await enqueue_material_ingest(
            None, actor_id=uuid4(), material_version_id=uuid4(), pipeline_run_id=uuid4()
        )
