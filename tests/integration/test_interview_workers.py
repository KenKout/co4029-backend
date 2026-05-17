"""Interview ARQ worker tests (T6.13).

Both workers are thin wrappers — pipeline / evaluation error handling
lives in the services they delegate to. The worker's responsibilities:

* Set ``current_actor_var`` from ``actor_id`` so audit columns get
  filled via the SQLAlchemy ``before_flush`` listener.
* Bind structured-log context with ``bind_request_context`` and tear
  down in ``finally`` so neighbouring tasks never see leaked state.
* Open an ``AsyncSession`` from the global sessionmaker and pass it
  to the service entrypoint.
"""

from __future__ import annotations

import inspect
import uuid
from typing import Any
from unittest.mock import AsyncMock
from uuid import UUID

import pytest

from abridgeai.core.audit import current_actor_var
from abridgeai.features.interviews.workers import (
    JOBS,
    evaluate_interview_session_task,
    run_interview_generation_task,
)
from abridgeai.features.interviews.workers import evaluation as eval_worker_mod
from abridgeai.features.interviews.workers import generation as gen_worker_mod
from abridgeai.workers.arq_app import WorkerSettings


class _FakeSession:
    async def __aenter__(self) -> _FakeSession:
        return self

    async def __aexit__(self, *_: Any) -> None:
        return None


class _FakeSessionmaker:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self) -> _FakeSession:
        self.calls += 1
        return _FakeSession()


def test_jobs_export() -> None:
    assert len(JOBS) == 2
    assert run_interview_generation_task in JOBS
    assert evaluate_interview_session_task in JOBS


def test_arq_app_includes_interview_jobs() -> None:
    assert run_interview_generation_task in WorkerSettings.functions
    assert evaluate_interview_session_task in WorkerSettings.functions


def test_run_interview_generation_task_signature() -> None:
    sig = inspect.signature(run_interview_generation_task)
    params = list(sig.parameters)
    assert params[0] == "ctx"
    assert params[1] == "actor_id"


def test_evaluate_interview_session_task_signature() -> None:
    sig = inspect.signature(evaluate_interview_session_task)
    params = list(sig.parameters)
    assert params[0] == "ctx"
    assert params[1] == "actor_id"


@pytest.mark.asyncio
async def test_run_interview_generation_task_calls_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline_mock = AsyncMock()
    monkeypatch.setattr(
        gen_worker_mod.generation_service, "run_interview_generation", pipeline_mock
    )
    fake_factory = _FakeSessionmaker()
    monkeypatch.setattr(gen_worker_mod, "get_sessionmaker", lambda: fake_factory)

    actor_id = uuid.uuid4()
    run_id = uuid.uuid4()

    await run_interview_generation_task(ctx={}, actor_id=actor_id, generation_run_id=run_id)

    assert fake_factory.calls == 1
    pipeline_mock.assert_awaited_once()
    call_args = pipeline_mock.await_args
    assert call_args is not None
    _, passed_run_id = call_args.args
    assert passed_run_id == run_id


@pytest.mark.asyncio
async def test_run_interview_generation_task_sets_actor_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actor_id = uuid.uuid4()
    seen: dict[str, UUID | None] = {"actor": None}

    async def _capture(_db: Any, _run_id: UUID) -> None:
        seen["actor"] = current_actor_var.get()

    monkeypatch.setattr(gen_worker_mod.generation_service, "run_interview_generation", _capture)
    monkeypatch.setattr(gen_worker_mod, "get_sessionmaker", lambda: _FakeSessionmaker())

    await run_interview_generation_task(ctx={}, actor_id=actor_id, generation_run_id=uuid.uuid4())

    assert seen["actor"] == actor_id
    assert current_actor_var.get() is None


@pytest.mark.asyncio
async def test_run_interview_generation_task_propagates_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline_mock = AsyncMock(side_effect=RuntimeError("pipeline blew up"))
    monkeypatch.setattr(
        gen_worker_mod.generation_service, "run_interview_generation", pipeline_mock
    )
    monkeypatch.setattr(gen_worker_mod, "get_sessionmaker", lambda: _FakeSessionmaker())

    with pytest.raises(RuntimeError, match="pipeline blew up"):
        await run_interview_generation_task(
            ctx={}, actor_id=uuid.uuid4(), generation_run_id=uuid.uuid4()
        )

    assert current_actor_var.get() is None


@pytest.mark.asyncio
async def test_evaluate_interview_session_task_calls_evaluation_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service_mock = AsyncMock()
    monkeypatch.setattr(
        eval_worker_mod.evaluation_service, "evaluate_and_generate_report", service_mock
    )
    fake_factory = _FakeSessionmaker()
    monkeypatch.setattr(eval_worker_mod, "get_sessionmaker", lambda: fake_factory)

    actor_id = uuid.uuid4()
    session_id = uuid.uuid4()

    await evaluate_interview_session_task(ctx={}, actor_id=actor_id, session_id=session_id)

    assert fake_factory.calls == 1
    service_mock.assert_awaited_once()
    call_args = service_mock.await_args
    assert call_args is not None
    _, passed_session_id = call_args.args
    assert passed_session_id == session_id


@pytest.mark.asyncio
async def test_evaluate_interview_session_task_sets_actor_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actor_id = uuid.uuid4()
    seen: dict[str, UUID | None] = {"actor": None}

    async def _capture(_db: Any, _session_id: UUID) -> None:
        seen["actor"] = current_actor_var.get()

    monkeypatch.setattr(
        eval_worker_mod.evaluation_service, "evaluate_and_generate_report", _capture
    )
    monkeypatch.setattr(eval_worker_mod, "get_sessionmaker", lambda: _FakeSessionmaker())

    await evaluate_interview_session_task(ctx={}, actor_id=actor_id, session_id=uuid.uuid4())

    assert seen["actor"] == actor_id
    assert current_actor_var.get() is None


@pytest.mark.asyncio
async def test_evaluate_interview_session_task_propagates_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service_mock = AsyncMock(side_effect=RuntimeError("evaluation blew up"))
    monkeypatch.setattr(
        eval_worker_mod.evaluation_service, "evaluate_and_generate_report", service_mock
    )
    monkeypatch.setattr(eval_worker_mod, "get_sessionmaker", lambda: _FakeSessionmaker())

    with pytest.raises(RuntimeError, match="evaluation blew up"):
        await evaluate_interview_session_task(
            ctx={}, actor_id=uuid.uuid4(), session_id=uuid.uuid4()
        )

    assert current_actor_var.get() is None
