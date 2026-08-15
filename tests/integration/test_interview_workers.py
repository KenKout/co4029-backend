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
from unittest.mock import ANY, AsyncMock
from uuid import UUID

import pytest
from arq import Retry

from abridgeai.ai.llm.errors import ProviderError
from abridgeai.core.audit import current_actor_var
from abridgeai.features.interviews.workers import (
    EVALUATION_MAX_TRIES,
    JOBS,
    evaluate_interview_session_task,
    reconcile_turn_analysis_task,
    run_interview_generation_task,
)
from abridgeai.features.interviews.workers import analysis as analysis_worker_mod
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
    # Was 4 before commit 4a3fffd removed the practice-mode job alongside the
    # 0072 feature drop; 3 tasks remain (generation, evaluation, turn-reconcile).
    assert len(JOBS) == 3
    assert reconcile_turn_analysis_task in JOBS
    assert run_interview_generation_task in JOBS
    evaluation_job = next(job for job in JOBS if getattr(job, "coroutine", None))
    assert evaluation_job.coroutine is evaluate_interview_session_task
    assert evaluation_job.max_tries == EVALUATION_MAX_TRIES


def test_arq_app_includes_interview_jobs() -> None:
    assert run_interview_generation_task in WorkerSettings.functions
    evaluation_job = next(
        job
        for job in WorkerSettings.functions
        if getattr(job, "coroutine", None) is evaluate_interview_session_task
    )
    assert evaluation_job.max_tries == EVALUATION_MAX_TRIES


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

    async def _capture(_db: Any, _run_id: UUID, **_kwargs: Any) -> None:
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

    async def _capture(_db: Any, _session_id: UUID, **_kwargs: Any) -> None:
        seen["actor"] = current_actor_var.get()

    monkeypatch.setattr(
        eval_worker_mod.evaluation_service, "evaluate_and_generate_report", _capture
    )
    monkeypatch.setattr(eval_worker_mod, "get_sessionmaker", lambda: _FakeSessionmaker())

    await evaluate_interview_session_task(ctx={}, actor_id=actor_id, session_id=uuid.uuid4())

    assert seen["actor"] == actor_id
    assert current_actor_var.get() is None


@pytest.mark.asyncio
async def test_evaluate_interview_session_task_final_attempt_detection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The production ARQ context's final job_try must mark the final attempt.

    Without this, a session stuck after the last ARQ retry never gets its
    terminal 'failed' status stamped and the student-facing poll in
    course-interview.tsx waits forever for a pass_verdict that never arrives.
    """
    service_mock = AsyncMock()
    monkeypatch.setattr(
        eval_worker_mod.evaluation_service, "evaluate_and_generate_report", service_mock
    )
    monkeypatch.setattr(eval_worker_mod, "get_sessionmaker", lambda: _FakeSessionmaker())

    session_id = uuid.uuid4()
    await evaluate_interview_session_task(
        ctx={"job_try": EVALUATION_MAX_TRIES},
        actor_id=uuid.uuid4(),
        session_id=session_id,
    )

    service_mock.assert_awaited_once_with(ANY, session_id, is_final_attempt=True)


@pytest.mark.asyncio
async def test_evaluate_interview_session_task_non_final_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """job_try below the configured budget is not the final attempt."""
    service_mock = AsyncMock()
    monkeypatch.setattr(
        eval_worker_mod.evaluation_service, "evaluate_and_generate_report", service_mock
    )
    monkeypatch.setattr(eval_worker_mod, "get_sessionmaker", lambda: _FakeSessionmaker())

    session_id = uuid.uuid4()
    await evaluate_interview_session_task(
        ctx={"job_try": 1}, actor_id=uuid.uuid4(), session_id=session_id
    )

    service_mock.assert_awaited_once_with(ANY, session_id, is_final_attempt=False)


@pytest.mark.asyncio
async def test_evaluate_interview_session_task_retries_on_transient_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A NON-final attempt that fails must raise arq's ``Retry`` (not the raw
    exception) so the job is re-enqueued and the AI judge gets another try.

    arq 0.28 only re-queues on ``Retry``/``CancelledError``/``RetryJob`` — a
    bare re-raise here would burn the whole 'max_tries' budget on attempt 1
    without ever retrying. The service still ran with is_final_attempt=False.
    """
    service_mock = AsyncMock(side_effect=ProviderError("judge 503"))
    monkeypatch.setattr(
        eval_worker_mod.evaluation_service, "evaluate_and_generate_report", service_mock
    )
    monkeypatch.setattr(eval_worker_mod, "get_sessionmaker", lambda: _FakeSessionmaker())

    session_id = uuid.uuid4()
    with pytest.raises(Retry):
        await evaluate_interview_session_task(
            ctx={"job_try": 1}, actor_id=uuid.uuid4(), session_id=session_id
        )

    # Service was invoked as a non-final attempt (so it did NOT stamp
    # terminal status='failed' — a later retry can still succeed).
    service_mock.assert_awaited_once_with(ANY, session_id, is_final_attempt=False)
    assert current_actor_var.get() is None


@pytest.mark.asyncio
async def test_evaluate_interview_session_task_retry_defer_grows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Retry defer must grow with job_try (exponential backoff)."""
    monkeypatch.setattr(
        eval_worker_mod.evaluation_service,
        "evaluate_and_generate_report",
        AsyncMock(side_effect=ProviderError("judge 503")),
    )
    monkeypatch.setattr(eval_worker_mod, "get_sessionmaker", lambda: _FakeSessionmaker())

    defers: dict[int, float | None] = {}
    for job_try in (1, 2):
        try:
            await evaluate_interview_session_task(
                ctx={"job_try": job_try},
                actor_id=uuid.uuid4(),
                session_id=uuid.uuid4(),
            )
        except Retry as retry:
            # Retry.defer_score is in milliseconds.
            defers[job_try] = retry.defer_score

    assert defers[1] is not None
    assert defers[2] is not None
    assert defers[2] > defers[1]  # backoff grows


@pytest.mark.asyncio
async def test_evaluate_interview_session_task_final_attempt_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """On the configured FINAL attempt the original exception must
    propagate (NOT Retry) so arq records a terminal job failure.

    The service was called with is_final_attempt=True, so it already stamped
    the session status='failed' — the student-facing poll stops waiting.
    """
    service_mock = AsyncMock(side_effect=ProviderError("judge 503"))
    monkeypatch.setattr(
        eval_worker_mod.evaluation_service, "evaluate_and_generate_report", service_mock
    )
    monkeypatch.setattr(eval_worker_mod, "get_sessionmaker", lambda: _FakeSessionmaker())

    session_id = uuid.uuid4()
    with pytest.raises(ProviderError, match="judge 503"):
        await evaluate_interview_session_task(
            ctx={"job_try": EVALUATION_MAX_TRIES},
            actor_id=uuid.uuid4(),
            session_id=session_id,
        )

    service_mock.assert_awaited_once_with(ANY, session_id, is_final_attempt=True)
    assert current_actor_var.get() is None


def test_reconcile_turn_analysis_task_signature() -> None:
    sig = inspect.signature(reconcile_turn_analysis_task)
    params = list(sig.parameters)
    assert params[0] == "ctx"
    assert params[1] == "actor_id"


def test_arq_app_includes_reconcile_turn_analysis_task() -> None:
    assert reconcile_turn_analysis_task in WorkerSettings.functions


@pytest.mark.asyncio
async def test_reconcile_turn_analysis_task_parses_payload_and_calls_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service_mock = AsyncMock()
    monkeypatch.setattr(
        analysis_worker_mod.reconciliation_service, "reconcile_turn_analysis", service_mock
    )
    fake_factory = _FakeSessionmaker()
    monkeypatch.setattr(analysis_worker_mod, "get_sessionmaker", lambda: fake_factory)

    session_id = uuid.uuid4()
    message_id = uuid.uuid4()
    question_id = uuid.uuid4()
    await reconcile_turn_analysis_task(
        ctx={},
        actor_id=uuid.uuid4(),
        session_id=session_id,
        turn={
            "message_id": str(message_id),
            "question_id": str(question_id),
            "turn_id": "sq-1",
            "probe_verdict": {
                "sufficient": True,
                "outcome_ids_touched": ["o-1"],
                "confidence": 0.9,
            },
        },
    )

    assert fake_factory.calls == 1
    service_mock.assert_awaited_once()
    request = service_mock.await_args.args[1]
    assert request.session_id == session_id
    assert request.message_id == message_id
    assert request.question_id == question_id
    assert request.turn_id == "sq-1"
    assert request.probe_verdict is not None
    assert request.probe_verdict.sufficient is True
    assert current_actor_var.get() is None


@pytest.mark.asyncio
async def test_reconcile_turn_analysis_task_sets_actor_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actor_id = uuid.uuid4()
    seen: dict[str, UUID | None] = {"actor": None}

    async def _capture(_db: Any, _request: Any) -> None:
        seen["actor"] = current_actor_var.get()

    monkeypatch.setattr(
        analysis_worker_mod.reconciliation_service, "reconcile_turn_analysis", _capture
    )
    monkeypatch.setattr(analysis_worker_mod, "get_sessionmaker", lambda: _FakeSessionmaker())

    await reconcile_turn_analysis_task(
        ctx={},
        actor_id=actor_id,
        session_id=uuid.uuid4(),
        turn={"message_id": str(uuid.uuid4()), "turn_id": "sq-1"},
    )

    assert seen["actor"] == actor_id
    assert current_actor_var.get() is None


@pytest.mark.asyncio
async def test_reconcile_turn_analysis_task_swallows_a_malformed_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A payload that cannot be parsed cannot be fixed by retrying it."""
    service_mock = AsyncMock()
    monkeypatch.setattr(
        analysis_worker_mod.reconciliation_service, "reconcile_turn_analysis", service_mock
    )
    fake_factory = _FakeSessionmaker()
    monkeypatch.setattr(analysis_worker_mod, "get_sessionmaker", lambda: fake_factory)

    await reconcile_turn_analysis_task(
        ctx={}, actor_id=uuid.uuid4(), session_id=uuid.uuid4(), turn={"turn_id": "sq-1"}
    )

    service_mock.assert_not_awaited()
    assert fake_factory.calls == 0
    assert current_actor_var.get() is None
