"""P2 regression: transient generation failures retry; permanent ones do not.

Audit P2: the generation worker used to re-raise every exception, so a
single transient provider hiccup terminally failed the run (and emailed the
teacher a dead error). Now: transient failures with retry budget remaining
put the run back to ``pending`` (re-claimable) and the worker raises
``arq.worker.Retry``; permanent failures stay terminal on the first attempt;
the final attempt is always terminal.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import pytest

from abridgeai.ai.llm.errors import ProviderError, ResponseFormatError
from abridgeai.features.interviews.services.generation_retry import (
    is_transient_generation_failure,
)


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (ProviderError("HTTP error calling https://x: ConnectError: boom"), True),
        (ProviderError("/chat/completions returned HTTP 503: busy"), True),
        (ProviderError("/chat/completions returned HTTP 429 after 5 attempts"), True),
        (ConnectionError("reset by peer"), True),
        (TimeoutError("upstream timeout"), True),
        (ProviderError("/chat/completions returned HTTP 401: bad key"), False),
        (ProviderError("/chat/completions returned HTTP 400: bad schema"), False),
        (ResponseFormatError("non-JSON content"), False),
        (ValueError("targeted generation resolved no interview outcomes"), False),
        (RuntimeError("targeted generation resolved no interview outcomes"), False),
    ],
)
def test_transient_classification(exc: Exception, expected: bool) -> None:
    assert is_transient_generation_failure(exc) is expected


class _FakePipeline:
    """Runs the wrapped task body with a scripted service outcome."""

    def __init__(
        self,
        monkeypatch: pytest.MonkeyPatch,
        outcome: Exception | None,
        *,
        job_try: int,
    ) -> None:
        self.outcome = outcome
        self.retry_eligible_seen: bool | None = None
        self.retries: list[float] = []
        self.monkeypatch = monkeypatch

        async def fake_run(
            db: Any,
            generation_run_id: Any,
            *,
            arq_pool: object | None = None,
            retry_eligible: bool = False,
        ) -> None:
            self.retry_eligible_seen = retry_eligible
            if outcome is not None:
                raise outcome

        import abridgeai.features.interviews.workers.generation as gen_worker

        monkeypatch.setattr(gen_worker, "GENERATION_MAX_TRIES", 3, raising=False)

        class _Ctx(dict):
            def get(self, key: str, default: Any = None) -> Any:
                if key == "job_try":
                    return job_try
                if key == "redis":
                    return None
                return default

        class _FailingTaskRunner:
            pass

        # Patch the service entrypoint the task calls.
        monkeypatch.setattr(
            "abridgeai.features.interviews.services.generation.run_interview_generation",
            fake_run,
        )
        self.ctx: dict[str, Any] = _Ctx(job_try=job_try)

    async def run_task(self) -> None:
        from types import SimpleNamespace

        from abridgeai.core.db import get_sessionmaker
        from abridgeai.features.interviews.workers.generation import (
            run_interview_generation_task,
        )

        class _NullSession:
            async def __aenter__(self):
                return SimpleNamespace()

            async def __aexit__(self, *args: object) -> None:
                return None

        self.monkeypatch.setattr(
            "abridgeai.features.interviews.workers.generation.get_sessionmaker",
            lambda: (lambda: _NullSession()),
        )
        # Reference to satisfy linters; real patch above.
        _ = get_sessionmaker
        await run_interview_generation_task(
            self.ctx, uuid4(), uuid4()
        )


@pytest.mark.asyncio
async def test_transient_failure_raises_arq_retry_with_backoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from arq import Retry

    harness = _FakePipeline(
        monkeypatch,
        ProviderError("HTTP error calling https://x: ConnectError: boom"),
        job_try=1,
    )
    with pytest.raises(Retry) as exc_info:
        await harness.run_task()
    # Bounded exponential: first retry defers 30s (arq stores milliseconds).
    assert exc_info.value.defer_score == 30_000
    # Budget remained, so the run was handed retry eligibility.
    assert harness.retry_eligible_seen is True


@pytest.mark.asyncio
async def test_permanent_failure_propagates_without_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _FakePipeline(
        monkeypatch,
        ResponseFormatError("non-JSON content"),
        job_try=1,
    )
    with pytest.raises(ResponseFormatError):
        await harness.run_task()
    assert harness.retry_eligible_seen is True  # offered, correctly declined


@pytest.mark.asyncio
async def test_final_attempt_does_not_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _FakePipeline(
        monkeypatch,
        ProviderError("/chat/completions returned HTTP 503: busy"),
        job_try=3,
    )
    with pytest.raises(ProviderError):
        await harness.run_task()
    # Budget exhausted: the pipeline was told it is NOT retry eligible, so a
    # terminal 'failed' stamp (not a pending requeue) is expected there.
    assert harness.retry_eligible_seen is False
