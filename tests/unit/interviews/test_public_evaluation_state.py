"""The student-facing session DTO must say whether grading is still coming.

The frontend had to guess. Its predicate was ``status in (completed, timed_out)
and pass_verdict is null``, which is wrong in both directions:

* ``status='failed'`` was treated as final, so a session the recovery sweep is
  about to re-drive showed a permanent error badge and stopped polling. The
  verdict landed seconds later and the UI never noticed.
* nothing told the UI when the recovery budget was actually gone, so the only
  alternative — polling every ``failed`` row — would poll forever.

So the backend derives it, from state it already holds: the terminal status, the
verdict, and the recovery attempt counter. ``internal_summary_json`` itself stays
teacher-only; only the derived label crosses the wire.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

from abridgeai.features.interviews.services.evaluation_state import (
    MAX_EVALUATION_RECOVERY_ATTEMPTS,
    derive_evaluation_state,
)

_LONG_AGO = "2020-01-01T00:00:00+00:00"


def _session(
    *,
    status: str,
    pass_verdict: bool | None = None,
    attempts: int | None = None,
    assessment_started_at: Any = "2026-09-05T00:00:00+00:00",
    last_attempt_at: Any = _LONG_AGO,
    claim_expires_at: Any = None,
) -> Any:
    """A session row stand-in.

    ``last_attempt_at`` defaults to long ago so a test that only sets ``attempts``
    describes a SETTLED final attempt — the interesting "at the ceiling" case
    where no job is running any more.
    """
    summary: dict[str, Any] = {}
    if attempts is not None:
        summary["evaluation_recovery"] = {"attempts": attempts, "last_attempt_at": last_attempt_at}
    return SimpleNamespace(
        status=status,
        pass_verdict=pass_verdict,
        assessment_started_at=assessment_started_at,
        internal_summary_json=summary,
        evaluation_claim_expires_at=claim_expires_at,
    )


def test_a_graded_session_is_succeeded() -> None:
    assert derive_evaluation_state(_session(status="completed", pass_verdict=True)) == "succeeded"


def test_a_failing_verdict_is_also_succeeded() -> None:
    """The GRADER succeeded. ``pass_verdict=False`` is a published judgement."""
    assert derive_evaluation_state(_session(status="completed", pass_verdict=False)) == "succeeded"


def test_a_terminal_ungraded_session_is_pending() -> None:
    assert derive_evaluation_state(_session(status="completed")) == "pending"
    assert derive_evaluation_state(_session(status="timed_out")) == "pending"


def test_a_grader_failure_with_budget_left_is_still_pending() -> None:
    """``status='failed'`` only means ARQ ran out of retries — recovery re-drives it."""
    assert derive_evaluation_state(_session(status="failed", attempts=0)) == "pending"
    assert (
        derive_evaluation_state(
            _session(status="failed", attempts=MAX_EVALUATION_RECOVERY_ATTEMPTS - 1)
        )
        == "pending"
    )


def test_the_last_attempt_is_still_pending_while_its_job_can_be_running() -> None:
    """Reaching the ceiling means the last job was DISPATCHED, not that it ended.

    The counter is charged before the enqueue (so a hard-killed task still spends
    its budget), so a session hits ``attempts == MAX`` the instant the final job
    is queued. Calling that ``exhausted`` told the student no verdict was coming
    while their last grading run was queued or mid-flight: the UI stopped polling
    and never showed the result that landed a minute later.
    """
    just_dispatched = _session(
        status="failed",
        attempts=MAX_EVALUATION_RECOVERY_ATTEMPTS,
        last_attempt_at=datetime.now(UTC).isoformat(),
    )
    assert derive_evaluation_state(just_dispatched) == "pending"


def test_a_live_claim_keeps_the_last_attempt_pending() -> None:
    """A held lease is direct evidence a grader owns the session right now."""
    working = _session(
        status="failed",
        attempts=MAX_EVALUATION_RECOVERY_ATTEMPTS,
        last_attempt_at=_LONG_AGO,  # dispatch timestamp is stale...
        claim_expires_at=datetime.now(UTC) + timedelta(minutes=5),  # ...but a job holds it
    )
    assert derive_evaluation_state(working) == "pending"


def test_an_expired_claim_with_a_terminal_phase_is_exhausted() -> None:
    """Lapsed lease AND terminal phase record: the last job is proven over."""
    dead = _session(
        status="failed",
        attempts=MAX_EVALUATION_RECOVERY_ATTEMPTS,
        claim_expires_at=datetime.now(UTC) - timedelta(minutes=5),
    )
    dead.internal_summary_json["evaluation_recovery"]["current"] = {
        "attempt": MAX_EVALUATION_RECOVERY_ATTEMPTS,
        "job_id": "j-final",
        "phase": "missing",
        "dispatched_at": _LONG_AGO,
    }
    assert derive_evaluation_state(dead) == "exhausted"


def test_an_expired_claim_alone_does_not_prove_exhausted() -> None:
    """A lapsed lease is evidence the OWNER is gone, not that the job is.

    The job may be queued behind a busy worker, never having claimed. Only a
    terminal phase record (or a lazy reconcile against ARQ) proves that.
    """
    dead = _session(
        status="failed",
        attempts=MAX_EVALUATION_RECOVERY_ATTEMPTS,
        claim_expires_at=datetime.now(UTC) - timedelta(minutes=5),
    )
    assert derive_evaluation_state(dead) == "pending"


def test_the_settle_window_outlasts_a_live_evaluation_job() -> None:
    """A slow-but-healthy job must never be declared dead while it works.

    ARQ kills a task at ``job_timeout``, so any window at least as long as the
    claim lease covers the worst case. A shorter one would flip a running
    evaluation to ``exhausted`` and stop the student's UI mid-grade.
    """
    from abridgeai.features.interviews.services.evaluation_claim import EVALUATION_LEASE_SECONDS
    from abridgeai.features.interviews.services.evaluation_state import (
        FINAL_ATTEMPT_SETTLE_SECONDS,
    )
    from abridgeai.workers.arq_app import WorkerSettings

    assert FINAL_ATTEMPT_SETTLE_SECONDS >= EVALUATION_LEASE_SECONDS
    assert WorkerSettings.job_timeout < FINAL_ATTEMPT_SETTLE_SECONDS


def test_an_exhausted_recovery_budget_is_terminal() -> None:
    """At the ceiling with the last job PROVEN terminal, the UI stops waiting."""
    for phase in ("succeeded", "failed", "missing"):
        session = _session_with_current(
            "failed",
            {
                "attempt": MAX_EVALUATION_RECOVERY_ATTEMPTS,
                "job_id": "j-final",
                "phase": phase,
                "dispatched_at": _LONG_AGO,
            },
        )
        assert derive_evaluation_state(session) == "exhausted", phase


def test_an_exhausted_session_that_later_grades_reads_as_succeeded() -> None:
    """A verdict outranks the counter — the attempts stay as an audit trail."""
    session = _session(
        status="completed",
        pass_verdict=True,
        attempts=MAX_EVALUATION_RECOVERY_ATTEMPTS,
    )
    assert derive_evaluation_state(session) == "succeeded"


def test_a_live_session_is_not_required_yet() -> None:
    assert derive_evaluation_state(_session(status="in_progress")) == "not_required"


def test_an_abandoned_session_is_never_graded() -> None:
    assert derive_evaluation_state(_session(status="abandoned")) == "not_required"


def test_a_session_that_never_reached_the_assessment_is_not_graded() -> None:
    """Same refusal as ``_ungradeable_reason``: onboarding-only runs are not judged."""
    session = _session(status="completed", assessment_started_at=None)
    assert derive_evaluation_state(session) == "not_required"


def test_a_missing_summary_is_treated_as_zero_attempts() -> None:
    session = _session(status="failed")
    session.internal_summary_json = None
    assert derive_evaluation_state(session) == "pending"


def test_the_schema_literal_matches_the_service_vocabulary() -> None:
    """The DTO literal is declared separately (schemas must not import services).

    Two copies drift silently: a state added here but not there serializes as a
    validation error at response time, in production, for one unlucky student.
    """
    from typing import get_args

    from abridgeai.features.interviews.schemas.session import EvaluationStateLiteral
    from abridgeai.features.interviews.services.evaluation_state import EvaluationState

    assert set(get_args(EvaluationStateLiteral)) == set(get_args(EvaluationState))


def test_the_recovery_ceiling_matches_the_sweep_default() -> None:
    """The label is only honest if it uses the sweep's real ceiling."""
    import inspect

    from abridgeai.features.interviews.services.lifecycle import recover_stalled_evaluations

    default = (
        inspect.signature(recover_stalled_evaluations).parameters["max_recovery_attempts"].default
    )
    assert default == MAX_EVALUATION_RECOVERY_ATTEMPTS


# ─────────────── the durable phase decides, not a timestamp heuristic ───────────────
#
# The dispatch age (``last_attempt_at``) can only say "a job was sent". It
# cannot distinguish a job that is queued/running/retrying from one that
# finished (and its result later dropped out of Redis) — so a queued job >30
# minutes old was reported ``exhausted`` to the student while the worker was
# still about to pick it up. The durable ``evaluation_recovery.current.phase``
# record is the authority: active means pending, terminal means the budget can
# be called spent, and a verdict outranks everything.


def _session_with_current(status: str, current: dict[str, Any] | None) -> Any:
    summary: dict[str, Any] = {
        "evaluation_recovery": {
            "attempts": MAX_EVALUATION_RECOVERY_ATTEMPTS,
            "last_attempt_at": _LONG_AGO,
        }
    }
    if current is not None:
        summary["evaluation_recovery"]["current"] = current
    return SimpleNamespace(
        status=status,
        pass_verdict=None,
        assessment_started_at="2026-09-05T00:00:00+00:00",
        internal_summary_json=summary,
        evaluation_claim_expires_at=None,
    )


def test_an_active_durable_phase_is_pending_no_matter_how_old_the_dispatch_is() -> None:
    """queued/retrying >30min stays pending — the UI must keep polling."""
    for phase in ("dispatching", "queued", "running", "retrying"):
        session = _session_with_current(
            "failed",
            {"attempt": 3, "job_id": "j-1", "phase": phase, "dispatched_at": _LONG_AGO},
        )
        assert derive_evaluation_state(session) == "pending", phase


def test_a_terminal_durable_phase_allows_exhausted() -> None:
    """exhausted needs the last job PROVEN terminal, not merely old."""
    for phase in ("succeeded", "failed", "missing"):
        session = _session_with_current(
            "failed",
            {"attempt": 3, "job_id": "j-1", "phase": phase, "dispatched_at": _LONG_AGO},
        )
        assert derive_evaluation_state(session) == "exhausted", phase


def test_a_currentless_ceiling_row_stays_pending_until_reconciled() -> None:
    """Legacy rows at the ceiling without ``current`` are NOT declared dead.

    The lifecycle lazy-reconciles them against ARQ; until there is terminal
    evidence, ``pending`` keeps the student's UI honest.
    """
    session = _session_with_current("failed", None)
    assert derive_evaluation_state(session) == "pending"
