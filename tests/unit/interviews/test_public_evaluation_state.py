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

from types import SimpleNamespace
from typing import Any

from abridgeai.features.interviews.services.evaluation_state import (
    MAX_EVALUATION_RECOVERY_ATTEMPTS,
    derive_evaluation_state,
)


def _session(
    *,
    status: str,
    pass_verdict: bool | None = None,
    attempts: int | None = None,
    assessment_started_at: Any = "2026-09-05T00:00:00+00:00",
) -> Any:
    summary: dict[str, Any] = {}
    if attempts is not None:
        summary["evaluation_recovery"] = {"attempts": attempts}
    return SimpleNamespace(
        status=status,
        pass_verdict=pass_verdict,
        assessment_started_at=assessment_started_at,
        internal_summary_json=summary,
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


def test_an_exhausted_recovery_budget_is_terminal() -> None:
    """At the ceiling the sweep drops the row, so the UI must stop waiting."""
    assert (
        derive_evaluation_state(
            _session(status="failed", attempts=MAX_EVALUATION_RECOVERY_ATTEMPTS)
        )
        == "exhausted"
    )
    assert (
        derive_evaluation_state(
            _session(status="completed", attempts=MAX_EVALUATION_RECOVERY_ATTEMPTS + 2)
        )
        == "exhausted"
    )


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
