"""A recovered evaluation must stop looking like a failure.

``status='failed'`` means one thing: the grader never finished. It is stamped by
the ARQ wrapper once the retry budget is gone, so the student-facing poll can
stop waiting instead of spinning on a ``pass_verdict`` that will never arrive.

That makes it a *provisional* state, not a judgement about the student — and the
recovery sweep can now pick those rows up and grade them. So the moment a verdict
lands, the failure state has to go: leaving ``status='failed'`` on a graded
session would show an error for an interview that is in fact finished, and would
hide the row from any reader that filters on terminal status.

The stale ``evaluation_failure`` note goes with it, for the same reason — it
describes an attempt that has since been superseded.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from abridgeai.features.interviews.ai.stages.evaluation.rubric import RubricScores
from abridgeai.features.interviews.services.evaluation import _stamp_session_summary


class _Verdicts(SimpleNamespace):
    """Minimal stand-in for OutcomeVerdicts."""


def _verdicts(*, total: int, met: int) -> Any:
    return _Verdicts(total=total, met_count=met, items=[])


def _rubric() -> Any:
    return RubricScores(response_evaluations=[], aggregated={"clarity": 4.0}, total_score=80.0)


def _session(*, status: str, summary: dict[str, Any] | None = None) -> Any:
    return SimpleNamespace(
        status=status,
        pass_verdict=None,
        internal_summary_json=summary if summary is not None else {},
    )


def _stamp(session: Any, *, total: int = 2, met: int = 2) -> None:
    _stamp_session_summary(
        session,
        rubric_scores=_rubric(),
        verdicts=_verdicts(total=total, met=met),
        min_outcomes_to_pass=None,
        question_count=2,
        answered_question_count=2,
    )


def test_a_recovered_failure_becomes_completed() -> None:
    session = _session(
        status="failed",
        summary={"evaluation_failure": {"message": "LLM timeout", "final_attempt": True}},
    )

    _stamp(session)

    assert session.status == "completed", (
        "a session that reached a verdict must not still read as a grader failure"
    )
    assert session.pass_verdict is True
    assert "evaluation_failure" not in session.internal_summary_json, (
        "the stale failure note describes a superseded attempt"
    )


def test_a_recovered_failure_that_does_not_pass_is_still_cleared() -> None:
    """Clearing the failure state is about the GRADER, not the outcome."""
    session = _session(
        status="failed", summary={"evaluation_failure": {"message": "boom"}}
    )

    _stamp(session, total=3, met=1)

    assert session.status == "completed"
    assert session.pass_verdict is False
    assert "evaluation_failure" not in session.internal_summary_json


def test_a_normal_completed_session_is_untouched() -> None:
    session = _session(status="completed")

    _stamp(session)

    assert session.status == "completed"
    assert session.pass_verdict is True


def test_a_timed_out_session_keeps_its_status() -> None:
    """``timed_out`` is a real assessment outcome and must survive grading.

    Only ``failed`` is provisional. Collapsing ``timed_out`` into ``completed``
    would erase the fact that the student ran out of time.
    """
    session = _session(status="timed_out")

    _stamp(session)

    assert session.status == "timed_out"
    assert session.pass_verdict is True


def test_recovery_bookkeeping_survives_a_successful_grade() -> None:
    """The attempt counter stays: it is the audit trail for the re-drive."""
    session = _session(
        status="failed",
        summary={
            "evaluation_failure": {"message": "boom"},
            "evaluation_recovery": {"attempts": 2, "last_attempt_at": "2026-09-04T00:00:00+00:00"},
        },
    )

    _stamp(session)

    assert session.internal_summary_json["evaluation_recovery"]["attempts"] == 2
    assert "evaluation_failure" not in session.internal_summary_json
