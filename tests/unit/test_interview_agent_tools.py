"""Contract tests for the interview agent TOOL surface.

These pin the guarantees that must survive when an LLM — not
``decide_next_action`` — drives turn-taking. The model owns the words; the server
owns what may be asked, whether the interview may end, and what counts as
covered. Every guarantee below is therefore enforced at a tool boundary the model
cannot route around.

Pinned here:
  * the outcome checklist is DERIVED from ``coverage_points`` (no second
    representation that could diverge from ``coverage.is_provisionally_sufficient``)
  * ``end_interview`` is refused while a required outcome is unticked AND
    questions remain AND time remains — all three, because any two of them alone
    can be satisfied in a state where refusing forever is possible
  * the refusal names the specific unmet outcomes (a generic "keep going" invites
    the model to retry the identical call)
  * the refusal counter stops refusing after a bound, so a stubborn model cannot
    deadlock the session
  * ``request_hint`` enforces ``MAX_CANNOT_ANSWER_HINTS`` server-side and reports
    the final rung, so the ladder cannot be extended by asking again
"""

from __future__ import annotations

from typing import Any

import pytest

from abridgeai.features.interviews.orchestrator.coverage import COVERAGE_SUFFICIENT_POINTS
from abridgeai.features.interviews.orchestrator.state import (
    InterviewRuntimeStateData,
    OutcomeCoverageState,
)
from abridgeai.features.interviews.orchestrator.tools import (
    MAX_END_REFUSALS,
    ProgressReport,
    build_progress_report,
    resolve_end_interview,
    resolve_hint_request,
)


def _state(**kw: object) -> InterviewRuntimeStateData:
    data = InterviewRuntimeStateData()
    for key, value in kw.items():
        setattr(data, key, value)
    return data


def _covered(outcome_id: str, points: int) -> OutcomeCoverageState:
    return OutcomeCoverageState(outcome_id=outcome_id, coverage_points=points)


# ── checklist is derived, never stored twice ──────────────────────────────────


def test_tick_is_derived_from_coverage_points() -> None:
    data = _state(
        outcome_coverage={
            "o_done": _covered("o_done", COVERAGE_SUFFICIENT_POINTS),
            "o_partial": _covered("o_partial", COVERAGE_SUFFICIENT_POINTS - 1),
            "o_none": _covered("o_none", 0),
        }
    )
    report = build_progress_report(
        data, required_outcome_ids=["o_done", "o_partial", "o_none"], questions_remaining=2
    )
    ticked = {o.outcome_id: o.ticked for o in report.outcomes}
    assert ticked == {"o_done": True, "o_partial": False, "o_none": False}
    assert report.required_unticked == ["o_partial", "o_none"]


def test_outcome_with_no_coverage_row_is_untickable() -> None:
    # An outcome the candidate has never touched has no row yet; it must count as
    # unticked rather than raising or silently disappearing from the checklist.
    report = build_progress_report(
        _state(), required_outcome_ids=["o_never_seen"], questions_remaining=1
    )
    assert report.required_unticked == ["o_never_seen"]
    assert report.outcomes[0].ticked is False


# ── end_interview gating ──────────────────────────────────────────────────────


def test_end_refused_while_required_outcome_unticked() -> None:
    verdict = resolve_end_interview(
        _state(outcome_coverage={"o1": _covered("o1", 0)}),
        required_outcome_ids=["o1"],
        outcome_titles={"o1": "Explain index selection"},
        questions_remaining=3,
        below_closing_threshold=False,
    )
    assert verdict.allowed is False
    # The refusal must be actionable: name the outcome, not just "keep going".
    assert "Explain index selection" in verdict.message
    assert verdict.refusal_count == 1


def test_end_allowed_once_every_required_outcome_is_ticked() -> None:
    verdict = resolve_end_interview(
        _state(outcome_coverage={"o1": _covered("o1", COVERAGE_SUFFICIENT_POINTS)}),
        required_outcome_ids=["o1"],
        outcome_titles={"o1": "Explain index selection"},
        questions_remaining=3,
        below_closing_threshold=False,
    )
    assert verdict.allowed is True


@pytest.mark.parametrize(
    ("questions_remaining", "below_closing_threshold", "why"),
    [
        (0, False, "no question left to ask, so coverage can never be completed"),
        (3, True, "past the closing threshold — time must win over coverage"),
    ],
)
def test_end_allowed_when_refusing_would_deadlock(
    questions_remaining: int, below_closing_threshold: bool, why: str
) -> None:
    verdict = resolve_end_interview(
        _state(outcome_coverage={"o1": _covered("o1", 0)}),
        required_outcome_ids=["o1"],
        outcome_titles={"o1": "Explain index selection"},
        questions_remaining=questions_remaining,
        below_closing_threshold=below_closing_threshold,
    )
    assert verdict.allowed is True, why


def test_refusal_counter_stops_refusing_at_the_bound() -> None:
    # Anti-deadlock layer 2: a model that keeps trying to end must eventually be
    # let through rather than ping-ponged forever.
    data = _state(outcome_coverage={"o1": _covered("o1", 0)})
    for expected in range(1, MAX_END_REFUSALS + 1):
        verdict = resolve_end_interview(
            data,
            required_outcome_ids=["o1"],
            outcome_titles={"o1": "Explain index selection"},
            questions_remaining=3,
            below_closing_threshold=False,
        )
        assert verdict.allowed is False
        assert verdict.refusal_count == expected

    final = resolve_end_interview(
        data,
        required_outcome_ids=["o1"],
        outcome_titles={"o1": "Explain index selection"},
        questions_remaining=3,
        below_closing_threshold=False,
    )
    assert final.allowed is True, "refused past the bound — the session can deadlock"


# ── request_hint ladder is server-owned ──────────────────────────────────────


def test_hint_ladder_escalates_then_reports_final() -> None:
    from abridgeai.features.interviews.orchestrator.decision import MAX_CANNOT_ANSWER_HINTS

    data = _state()
    levels = []
    for _ in range(MAX_CANNOT_ANSWER_HINTS):
        result = resolve_hint_request(data)
        assert result.granted is True
        levels.append(result.level)
    assert levels == sorted(set(levels)), "ladder did not escalate"

    exhausted = resolve_hint_request(data)
    assert exhausted.granted is False
    assert exhausted.is_final is True, "model not told the ladder is spent"


def test_progress_report_is_json_safe() -> None:
    report = build_progress_report(
        _state(outcome_coverage={"o1": _covered("o1", 2)}),
        required_outcome_ids=["o1"],
        outcome_titles={"o1": "Explain index selection"},
        questions_remaining=1,
    )
    assert isinstance(report, ProgressReport)
    payload = report.to_dict()
    import json

    json.dumps(payload)  # must not raise — this crosses the tool boundary
    assert payload["required_unticked"] == []


# ── next_question gating: finish the current question before moving on ────────


def _adv(**kw: object) -> Any:
    from abridgeai.features.interviews.orchestrator.tools import resolve_next_question

    base: dict[str, Any] = {
        "current_outcome_id": "o1",
        "questions_remaining": 3,
        "below_closing_threshold": False,
        "max_follow_ups_per_question": 2,
    }
    base.update(kw)
    data = base.pop("data", None) or _state(
        outcome_coverage={"o1": _covered("o1", 0)},
    )
    return resolve_next_question(data, **base)


def test_next_question_refused_while_current_question_unresolved() -> None:
    # The candidate has not answered sufficiently and the budgets are untouched:
    # the agent must probe or hint, not skip ahead.
    verdict = _adv()
    assert verdict.allowed is False
    assert "probe" in verdict.message.lower() or "hint" in verdict.message.lower()


def test_next_question_allowed_once_current_outcome_is_ticked() -> None:
    verdict = _adv(data=_state(outcome_coverage={"o1": _covered("o1", COVERAGE_SUFFICIENT_POINTS)}))
    assert verdict.allowed is True


def test_next_question_allowed_when_hint_ladder_is_spent() -> None:
    # The escape hatch that prevents a deadlock on a candidate who genuinely
    # cannot answer: once the ladder is exhausted the interview MUST move on.
    from abridgeai.features.interviews.orchestrator.decision import MAX_CANNOT_ANSWER_HINTS

    data = _state(outcome_coverage={"o1": _covered("o1", 0)}, hint_level=MAX_CANNOT_ANSWER_HINTS)
    assert _adv(data=data).allowed is True


def test_next_question_allowed_when_followup_budget_is_spent() -> None:
    data = _state(outcome_coverage={"o1": _covered("o1", 0)}, current_question_follow_up_count=2)
    assert _adv(data=data).allowed is True


def test_next_question_allowed_for_the_first_question() -> None:
    # No current question yet — nothing to resolve.
    assert _adv(current_outcome_id=None).allowed is True


def test_next_question_allowed_past_the_closing_threshold() -> None:
    assert _adv(below_closing_threshold=True).allowed is True


def test_next_question_refusals_are_bounded() -> None:
    from abridgeai.features.interviews.orchestrator.tools import MAX_ADVANCE_REFUSALS

    data = _state(outcome_coverage={"o1": _covered("o1", 0)})
    for _ in range(MAX_ADVANCE_REFUSALS):
        assert _adv(data=data).allowed is False
    assert _adv(data=data).allowed is True, "unbounded refusal can trap the interview"


# ── auto-appended state reminder (no tool round-trip on the common turn) ──────


def test_reminder_tells_the_agent_to_probe_when_outcome_uncovered() -> None:
    from abridgeai.features.interviews.orchestrator.tools import build_turn_reminder

    note = build_turn_reminder(
        _state(outcome_coverage={"o1": _covered("o1", 0)}),
        current_outcome_id="o1",
        outcome_titles={"o1": "Explain index selection"},
        required_outcome_ids=["o1"],
        questions_remaining=3,
        max_follow_ups_per_question=2,
        below_closing_threshold=False,
    )
    lowered = note.lower()
    assert "not" in lowered
    assert "covered" in lowered
    # It must name the concrete next move, not just describe state.
    assert "next_question" in lowered
    # And it must be explicit that advancing is blocked right now.
    assert "do not" in lowered or "cannot" in lowered


def test_reminder_releases_the_agent_once_outcome_is_covered() -> None:
    from abridgeai.features.interviews.orchestrator.tools import build_turn_reminder

    note = build_turn_reminder(
        _state(outcome_coverage={"o1": _covered("o1", COVERAGE_SUFFICIENT_POINTS)}),
        current_outcome_id="o1",
        outcome_titles={"o1": "Explain index selection"},
        required_outcome_ids=["o1"],
        questions_remaining=3,
        max_follow_ups_per_question=2,
        below_closing_threshold=False,
    )
    assert "next_question" in note.lower()
    assert "do not call next_question" not in note.lower()


def test_reminder_reports_the_hint_ladder_is_spent() -> None:
    from abridgeai.features.interviews.orchestrator.decision import MAX_CANNOT_ANSWER_HINTS
    from abridgeai.features.interviews.orchestrator.tools import build_turn_reminder

    note = build_turn_reminder(
        _state(outcome_coverage={"o1": _covered("o1", 0)}, hint_level=MAX_CANNOT_ANSWER_HINTS),
        current_outcome_id="o1",
        outcome_titles={"o1": "Explain index selection"},
        required_outcome_ids=["o1"],
        questions_remaining=3,
        max_follow_ups_per_question=2,
        below_closing_threshold=False,
    )
    lowered = note.lower()
    assert "hint" in lowered
    # Ladder spent → advancing is the required move, not blocked.
    assert "do not call next_question" not in lowered


def test_reminder_never_leaks_answer_content() -> None:
    # The reminder is injected into the LLM context every turn. It must carry
    # PROGRESS only — never rubric, expected answers or evidence text.
    from abridgeai.features.interviews.orchestrator.tools import build_turn_reminder

    note = build_turn_reminder(
        _state(outcome_coverage={"o1": _covered("o1", 1)}),
        current_outcome_id="o1",
        outcome_titles={"o1": "Explain index selection"},
        required_outcome_ids=["o1"],
        questions_remaining=2,
        max_follow_ups_per_question=2,
        below_closing_threshold=False,
    )
    lowered = note.lower()
    for banned in ("the answer is", "correct answer", "rubric", "expected evidence"):
        assert banned not in lowered


def test_reminder_is_compact() -> None:
    # It rides in the context on EVERY turn, so it must stay short or it crowds
    # out the conversation it is supposed to support.
    from abridgeai.features.interviews.orchestrator.tools import build_turn_reminder

    note = build_turn_reminder(
        _state(outcome_coverage={f"o{i}": _covered(f"o{i}", 0) for i in range(8)}),
        current_outcome_id="o0",
        outcome_titles={f"o{i}": f"Outcome number {i}" for i in range(8)},
        required_outcome_ids=[f"o{i}" for i in range(8)],
        questions_remaining=5,
        max_follow_ups_per_question=2,
        below_closing_threshold=False,
    )
    assert len(note) < 600, f"reminder too long ({len(note)} chars)"


# ── the advance gate must apply per QUESTION, not once per session ─────────────


def test_advance_refusals_reset_when_the_question_changes() -> None:
    """Otherwise the gate opens permanently after the first two refusals.

    A session-level counter means question one gets probed and every question
    after it can be rushed — the coverage pressure this gate exists to create
    would apply to the first question only.
    """
    from abridgeai.features.interviews.orchestrator.tools import (
        MAX_ADVANCE_REFUSALS,
        reset_for_new_question,
    )

    data = _state(outcome_coverage={"o1": _covered("o1", 0)})
    for _ in range(MAX_ADVANCE_REFUSALS):
        assert _adv(data=data).allowed is False
    assert _adv(data=data).allowed is True  # escape hatch fires

    # Now the interview moves on. The next question must start with full pressure.
    reset_for_new_question(data)
    assert data.advance_refusal_count == 0
    assert _adv(data=data).allowed is False, "the gate stayed open on a new question"


def test_reset_for_new_question_clears_the_per_question_counters() -> None:
    # These are the counters that make a question's scaffolding budget its own.
    from abridgeai.features.interviews.orchestrator.tools import reset_for_new_question

    data = _state()
    data.hint_level = 2
    data.reframe_count = 1
    data.current_question_follow_up_count = 2
    data.current_question_hint_refunds = 1
    data.advance_refusal_count = 2
    data.end_refusal_count = 1

    reset_for_new_question(data)

    assert data.hint_level == 0
    assert data.reframe_count == 0
    assert data.current_question_follow_up_count == 0
    assert data.current_question_hint_refunds == 0
    assert data.advance_refusal_count == 0
    # Ending is a SESSION decision, so its budget must NOT reset per question —
    # otherwise a model that wants to quit early gets a fresh argument every time.
    assert data.end_refusal_count == 1


# ── the reminder must carry the clock (frontend R1) ───────────────────────────


def test_reminder_reports_time_remaining_when_the_session_is_timed() -> None:
    """A timed session whose clock never reaches the client cannot auto-close.

    The client's `reconcileDeadline` returns early on a null value and
    `useInterviewTimeout` then bails, both silently by design — so a missing clock
    is indistinguishable from an untimed session. The candidate keeps answering
    past the limit with nothing on screen to warn them.
    """
    from abridgeai.features.interviews.orchestrator.tools import build_turn_reminder

    note = build_turn_reminder(
        _state(outcome_coverage={"o1": _covered("o1", 0)}),
        current_outcome_id="o1",
        outcome_titles={"o1": "Explain index selection"},
        required_outcome_ids=["o1"],
        questions_remaining=3,
        max_follow_ups_per_question=2,
        below_closing_threshold=False,
        time_remaining_seconds=420,
    )
    assert "7 minutes" in note or "7 min" in note, f"clock missing from note: {note}"


def test_reminder_omits_the_clock_for_an_untimed_session() -> None:
    # An untimed session must not be told it has 0 minutes left — that would push
    # the agent to rush a session that has no limit at all.
    from abridgeai.features.interviews.orchestrator.tools import build_turn_reminder

    note = build_turn_reminder(
        _state(outcome_coverage={"o1": _covered("o1", 0)}),
        current_outcome_id="o1",
        required_outcome_ids=["o1"],
        questions_remaining=3,
        max_follow_ups_per_question=2,
        below_closing_threshold=False,
        time_remaining_seconds=None,
    )
    assert "minute" not in note.lower()


def test_reminder_warns_when_time_is_nearly_up() -> None:
    from abridgeai.features.interviews.orchestrator.tools import build_turn_reminder

    note = build_turn_reminder(
        _state(outcome_coverage={"o1": _covered("o1", 0)}),
        current_outcome_id="o1",
        required_outcome_ids=["o1"],
        questions_remaining=3,
        max_follow_ups_per_question=2,
        below_closing_threshold=True,
        time_remaining_seconds=60,
    )
    lowered = note.lower()
    assert "wrap up" in lowered or "closing" in lowered, f"no wrap-up signal: {note}"


# ── teacher-configured budgets (Phase 16) ─────────────────────────────────────


def test_hint_ladder_respects_a_configured_cap() -> None:
    from abridgeai.features.interviews.orchestrator.tools import resolve_hint_request

    data = _state()
    # Cap of 1: the second request must refuse even though the shipped
    # constant is 3 — the teacher's setting wins.
    first = resolve_hint_request(data, max_hints=1)
    assert first.granted is True
    second = resolve_hint_request(data, max_hints=1)
    assert second.granted is False
    assert second.is_final is True


def test_current_question_resolves_at_a_configured_hint_cap() -> None:
    from abridgeai.features.interviews.orchestrator.tools import current_question_resolved

    data = _state(outcome_coverage={"o1": _covered("o1", 0)}, hint_level=1)
    assert current_question_resolved(
        data, current_outcome_id="o1", max_follow_ups_per_question=2, max_hints=1
    ), "a hint cap of 1 must resolve the question after one hint"


def test_followup_budget_cap_zero_resolves_immediately() -> None:
    from abridgeai.features.interviews.orchestrator.tools import current_question_resolved

    data = _state(outcome_coverage={"o1": _covered("o1", 0)})
    assert current_question_resolved(
        data, current_outcome_id="o1", max_follow_ups_per_question=0, max_hints=3
    ), "a follow-up cap of 0 must resolve the question on the first unanswered turn"
