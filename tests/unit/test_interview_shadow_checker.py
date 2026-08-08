"""Shadow checker: run the deterministic ladder beside the LLM and record both.

The native agent decides turn-taking conversationally, so the ``ReasonCode`` that
used to justify every decision no longer exists — and fabricating one for a turn
the model chose would be worse than having none when a student appeals a grade.

``decide_next_action`` is pure (no DB, no LLM), so running it every turn costs
nothing. What it buys:

  * a ``ReasonCode`` on every turn, so the appeals process keeps working across
    the migration instead of going dark on the native path
  * a live divergence signal — how often the model does something the audited
    policy would not have — measurable BEFORE a student complains
  * evidence, if one does: "the model probed here; the policy would have advanced,
    reason ``FOLLOWUP_LIMIT_REACHED``"

It is explicitly a CHECKER, never a gate: it must not change what the interview
does. A shadow that could block a turn is no longer a shadow, and the whole point
is that it is free to be wrong.
"""

from __future__ import annotations

from abridgeai.features.interviews.orchestrator.decision import (
    DEFAULT_MAX_FOLLOWUPS_PER_QUESTION,
    InterviewerActionType,
    ReasonCode,
)
from abridgeai.features.interviews.orchestrator.intent import IntentClassification, StudentIntent
from abridgeai.features.interviews.orchestrator.shadow import (
    ShadowVerdict,
    shadow_check_turn,
)
from abridgeai.features.interviews.orchestrator.state import (
    InterviewRuntimeStateData,
    OutcomeCoverageState,
)


def _state(**kw: object) -> InterviewRuntimeStateData:
    data = InterviewRuntimeStateData()
    data.outcome_coverage = {"o1": OutcomeCoverageState(outcome_id="o1", coverage_points=0)}
    data.current_outcome_id = "o1"
    for key, value in kw.items():
        setattr(data, key, value)
    return data


def _intent(kind: StudentIntent) -> IntentClassification:
    return IntentClassification(intent=kind, confidence=0.9, rationale="test")


# ── it produces a ReasonCode for every turn ───────────────────────────────────


def test_a_reason_code_is_produced_for_an_ordinary_answer() -> None:
    verdict = shadow_check_turn(
        state=_state(),
        intent=_intent(StudentIntent.ANSWER),
        model_advanced=False,
        questions_remaining=3,
        time_fraction_remaining=0.8,
    )
    assert isinstance(verdict, ShadowVerdict)
    assert isinstance(verdict.reason_code, ReasonCode)
    assert isinstance(verdict.would_have_action, InterviewerActionType)


def test_the_cannot_answer_ladder_is_reflected() -> None:
    # The policy's own recent fix: "I don't know" hints before advancing. The
    # shadow must report that, or a divergence log would blame the model for
    # behaviour the policy also wanted.
    verdict = shadow_check_turn(
        state=_state(hint_level=0),
        intent=_intent(StudentIntent.CANNOT_ANSWER),
        model_advanced=False,
        questions_remaining=3,
        time_fraction_remaining=0.8,
        hint_ladder_enabled=True,
    )
    assert verdict.would_have_action is InterviewerActionType.PROVIDE_NEUTRAL_HINT
    assert verdict.reason_code is ReasonCode.CANNOT_ANSWER_HINT_OFFERED
    assert verdict.diverged is False, "the model also stayed; that is agreement"


# ── divergence is computed on the one axis that matters ───────────────────────


def test_divergence_when_the_model_advances_but_the_policy_would_probe() -> None:
    verdict = shadow_check_turn(
        state=_state(),
        intent=_intent(StudentIntent.CANNOT_ANSWER),
        model_advanced=True,
        questions_remaining=3,
        time_fraction_remaining=0.8,
        hint_ladder_enabled=True,
    )
    assert verdict.diverged is True
    assert verdict.divergence == "model_advanced_policy_stayed"


def test_divergence_when_the_model_stays_but_the_policy_would_advance() -> None:
    # Follow-up budget spent: the policy insists on moving on.
    verdict = shadow_check_turn(
        state=_state(current_question_follow_up_count=DEFAULT_MAX_FOLLOWUPS_PER_QUESTION),
        intent=_intent(StudentIntent.ANSWER),
        model_advanced=False,
        questions_remaining=3,
        time_fraction_remaining=0.8,
    )
    assert verdict.diverged is True
    assert verdict.divergence == "model_stayed_policy_advanced"


def test_agreement_is_reported_as_no_divergence() -> None:
    verdict = shadow_check_turn(
        state=_state(current_question_follow_up_count=DEFAULT_MAX_FOLLOWUPS_PER_QUESTION),
        intent=_intent(StudentIntent.ANSWER),
        model_advanced=True,
        questions_remaining=3,
        time_fraction_remaining=0.8,
    )
    assert verdict.diverged is False
    assert verdict.divergence is None


# ── it is a checker, not a gate ────────────────────────────────────────────────


def test_the_shadow_never_mutates_the_runtime_state() -> None:
    """A shadow that can change state is not a shadow.

    ``decide_next_action`` is pure, but the INPUTS are built from live state and an
    accidental mutation here would silently alter the real interview — the one
    failure mode that would make this dangerous rather than free.
    """
    state = _state(hint_level=1, current_question_follow_up_count=1, advance_refusal_count=1)
    before = state.to_dict()

    shadow_check_turn(
        state=state,
        intent=_intent(StudentIntent.CANNOT_ANSWER),
        model_advanced=True,
        questions_remaining=2,
        time_fraction_remaining=0.4,
        hint_ladder_enabled=True,
    )

    assert state.to_dict() == before, "the shadow checker mutated live state"


def test_the_shadow_never_raises_on_a_malformed_turn() -> None:
    # It runs on every turn of a graded interview. A checker that can throw would
    # take the interview down with it, which is not a trade worth making for audit.
    verdict = shadow_check_turn(
        state=InterviewRuntimeStateData(),  # no coverage rows, no current outcome
        intent=_intent(StudentIntent.ANSWER),
        model_advanced=False,
        questions_remaining=0,
        time_fraction_remaining=None,
    )
    assert isinstance(verdict, ShadowVerdict)


def test_verdict_is_json_safe_for_the_audit_row() -> None:
    import json

    verdict = shadow_check_turn(
        state=_state(),
        intent=_intent(StudentIntent.PARTIAL_ANSWER),
        model_advanced=True,
        questions_remaining=1,
        time_fraction_remaining=0.5,
    )
    json.dumps(verdict.to_dict())  # must not raise — this is persisted
    assert verdict.to_dict()["reason_code"] == verdict.reason_code.value
