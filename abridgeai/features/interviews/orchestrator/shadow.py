"""Shadow checker — the deterministic policy run beside the LLM, for audit.

On the native path the model decides turn-taking, so nothing produces the
``ReasonCode`` that used to justify every decision. Inventing one for a turn the
model chose would be worse than having none when a student appeals a grade, so
instead this runs the real :func:`decision.decide_next_action` on the same inputs
and records what it WOULD have done beside what actually happened.

``decide_next_action`` is pure — no DB, no LLM, no I/O — so this is free to run on
every turn. It yields three things a conversational agent otherwise loses: a
``ReasonCode`` per turn so the appeals process keeps working across the migration,
a live divergence rate measurable before anyone complains, and concrete evidence
if someone does.

Two properties are load-bearing and both are pinned by tests:

* **It never mutates state.** ``decide_next_action`` is pure, but the inputs are
  built from live runtime state; an accidental write here would silently change a
  graded interview. The state is therefore round-tripped into a detached copy
  before it is read.
* **It never raises.** It runs inside every turn of a graded interview. A checker
  that can throw takes the interview with it, which is not a trade worth making
  for an audit record.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from abridgeai.features.interviews.orchestrator.decision import (
    DecisionInputs,
    InterviewerActionType,
    ReasonCode,
    decide_next_action,
)
from abridgeai.features.interviews.orchestrator.state import InterviewRuntimeStateData

if TYPE_CHECKING:
    from abridgeai.features.interviews.orchestrator.intent import IntentClassification

# Actions that move OFF the current question. Divergence is measured on this one
# axis: advance-vs-stay is the decision that changes what the candidate is asked,
# and it is the only one both the model and the policy can be compared on.
_ADVANCE_ACTIONS = frozenset(
    {
        InterviewerActionType.TRANSITION_TOPIC,
        InterviewerActionType.SKIP_QUESTION,
        InterviewerActionType.BEGIN_CLOSING,
        InterviewerActionType.CLOSE_INTERVIEW,
    }
)


@dataclass(frozen=True)
class ShadowVerdict:
    """What the deterministic policy would have done, and whether that differs."""

    would_have_action: InterviewerActionType
    reason_code: ReasonCode
    would_have_advanced: bool
    model_advanced: bool
    diverged: bool
    divergence: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "would_have_action": self.would_have_action.value,
            "reason_code": self.reason_code.value,
            "would_have_advanced": self.would_have_advanced,
            "model_advanced": self.model_advanced,
            "diverged": self.diverged,
            "divergence": self.divergence,
        }


def shadow_check_turn(
    *,
    state: InterviewRuntimeStateData,
    intent: IntentClassification,
    model_advanced: bool,
    questions_remaining: int,
    time_fraction_remaining: float | None,
    hint_ladder_enabled: bool = True,
    required_outcome_ids: Sequence[str] = (),
) -> ShadowVerdict:
    """Run the audited policy on this turn's inputs. Read-only, never raises."""
    try:
        detached = InterviewRuntimeStateData.from_dict(state.to_dict())
        # All REQUIRED outcomes ticked — never whatever happens to be in
        # `outcome_coverage`. That map only gains an entry when a question
        # targeting the outcome is asked, so reading it said "all covered" the
        # moment ONE outcome was graded, which made the audit call a session with
        # five outcomes left to cover "complete" and blare a false
        # `model_stayed_policy_advanced` divergence on every later turn.
        required = tuple(required_outcome_ids) or tuple(detached.outcome_coverage)
        all_covered = bool(required) and all(
            detached.outcome_coverage.get(oid) is not None
            and detached.outcome_coverage[oid].coverage_points >= 2
            for oid in required
        )
        decision = decide_next_action(
            DecisionInputs(
                intent=intent,
                analysis=None,
                current_question_follow_up_count=detached.current_question_follow_up_count,
                total_follow_up_count=detached.total_follow_up_count,
                time_fraction_remaining=time_fraction_remaining,
                has_next_question=questions_remaining > 0,
                all_required_outcomes_covered=all_covered,
                hint_ladder_enabled=hint_ladder_enabled,
                hint_level=detached.hint_level,
            )
        )
        would_advance = decision.action in _ADVANCE_ACTIONS or decision.should_advance_question
        divergence: str | None = None
        if model_advanced and not would_advance:
            divergence = "model_advanced_policy_stayed"
        elif would_advance and not model_advanced:
            divergence = "model_stayed_policy_advanced"
        return ShadowVerdict(
            would_have_action=decision.action,
            reason_code=decision.reason_code,
            would_have_advanced=would_advance,
            model_advanced=model_advanced,
            diverged=divergence is not None,
            divergence=divergence,
        )
    except Exception:  # noqa: BLE001 -- audit must never cost a graded interview
        # A verdict is still returned so the audit row records that the shadow ran
        # and could not decide, rather than leaving a silent hole in the trail.
        return ShadowVerdict(
            would_have_action=InterviewerActionType.ACKNOWLEDGE,
            reason_code=ReasonCode.PARTIAL_OUTCOME_COVERAGE,
            would_have_advanced=False,
            model_advanced=model_advanced,
            diverged=False,
            divergence=None,
        )


__all__ = ["ShadowVerdict", "shadow_check_turn"]
