"""Canonical legacy↔adaptive response mapping (Slice 4, safeguard #4).

The single source of truth that turns an authoritative :class:`InterviewerDecision`
(plus the selected question + generated utterance) into ONE result dict carrying
BOTH the legacy fields (``next_question`` / ``is_finished`` / ``followup_text``)
and the new structured fields (``action`` / ``reason_code`` / ``ai_turn_text`` /
``should_await_response`` / ``should_finish`` / …).

Why one function (safeguard #4): building the legacy and adaptive payloads
independently risks them contradicting each other (e.g. legacy says "advance"
while the structured action says "probe"). Deriving both from this mapper makes
that impossible.

Anti-double-render rule (safeguard #5): on an ADVANCE action, the legacy
``followup_text`` carries ONLY the acknowledgement + transition (never the
question text) while ``next_question`` carries the question — so the current
frontend, which renders both, shows "Thanks, next question." then the question,
with no duplication. On a NON-advance action (probe/clarify/repeat/…), the whole
utterance goes in ``followup_text`` and ``next_question`` is None. The new client
may instead render the combined ``ai_turn_text`` directly.

Pure, no I/O — trivially unit-testable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from abridgeai.features.interviews.orchestrator.decision import (
    InterviewerActionType,
)

if TYPE_CHECKING:
    from abridgeai.features.interviews.orchestrator.decision import InterviewerDecision
    from abridgeai.features.interviews.orchestrator.utterance import Utterance


# Actions that ADVANCE to a freshly-selected question (the question rides in the
# legacy ``next_question`` field; ack+transition ride in ``followup_text``).
_ADVANCE_ACTIONS: frozenset[InterviewerActionType] = frozenset(
    {
        InterviewerActionType.ASK_MAIN_QUESTION,
        InterviewerActionType.TRANSITION_TOPIC,
        InterviewerActionType.SKIP_QUESTION,
    }
)

# Actions that END the interview (legacy ``is_finished=True``). The closing
# utterance still rides in ``followup_text`` / ``ai_turn_text`` so the final
# message renders + narrates BEFORE the client transitions to evaluation
# (safeguard #6).
_CLOSING_ACTIONS: frozenset[InterviewerActionType] = frozenset(
    {
        InterviewerActionType.BEGIN_CLOSING,
        InterviewerActionType.CLOSE_INTERVIEW,
    }
)


def canonical_step_result(
    *,
    decision: InterviewerDecision,
    utterance: Utterance,
    selected_question: Any | None,  # noqa: ANN401 -- ORM InterviewQuestion, kept loose to avoid import cycle
    language: str,
    state_version: int,
    ai_turn_id: str | None,
    utterance_status: str,
) -> dict[str, Any]:
    """Produce the canonical superset result for one adaptive turn.

    ``selected_question`` is the ORM question row for an advance action, else
    None. ``ai_turn_id`` is the id of the persisted AI message (None only when
    no AI turn was written). ``utterance_status`` is ``"llm"`` / ``"fallback"``.

    The returned dict is a SUPERSET of the legacy contract: the router maps it
    straight onto :class:`InterviewSubmitAnswerResponse` (legacy fields first,
    then the optional structured fields).
    """
    is_closing = decision.action in _CLOSING_ACTIONS
    is_advance = decision.action in _ADVANCE_ACTIONS and selected_question is not None

    # Legacy followup_text:
    #  - advance      → ack + transition ONLY (question comes via next_question)
    #  - non-advance  → the full utterance (probe / clarify / repeat / closing)
    if is_advance:
        followup_text = utterance.acknowledgement_and_transition() or None
    else:
        followup_text = utterance.ai_turn_text or None

    next_question = selected_question if is_advance else None
    is_finished = is_closing

    # should_await_response: the client should collect another answer UNLESS we
    # are finishing. should_finish mirrors is_finished for the new client.
    should_await = not is_closing
    should_finish = is_closing

    target_outcome_id = decision.target_outcome_id
    current_question_id = str(getattr(selected_question, "id", "")) or None if is_advance else None

    return {
        # ── legacy fields (authoritative for existing clients) ───────────────
        "next_question": next_question,
        "is_finished": is_finished,
        "followup_text": followup_text,
        # ── structured fields (new clients; all derived from the SAME decision)
        "action": decision.action.value,
        "reason_code": decision.reason_code.value,
        "ai_turn_text": utterance.ai_turn_text or None,
        "language": language,
        "should_narrate": bool(utterance.ai_turn_text),
        "current_question_id": current_question_id,
        "target_outcome_id": target_outcome_id,
        "should_await_response": should_await,
        "should_finish": should_finish,
        "state_version": state_version,
        "ai_turn_id": ai_turn_id,
        # ── internal (NOT projected to the student API; audit/logs only) ─────
        "_utterance_status": utterance_status,
        "_should_record_evidence": decision.should_record_academic_evidence,
    }


__all__ = ["canonical_step_result"]
