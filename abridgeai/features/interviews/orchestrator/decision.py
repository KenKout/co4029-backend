"""Interviewer action / decision model (Phase 4).

Replaces the binary "follow-up or next question" branch with a structured
:class:`InterviewerDecision`: WHAT action to take, WHY (reason code), whether to
record academic evidence, whether to advance the question, and whether to end.

Design (mirrors the perception layer's conventions):
* Pure types + a deterministic policy here — NO DB access, NO LLM calls. The
  policy maps perception signals (intent + analysis) + runtime state → a
  decision using explicit, documented rules. This keeps the *state machine*
  deterministic (per the brief's non-goal "do not depend entirely on an LLM for
  deterministic state transitions"); natural-language phrasing of the utterance
  (Phase 9) is layered on later and never decides control flow.
* The utterance text on the decision is a safe deterministic placeholder here;
  Slice 4 / Phase 9 replaces it with an LLM-generated natural utterance while
  keeping this action/reason skeleton as the source of truth.

Only the fields the transport layer needs for rendering / narration / state sync
are meant to cross the API boundary; ``internal_rationale`` stays server-side.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from abridgeai.features.interviews.orchestrator.analysis import (
    AnswerAnalysis,
    Correctness,
    ProbeType,
    Relevance,
)
from abridgeai.features.interviews.orchestrator.coverage import is_strong_answer
from abridgeai.features.interviews.orchestrator.intent import (
    IntentClassification,
    StudentIntent,
)
from abridgeai.features.interviews.orchestrator.state import InterviewPhase


class InterviewerActionType(str, Enum):  # noqa: UP042 -- match codebase convention
    OPENING = "opening"
    ACKNOWLEDGE = "acknowledge"
    REPEAT_QUESTION = "repeat_question"
    REFRAME_QUESTION = "reframe_question"
    CLARIFY_WITHOUT_REVEALING_ANSWER = "clarify_without_revealing_answer"
    PROVIDE_NEUTRAL_HINT = "provide_neutral_hint"
    ASK_MAIN_QUESTION = "ask_main_question"
    PROBE_DEEPER = "probe_deeper"
    ASK_FOR_EXAMPLE = "ask_for_example"
    CHALLENGE_REASONING = "challenge_reasoning"
    EXPLORE_TRADEOFF = "explore_tradeoff"
    # Depth probes on a strong answer (Slice 8, v2) — dig for the ceiling.
    EXTEND_ANSWER = "extend_answer"
    PROBE_EDGE_CASE = "probe_edge_case"
    RESOLVE_CONTRADICTION = "resolve_contradiction"
    REDIRECT_TO_TOPIC = "redirect_to_topic"
    TRANSITION_TOPIC = "transition_topic"
    SKIP_QUESTION = "skip_question"
    OFFER_BRIEF_PAUSE = "offer_brief_pause"
    HANDLE_TECHNICAL_ISSUE = "handle_technical_issue"
    # End-confirmation gate (Slice 4): a natural-language end request no longer
    # closes immediately — it asks the candidate to confirm. The next turn
    # resolves to CONFIRM_END (→ closing) or CANCEL_END (→ back to the question).
    REQUEST_END_CONFIRMATION = "request_end_confirmation"
    CANCEL_END = "cancel_end"
    BEGIN_CLOSING = "begin_closing"
    CLOSE_INTERVIEW = "close_interview"


class AcknowledgementStyle(str, Enum):  # noqa: UP042 -- match codebase convention
    NONE = "none"
    NEUTRAL = "neutral"
    POSITIVE = "positive"
    CORRECTIVE = "corrective"


class ReasonCode(str, Enum):  # noqa: UP042 -- match codebase convention
    OPENING_REQUIRED = "opening_required"
    STUDENT_REQUESTED_REPEAT = "student_requested_repeat"
    STUDENT_REQUESTED_CLARIFICATION = "student_requested_clarification"
    STUDENT_REQUESTED_HINT = "student_requested_hint"
    ANSWER_TOO_VAGUE = "answer_too_vague"
    MISSING_EXAMPLE = "missing_example"
    PARTIAL_OUTCOME_COVERAGE = "partial_outcome_coverage"
    CONTRADICTION_DETECTED = "contradiction_detected"
    OUTCOME_SUFFICIENTLY_COVERED = "outcome_sufficiently_covered"
    OUTCOME_NOT_COVERED = "outcome_not_covered"
    # Depth probe on a strong answer (Slice 8, v2).
    STRONG_ANSWER_DEPTH_PROBE = "strong_answer_depth_probe"
    TIME_RUNNING_LOW = "time_running_low"
    ALL_REQUIRED_OUTCOMES_COVERED = "all_required_outcomes_covered"
    FOLLOWUP_LIMIT_REACHED = "followup_limit_reached"
    STUDENT_REQUESTED_END = "student_requested_end"
    TECHNICAL_ISSUE = "technical_issue"
    CLOSING_REQUIRED = "closing_required"
    OFF_TOPIC_REDIRECT = "off_topic_redirect"
    CANNOT_ANSWER_TRANSITION = "cannot_answer_transition"
    # End-confirmation gate (Slice 4).
    END_CONFIRMATION_REQUESTED = "end_confirmation_requested"
    END_CONFIRMED = "end_confirmed"
    END_CANCELLED = "end_cancelled"


# Follow-up / loop-protection limits (Phase 11). Conservative defaults; Phase 6
# will let a question override ``max_follow_ups`` and Phase 16 will expose these
# in authoring. Kept here as the single source of truth until then.
DEFAULT_MAX_FOLLOWUPS_PER_QUESTION = 2
DEFAULT_MAX_TOTAL_FOLLOWUPS = 12


@dataclass
class InterviewerDecision:
    """Structured interviewer action + the bookkeeping the caller applies.

    ``interviewer_utterance`` is a deterministic placeholder at this layer;
    Phase 9 replaces it with a persona-styled natural-language utterance while
    preserving ``action`` / ``reason_code`` as the authoritative control signal.
    """

    action: InterviewerActionType
    reason_code: ReasonCode

    target_question_id: str | None = None
    target_outcome_id: str | None = None

    acknowledgement_style: AcknowledgementStyle = AcknowledgementStyle.NONE
    acknowledgement: str | None = None
    interviewer_utterance: str = ""

    should_record_academic_evidence: bool = False
    should_advance_question: bool = False
    should_end_session: bool = False

    internal_rationale: str = ""
    tags: list[str] = field(default_factory=list)

    def to_public_dict(self) -> dict[str, object]:
        """Only the fields the transport layer needs — no internal rationale."""
        return {
            "action": self.action.value,
            "reason_code": self.reason_code.value,
            "target_question_id": self.target_question_id,
            "target_outcome_id": self.target_outcome_id,
            "acknowledgement_style": self.acknowledgement_style.value,
            "acknowledgement": self.acknowledgement,
            "interviewer_utterance": self.interviewer_utterance,
            "should_advance_question": self.should_advance_question,
            "should_end_session": self.should_end_session,
        }

    def to_audit_dict(self) -> dict[str, object]:
        """Full payload incl. internal rationale — for state/observability only."""
        return {
            **self.to_public_dict(),
            "should_record_academic_evidence": self.should_record_academic_evidence,
            "internal_rationale": self.internal_rationale,
            "tags": list(self.tags),
        }


@dataclass(frozen=True)
class DecisionInputs:
    """Everything the deterministic policy needs, decoupled from the DB.

    ``time_fraction_remaining`` is None when the interview is untimed. The
    follow-up counters come from the runtime state; ``has_next_question`` says
    whether the adaptive selector found any un-asked candidate (when False the
    only forward move is closing).
    """

    intent: IntentClassification
    analysis: AnswerAnalysis | None
    current_question_follow_up_count: int
    total_follow_up_count: int
    time_fraction_remaining: float | None
    has_next_question: bool
    all_required_outcomes_covered: bool
    max_follow_ups_per_question: int = DEFAULT_MAX_FOLLOWUPS_PER_QUESTION
    max_total_follow_ups: int = DEFAULT_MAX_TOTAL_FOLLOWUPS
    closing_time_fraction: float = 0.1
    # End-confirmation gate (Slice 4): True while a prior turn asked the
    # candidate to confirm ending. It changes how CONFIRM_END / CANCEL_END and a
    # bare answer are interpreted this turn (see _decide_from_intent_request).
    pending_confirmation: bool = False
    # Depth probing (Slice 8, v2): when enabled AND the answer is strong AND we
    # are in CORE/DEEP_PROBE with follow-up budget + time, probe for the ceiling
    # instead of advancing. Defaults False + CORE → byte-for-byte v1 behaviour.
    depth_probe_enabled: bool = False
    phase: InterviewPhase = InterviewPhase.CORE


# Below this fraction of time remaining, stop probing and head for closing.
_LOW_TIME_FRACTION = 0.2


def _probe_action(probe: ProbeType) -> InterviewerActionType:
    return {
        ProbeType.CLARIFICATION: InterviewerActionType.CLARIFY_WITHOUT_REVEALING_ANSWER,
        ProbeType.ASK_FOR_EXAMPLE: InterviewerActionType.ASK_FOR_EXAMPLE,
        ProbeType.PROBE_REASONING: InterviewerActionType.PROBE_DEEPER,
        ProbeType.CHALLENGE_ASSUMPTION: InterviewerActionType.CHALLENGE_REASONING,
        ProbeType.EXPLORE_TRADEOFF: InterviewerActionType.EXPLORE_TRADEOFF,
        ProbeType.RESOLVE_CONTRADICTION: InterviewerActionType.RESOLVE_CONTRADICTION,
        ProbeType.EXTEND_STRONG: InterviewerActionType.EXTEND_ANSWER,
        ProbeType.PROBE_EDGE_CASE: InterviewerActionType.PROBE_EDGE_CASE,
    }.get(probe, InterviewerActionType.PROBE_DEEPER)


def _probe_reason(probe: ProbeType) -> ReasonCode:
    return {
        ProbeType.CLARIFICATION: ReasonCode.ANSWER_TOO_VAGUE,
        ProbeType.ASK_FOR_EXAMPLE: ReasonCode.MISSING_EXAMPLE,
        ProbeType.PROBE_REASONING: ReasonCode.PARTIAL_OUTCOME_COVERAGE,
        ProbeType.CHALLENGE_ASSUMPTION: ReasonCode.PARTIAL_OUTCOME_COVERAGE,
        ProbeType.EXPLORE_TRADEOFF: ReasonCode.PARTIAL_OUTCOME_COVERAGE,
        ProbeType.RESOLVE_CONTRADICTION: ReasonCode.CONTRADICTION_DETECTED,
        ProbeType.EXTEND_STRONG: ReasonCode.STRONG_ANSWER_DEPTH_PROBE,
        ProbeType.PROBE_EDGE_CASE: ReasonCode.STRONG_ANSWER_DEPTH_PROBE,
    }.get(probe, ReasonCode.PARTIAL_OUTCOME_COVERAGE)


def _advance_or_close(inputs: DecisionInputs, reason: ReasonCode) -> InterviewerDecision:
    """Move to the next question, or begin closing when none remain / time low."""
    if not inputs.has_next_question:
        return InterviewerDecision(
            action=InterviewerActionType.BEGIN_CLOSING,
            reason_code=ReasonCode.CLOSING_REQUIRED,
            should_record_academic_evidence=False,
            should_advance_question=False,
            internal_rationale="No further approved questions available; closing.",
        )
    return InterviewerDecision(
        action=InterviewerActionType.TRANSITION_TOPIC,
        reason_code=reason,
        should_advance_question=True,
        acknowledgement_style=AcknowledgementStyle.NEUTRAL,
        internal_rationale=f"Advancing to next question ({reason.value}).",
    )


# Non-academic intents map to a fixed action that NEVER scores. Each entry is
# (action, reason_code, internal_rationale). Handled before any answer analysis.
_SIMPLE_INTENT_ACTIONS: dict[StudentIntent, tuple[InterviewerActionType, ReasonCode, str]] = {
    StudentIntent.TECHNICAL_ISSUE: (
        InterviewerActionType.HANDLE_TECHNICAL_ISSUE,
        ReasonCode.TECHNICAL_ISSUE,
        "Student reported a technical issue; not scored.",
    ),
    StudentIntent.ASK_TO_REPEAT: (
        InterviewerActionType.REPEAT_QUESTION,
        ReasonCode.STUDENT_REQUESTED_REPEAT,
        "Student asked to repeat the question.",
    ),
    StudentIntent.ASK_FOR_CLARIFICATION: (
        InterviewerActionType.CLARIFY_WITHOUT_REVEALING_ANSWER,
        ReasonCode.STUDENT_REQUESTED_CLARIFICATION,
        "Student asked for clarification; do not leak answer.",
    ),
    StudentIntent.ASK_FOR_HINT: (
        InterviewerActionType.PROVIDE_NEUTRAL_HINT,
        ReasonCode.STUDENT_REQUESTED_HINT,
        "Student asked for a neutral scaffold; do not leak answer content.",
    ),
    StudentIntent.ASK_FOR_MORE_TIME: (
        InterviewerActionType.OFFER_BRIEF_PAUSE,
        ReasonCode.STUDENT_REQUESTED_CLARIFICATION,
        "Student asked for more time.",
    ),
}


def _decide_from_intent_request(inputs: DecisionInputs) -> InterviewerDecision | None:
    """Handle student *requests* (rules 1-8) that pre-empt answer analysis.

    Returns None when the intent is a genuine answer that should flow into the
    analysis-driven probing / advancement logic.
    """
    intent = inputs.intent.intent

    # End-confirmation gate (Slice 4).
    if inputs.pending_confirmation:
        # A confirmation is outstanding. An explicit end-confirm — or a repeated
        # natural end request — closes; anything else cancels and resumes the
        # SAME question (never scored, never advanced).
        if intent in (StudentIntent.CONFIRM_END, StudentIntent.END_INTERVIEW):
            return InterviewerDecision(
                action=InterviewerActionType.BEGIN_CLOSING,
                reason_code=ReasonCode.END_CONFIRMED,
                should_record_academic_evidence=False,
                internal_rationale="Candidate confirmed ending; closing.",
                tags=["end_confirmed"],
            )
        return InterviewerDecision(
            action=InterviewerActionType.CANCEL_END,
            reason_code=ReasonCode.END_CANCELLED,
            should_record_academic_evidence=False,
            should_advance_question=False,
            internal_rationale="End cancelled; resume the current question.",
            tags=["end_cancelled"],
        )

    # A fresh end request does NOT close immediately — it asks to confirm.
    if intent is StudentIntent.END_INTERVIEW:
        return InterviewerDecision(
            action=InterviewerActionType.REQUEST_END_CONFIRMATION,
            reason_code=ReasonCode.END_CONFIRMATION_REQUESTED,
            should_record_academic_evidence=False,
            should_advance_question=False,
            internal_rationale="Candidate asked to end; request confirmation.",
            tags=["confirm_end_requested"],
        )

    # A confirm/cancel reply with NO pending confirmation is not an end signal —
    # treat it as an ordinary answer (fall through to analysis-driven handling).
    if intent in (StudentIntent.CONFIRM_END, StudentIntent.CANCEL_END):
        return None

    simple = _SIMPLE_INTENT_ACTIONS.get(intent)
    if simple is not None:
        action, reason, rationale = simple
        return InterviewerDecision(
            action=action,
            reason_code=reason,
            should_record_academic_evidence=False,
            internal_rationale=rationale,
        )

    # Skip — respect it, mark skipped, advance.
    if intent is StudentIntent.SKIP_QUESTION:
        decision = _advance_or_close(inputs, ReasonCode.OUTCOME_NOT_COVERED)
        decision.action = (
            InterviewerActionType.SKIP_QUESTION
            if decision.should_advance_question
            else decision.action
        )
        decision.internal_rationale = "Student asked to skip; recorded as skipped."
        decision.tags = ["skipped"]
        return decision

    # Cannot answer — acknowledge, record insufficient evidence, advance.
    if intent is StudentIntent.CANNOT_ANSWER:
        decision = _advance_or_close(inputs, ReasonCode.CANNOT_ANSWER_TRANSITION)
        decision.acknowledgement_style = AcknowledgementStyle.NEUTRAL
        decision.should_record_academic_evidence = True
        decision.internal_rationale = (
            "Student cannot answer; record insufficient evidence and move on."
        )
        decision.tags = ["insufficient_evidence"]
        return decision

    # Off-topic — redirect once, then advance if it persists.
    analysis = inputs.analysis
    if intent is StudentIntent.OFF_TOPIC or (
        analysis is not None and analysis.relevance is Relevance.OFF_TOPIC
    ):
        if inputs.current_question_follow_up_count == 0:
            return InterviewerDecision(
                action=InterviewerActionType.REDIRECT_TO_TOPIC,
                reason_code=ReasonCode.OFF_TOPIC_REDIRECT,
                should_record_academic_evidence=False,
                internal_rationale="Off-topic; redirect once before moving on.",
            )
        decision = _advance_or_close(inputs, ReasonCode.OUTCOME_NOT_COVERED)
        decision.internal_rationale = "Still off-topic after redirect; advancing."
        return decision

    return None


def decide_next_action(inputs: DecisionInputs) -> InterviewerDecision:
    """Deterministically map perception + state to an interviewer decision.

    Precedence (highest first) — chosen so student *requests* and hard limits
    always win over probing heuristics:

    1-8. Student requests / non-answers (technical issue, end, repeat, clarify,
         more time, skip, cannot-answer, off-topic) — see
         :func:`_decide_from_intent_request`.
    9. Time running low       → stop probing, advance/close.
    10. Follow-up limit hit   → advance/close.
    11. Analysis recommends a probe → probe (record evidence).
    12. Otherwise             → advance to next question / close.
    """
    request_decision = _decide_from_intent_request(inputs)
    if request_decision is not None:
        return request_decision

    analysis = inputs.analysis
    # From here the intent is a genuine (partial) answer → we CAN record evidence.
    # 9. Time low → stop probing, advance / close.
    if (
        inputs.time_fraction_remaining is not None
        and inputs.time_fraction_remaining <= inputs.closing_time_fraction
    ):
        decision = _advance_or_close(inputs, ReasonCode.TIME_RUNNING_LOW)
        decision.should_record_academic_evidence = True
        decision.internal_rationale = "Closing-threshold time reached; wrap up."
        return decision

    time_low = (
        inputs.time_fraction_remaining is not None
        and inputs.time_fraction_remaining <= _LOW_TIME_FRACTION
    )

    # 10. Follow-up limits (loop protection).
    followups_exhausted = (
        inputs.current_question_follow_up_count >= inputs.max_follow_ups_per_question
        or inputs.total_follow_up_count >= inputs.max_total_follow_ups
    )

    # 11. Probe when the analysis recommends it AND we're allowed to.
    probe = analysis.recommended_probe_type if analysis is not None else ProbeType.NONE
    if probe is not ProbeType.NONE and not followups_exhausted and not time_low:
        return InterviewerDecision(
            action=_probe_action(probe),
            reason_code=_probe_reason(probe),
            should_record_academic_evidence=True,
            should_advance_question=False,
            acknowledgement_style=AcknowledgementStyle.NEUTRAL,
            internal_rationale=f"Analysis recommends probe={probe.value}.",
            tags=["probe", probe.value],
        )

    # 11.5 Depth probe (Slice 8, v2): dig into a STRONG answer to find the
    # candidate's ceiling instead of advancing. Only when the feature is
    # enabled, the answer is strong, we are in CORE/DEEP_PROBE, and we still
    # have follow-up budget + time. DEEP_PROBE pushes on edge cases; CORE asks
    # them to extend. Consumes the follow-up budget (falls into the else branch
    # of _apply_state_updates), so loop protection is preserved.
    if (
        inputs.depth_probe_enabled
        and analysis is not None
        and is_strong_answer(analysis)
        and inputs.phase in (InterviewPhase.CORE, InterviewPhase.DEEP_PROBE)
        and not followups_exhausted
        and not time_low
    ):
        depth = (
            ProbeType.PROBE_EDGE_CASE
            if inputs.phase is InterviewPhase.DEEP_PROBE
            else ProbeType.EXTEND_STRONG
        )
        return InterviewerDecision(
            action=_probe_action(depth),
            reason_code=ReasonCode.STRONG_ANSWER_DEPTH_PROBE,
            should_record_academic_evidence=True,
            should_advance_question=False,
            acknowledgement_style=AcknowledgementStyle.POSITIVE,
            internal_rationale="Strong answer; probing for depth/ceiling.",
            tags=["depth_probe", depth.value],
        )

    # 12. Otherwise advance (recording this answer's evidence first).
    if followups_exhausted:
        reason = ReasonCode.FOLLOWUP_LIMIT_REACHED
    elif inputs.all_required_outcomes_covered:
        reason = ReasonCode.ALL_REQUIRED_OUTCOMES_COVERED
    elif analysis is not None and analysis.correctness in (
        Correctness.CORRECT,
        Correctness.MOSTLY_CORRECT,
    ):
        reason = ReasonCode.OUTCOME_SUFFICIENTLY_COVERED
    else:
        reason = ReasonCode.PARTIAL_OUTCOME_COVERAGE

    if inputs.all_required_outcomes_covered and not time_low:
        # Everything required is covered → begin closing rather than pad.
        return InterviewerDecision(
            action=InterviewerActionType.BEGIN_CLOSING,
            reason_code=ReasonCode.ALL_REQUIRED_OUTCOMES_COVERED,
            should_record_academic_evidence=True,
            internal_rationale="All required outcomes covered; begin closing.",
        )

    decision = _advance_or_close(inputs, reason)
    decision.should_record_academic_evidence = True
    return decision


__all__ = [
    "DEFAULT_MAX_FOLLOWUPS_PER_QUESTION",
    "DEFAULT_MAX_TOTAL_FOLLOWUPS",
    "AcknowledgementStyle",
    "DecisionInputs",
    "InterviewerActionType",
    "InterviewerDecision",
    "ReasonCode",
    "decide_next_action",
]
