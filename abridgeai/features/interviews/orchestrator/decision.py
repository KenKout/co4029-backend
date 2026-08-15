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

from abridgeai.features.interviews.orchestrator.analysis import (
    AnswerAnalysis,
    Correctness,
    ProbeType,
    Relevance,
)
from abridgeai.features.interviews.orchestrator.coverage import (
    is_confidently_wrong,
    is_strong_answer,
)

# Action / acknowledgement / reason enums live in a sibling module so this file
# stays under the feature-wide 800-LOC ceiling as v2 slices add cases. Re-export
# them here so existing ``from ...decision import ReasonCode`` imports (21 sites)
# keep working unchanged.
from abridgeai.features.interviews.orchestrator.decision_types import (  # noqa: E402
    SIMPLE_INTENT_ACTIONS,
    AcknowledgementStyle,
    InterviewerActionType,
    ReasonCode,
)
from abridgeai.features.interviews.orchestrator.intent import (
    IntentClassification,
    StudentIntent,
)
from abridgeai.features.interviews.orchestrator.state import InterviewPhase

# Follow-up / loop-protection limits (Phase 11). Conservative defaults; Phase 6
# will let a question override ``max_follow_ups`` and Phase 16 will expose these
# in authoring. Kept here as the single source of truth until then.
DEFAULT_MAX_FOLLOWUPS_PER_QUESTION = 2
DEFAULT_MAX_TOTAL_FOLLOWUPS = 12

# How many escalating hints a candidate may receive on ONE question before a
# non-answer advances anyway (Slice 11, v2). Bounded so the hint ladder can
# never hold the interview on a single question: at this level the CANNOT_ANSWER
# branch falls through to the v1 advance.
# Per QUESTION: ``hint_level`` resets on advance (turn_state.py). Mirrored in the
# learner UI as MAX_HINTS_PER_QUESTION (frontend/src/lib/interview/hint-ladder.ts)
# — change both together. Reaching the deepest rung needs follow-up budget too.
MAX_CANNOT_ANSWER_HINTS = 3


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
    max_hints_per_question: int = MAX_CANNOT_ANSWER_HINTS
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
    # Rich closing (Slice 13, v2): when enabled, the CLOSING phase runs a short
    # deterministic sub-sequence (self-reflection → invite questions → sign-off)
    # instead of a one-shot close. ``closing_step`` is the current marker from
    # state (""/"reflection"/"questions"/"done"). Off → v1 one-shot close.
    rich_closing_enabled: bool = False
    closing_step: str = ""
    # Self-correction (Slice 15, v2): when enabled AND the analysis flags the
    # candidate fixed their own mistake, reward it with a POSITIVE acknowledgement
    # and suppress a RESOLVE_CONTRADICTION probe pointing at what they already
    # resolved. Off → the self_corrected signal is ignored (byte-for-byte v1).
    self_correction_enabled: bool = False
    # Confident-but-wrong forced challenge (Slice 16, v2): when enabled AND the
    # answer is confidently wrong (relevant, specific, high-confidence
    # incorrect/mixed) AND budget + time remain AND the analyzer recommended no
    # other probe, force a CHALLENGE_REASONING probe instead of advancing. Off →
    # the confidently-wrong answer advances as in v1.
    confident_wrong_challenge_enabled: bool = False
    # Rambling redirect (Slice 17, v2): ``rambling`` is the plain affect signal
    # (candidate gave a long, on-topic, low-substance answer). When enabled AND
    # rambling AND the answer is on-topic AND budget + time remain AND no other
    # probe fired, steer back with REDIRECT_TO_TOPIC instead of advancing. Off →
    # the rambling signal is ignored (byte-for-byte v1).
    rambling: bool = False
    rambling_redirect_enabled: bool = False
    # Frustration de-escalation (Slice 19A, v2): when enabled AND the intent is
    # FRUSTRATED, acknowledge and resume the SAME question (never scored, never
    # advanced, does not consume the follow-up budget). Off → FRUSTRATED is not
    # a recognised request and falls through to answer handling (byte-for-byte v1).
    frustration_deescalation_enabled: bool = False
    # Mid-interview question deferral (Slice 19B, v2): when enabled AND the intent
    # is ASK_INTERVIEWER_QUESTION OUTSIDE the closing phase, briefly defer and
    # resume the current question. Off → the intent falls through to v1 handling.
    question_deferral_enabled: bool = False
    # Assistance laddering on a non-answer (Slice 11, v2). When enabled, a
    # CANNOT_ANSWER intent offers an escalating hint on the SAME question for the
    # first ``MAX_CANNOT_ANSWER_HINTS`` attempts instead of advancing on the very
    # first "I don't know". ``hint_level`` is the ladder position carried in
    # runtime state (reset to 0 on every advance, so escalation is per-question).
    # Off → CANNOT_ANSWER advances immediately (byte-for-byte v1).
    hint_ladder_enabled: bool = False
    hint_level: int = 0


# Below this fraction of time remaining, stop probing and head for closing.
_LOW_TIME_FRACTION = 0.2


def _below_closing_threshold(inputs: DecisionInputs) -> bool:
    return (
        inputs.time_fraction_remaining is not None
        and inputs.time_fraction_remaining <= inputs.closing_time_fraction
    )


def _cannot_answer_decision(inputs: DecisionInputs) -> InterviewerDecision:
    """Resolve a CANNOT_ANSWER turn: hint ladder first, then v1 advance.

    A CANNOT_ANSWER intent gets an escalating neutral hint on the SAME question
    (Slice 11, v2) while the ladder AND the follow-up budgets AND the closing-
    time threshold allow it; once any gate trips, the turn falls through to the
    exact v1 behaviour (record insufficient evidence and advance). The budgets
    keep the loop-freedom invariant: a candidate who keeps saying "I don't know"
    gets at most MAX_CANNOT_ANSWER_HINTS hints before the interview moves on.
    """
    if (
        inputs.hint_ladder_enabled
        and inputs.hint_level < inputs.max_hints_per_question
        and inputs.current_question_follow_up_count < inputs.max_follow_ups_per_question
        and inputs.total_follow_up_count < inputs.max_total_follow_ups
        and not _below_closing_threshold(inputs)
    ):
        return InterviewerDecision(
            action=InterviewerActionType.PROVIDE_NEUTRAL_HINT,
            reason_code=ReasonCode.CANNOT_ANSWER_HINT_OFFERED,
            # Assistance, not assessment: the candidate has not answered yet, so
            # there is nothing to score. Evidence is recorded on the advance turn
            # once the ladder is exhausted.
            should_record_academic_evidence=False,
            should_advance_question=False,
            # NOT positive: the ack table renders POSITIVE as praise ("That's
            # helpful."), and nothing was answered here. The hint IS the response;
            # warmth belongs to the affect lead-in, not to praising a non-answer.
            acknowledgement_style=AcknowledgementStyle.NONE,
            internal_rationale=(
                "Candidate cannot answer; offer a hint on the same question "
                f"(ladder level {inputs.hint_level})."
            ),
            tags=["hint_ladder", "cannot_answer"],
        )
    decision = _advance_or_close(inputs, ReasonCode.CANNOT_ANSWER_TRANSITION)
    decision.acknowledgement_style = AcknowledgementStyle.NEUTRAL
    decision.should_record_academic_evidence = True
    decision.internal_rationale = "Student cannot answer; record insufficient evidence and move on."
    decision.tags = ["insufficient_evidence"]
    return decision


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


def _rich_closing_step(inputs: DecisionInputs) -> InterviewerDecision | None:
    """Advance the rich-closing sub-sequence (Slice 13, v2).

    Runs a short deterministic sequence once closing is reached, keyed by the
    ``closing_step`` marker from state:

        ""          → prompt self-reflection      (marker becomes "reflection")
        "reflection"→ invite candidate questions  (marker becomes "questions")
        "questions" → answer a candidate question OR final sign-off
        "done"      → final sign-off (CLOSE_INTERVIEW)

    Every sub-step except the final one is a NON-finishing turn (mapping.py only
    finishes on BEGIN_CLOSING / CLOSE_INTERVIEW), so the interview keeps
    collecting one more reply. None means rich closing does not apply (feature
    off), so the caller falls back to the v1 one-shot close.
    """
    if not inputs.rich_closing_enabled:
        return None

    step = inputs.closing_step
    if step == "":
        return InterviewerDecision(
            action=InterviewerActionType.PROMPT_SELF_REFLECTION,
            reason_code=ReasonCode.CLOSING_SELF_REFLECTION,
            should_record_academic_evidence=False,
            should_advance_question=False,
            acknowledgement_style=AcknowledgementStyle.POSITIVE,
            internal_rationale="Closing: prompt candidate self-reflection.",
            tags=["closing", "self_reflection"],
        )
    if step == "reflection":
        return InterviewerDecision(
            action=InterviewerActionType.INVITE_CANDIDATE_QUESTIONS,
            reason_code=ReasonCode.CLOSING_INVITE_QUESTIONS,
            should_record_academic_evidence=False,
            should_advance_question=False,
            acknowledgement_style=AcknowledgementStyle.NEUTRAL,
            internal_rationale="Closing: invite candidate questions.",
            tags=["closing", "invite_questions"],
        )
    if step == "questions" and inputs.intent.intent is StudentIntent.ASK_INTERVIEWER_QUESTION:
        # Candidate asked something — acknowledge briefly (answer-safe), then the
        # NEXT turn (still "questions") will sign off. Non-finishing.
        return InterviewerDecision(
            action=InterviewerActionType.ANSWER_CANDIDATE_QUESTION,
            reason_code=ReasonCode.CLOSING_ANSWERED_QUESTION,
            should_record_academic_evidence=False,
            should_advance_question=False,
            acknowledgement_style=AcknowledgementStyle.NEUTRAL,
            internal_rationale="Closing: acknowledge candidate question (answer-safe).",
            tags=["closing", "answer_question"],
        )
    # step == "questions" (no question asked) or "done" → final sign-off.
    return InterviewerDecision(
        action=InterviewerActionType.CLOSE_INTERVIEW,
        reason_code=ReasonCode.CLOSING_REQUIRED,
        should_record_academic_evidence=False,
        should_advance_question=False,
        acknowledgement_style=AcknowledgementStyle.POSITIVE,
        internal_rationale="Closing: final sign-off.",
        tags=["closing", "sign_off"],
    )


def _begin_closing_decision(
    inputs: DecisionInputs, reason: ReasonCode, rationale: str
) -> InterviewerDecision:
    """The turn that ENTERS closing.

    Rich closing (Slice 13): emit the first sub-step (self-reflection), a
    NON-finishing turn that also flips the phase to CLOSING in turn_state so
    subsequent turns route through the sub-sequence. Off → the v1 one-shot
    BEGIN_CLOSING (finishing).
    """
    if inputs.rich_closing_enabled:
        rich = _rich_closing_step(inputs)
        if rich is not None:
            return rich
    return InterviewerDecision(
        action=InterviewerActionType.BEGIN_CLOSING,
        reason_code=reason,
        should_record_academic_evidence=False,
        should_advance_question=False,
        internal_rationale=rationale,
    )


def _reward_self_correction(decision: InterviewerDecision, *, self_corrected: bool) -> None:
    """Reward a self-correction (Slice 15, v2) on an advancing/closing decision.

    When the candidate fixed their own mistake this turn, upgrade the
    acknowledgement to POSITIVE and tag it — a real interviewer credits the
    catch. Only ever touches TONE (ack style + tag); the action, advance/close,
    and evidence flags are untouched, so every decision invariant holds. No-op
    when the feature is off or the answer was not self-corrected.
    """
    if not self_corrected:
        return
    decision.acknowledgement_style = AcknowledgementStyle.POSITIVE
    if "self_correction" not in decision.tags:
        decision.tags.append("self_correction")


def _depth_probe(
    inputs: DecisionInputs, *, followups_exhausted: bool, time_low: bool
) -> InterviewerDecision | None:
    """Depth probe on a STRONG answer to find the candidate's ceiling (Slice 8).

    Only when the feature is enabled, the answer is strong, we are in
    CORE/DEEP_PROBE, and we still have follow-up budget + time. DEEP_PROBE
    pushes on edge cases; CORE asks them to extend. Consumes the follow-up
    budget (falls into the else branch of _apply_state_updates), so loop
    protection is preserved. Returns None when it does not apply.
    """
    if not (
        inputs.depth_probe_enabled
        and inputs.analysis is not None
        and is_strong_answer(inputs.analysis)
        and inputs.phase in (InterviewPhase.CORE, InterviewPhase.DEEP_PROBE)
        and not followups_exhausted
        and not time_low
    ):
        return None
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


def _confident_wrong_challenge(
    inputs: DecisionInputs, *, followups_exhausted: bool, time_low: bool
) -> InterviewerDecision | None:
    """Forced challenge on a confident-but-wrong answer (Slice 16, v2).

    The candidate committed confidently to a specific, relevant, but WRONG
    claim, and no other probe was recommended. Instead of quietly advancing
    (which feels like the interviewer didn't notice), lean in with a
    CHALLENGE_REASONING probe so they can defend or revise. Gated on budget +
    time, so loop protection holds. Returns None when the feature is off, the
    answer is not confidently wrong, or budget/time are spent → the caller
    advances as in v1.
    """
    if not (
        inputs.confident_wrong_challenge_enabled
        and is_confidently_wrong(inputs.analysis)
        and not followups_exhausted
        and not time_low
    ):
        return None
    return InterviewerDecision(
        action=InterviewerActionType.CHALLENGE_REASONING,
        reason_code=ReasonCode.CONFIDENT_BUT_WRONG_CHALLENGE,
        should_record_academic_evidence=True,
        should_advance_question=False,
        acknowledgement_style=AcknowledgementStyle.CORRECTIVE,
        internal_rationale="Confident but wrong; challenge the reasoning.",
        tags=["confident_wrong", "challenge"],
    )


def _rambling_redirect(
    inputs: DecisionInputs, *, followups_exhausted: bool, time_low: bool
) -> InterviewerDecision | None:
    """Steer a long, on-topic, low-substance ramble back to focus (Slice 17, v2).

    The candidate is meandering: the affect layer flagged the answer as rambling
    and it is on-topic (off-topic is already handled at rule 8). Instead of
    quietly advancing past the sprawl, interrupt gently and redirect them to the
    point. Gated on budget + time (REDIRECT_TO_TOPIC consumes the follow-up
    budget, so loop protection holds). Returns None when the feature is off, the
    answer is not rambling, it is off-topic, or budget/time are spent → the
    caller advances as in v1.
    """
    analysis = inputs.analysis
    off_topic = analysis is not None and analysis.relevance is Relevance.OFF_TOPIC
    if not (
        inputs.rambling_redirect_enabled
        and inputs.rambling
        and not off_topic
        and not followups_exhausted
        and not time_low
    ):
        return None
    return InterviewerDecision(
        action=InterviewerActionType.REDIRECT_TO_TOPIC,
        reason_code=ReasonCode.RAMBLING_REDIRECT,
        should_record_academic_evidence=True,
        should_advance_question=False,
        acknowledgement_style=AcknowledgementStyle.NEUTRAL,
        internal_rationale="Rambling; steer back to the focus of the question.",
        tags=["rambling", "redirect"],
    )


def _advance_reason(inputs: DecisionInputs, *, followups_exhausted: bool) -> ReasonCode:
    """Pick the reason code for a plain advance (rule 12).

    Precedence: exhausted budget → all-covered → (mostly-)correct answer →
    partial coverage. Extracted from ``decide_next_action`` to keep that
    function under the cyclomatic-complexity cap.
    """
    if followups_exhausted:
        return ReasonCode.FOLLOWUP_LIMIT_REACHED
    if inputs.all_required_outcomes_covered:
        return ReasonCode.ALL_REQUIRED_OUTCOMES_COVERED
    analysis = inputs.analysis
    if analysis is not None and analysis.correctness in (
        Correctness.CORRECT,
        Correctness.MOSTLY_CORRECT,
    ):
        return ReasonCode.OUTCOME_SUFFICIENTLY_COVERED
    return ReasonCode.PARTIAL_OUTCOME_COVERAGE


def _advance_or_close(inputs: DecisionInputs, reason: ReasonCode) -> InterviewerDecision:
    """Move to the next question, or begin closing when none remain / time low."""
    if not inputs.has_next_question:
        return _begin_closing_decision(
            inputs,
            ReasonCode.CLOSING_REQUIRED,
            "No further approved questions available; closing.",
        )
    return InterviewerDecision(
        action=InterviewerActionType.TRANSITION_TOPIC,
        reason_code=reason,
        should_advance_question=True,
        acknowledgement_style=AcknowledgementStyle.NEUTRAL,
        internal_rationale=f"Advancing to next question ({reason.value}).",
    )


def _decide_from_intent_request(inputs: DecisionInputs) -> InterviewerDecision | None:
    """Handle student *requests* (rules 1-8) that pre-empt answer analysis.

    Returns None when the intent is a genuine answer that should flow into the
    analysis-driven probing / advancement logic.
    """
    intent = inputs.intent.intent

    # Frustration de-escalation (Slice 19A, v2). Acknowledge the candidate's
    # frustration and resume the SAME question — never scored, never advanced,
    # and NOT gated on the follow-up budget (this is candidate support, not an
    # academic probe). Off → FRUSTRATED falls through to answer handling (v1).
    if inputs.frustration_deescalation_enabled and intent is StudentIntent.FRUSTRATED:
        return InterviewerDecision(
            action=InterviewerActionType.DEESCALATE,
            reason_code=ReasonCode.CANDIDATE_FRUSTRATED,
            should_record_academic_evidence=False,
            should_advance_question=False,
            acknowledgement_style=AcknowledgementStyle.POSITIVE,
            internal_rationale="Candidate frustrated; de-escalate and resume.",
            tags=["frustrated"],
        )

    # Mid-interview question deferral (Slice 19B, v2). When the candidate asks
    # the interviewer a question OUTSIDE closing, briefly defer and resume the
    # current question. During CLOSING the rich-closing sub-sequence (which runs
    # before this) answers it instead. Off → falls through to v1 handling.
    if (
        inputs.question_deferral_enabled
        and intent is StudentIntent.ASK_INTERVIEWER_QUESTION
        and inputs.phase is not InterviewPhase.CLOSING
    ):
        return InterviewerDecision(
            action=InterviewerActionType.DEFER_CANDIDATE_QUESTION,
            reason_code=ReasonCode.CANDIDATE_QUESTION_DEFERRED,
            should_record_academic_evidence=False,
            should_advance_question=False,
            acknowledgement_style=AcknowledgementStyle.NEUTRAL,
            internal_rationale="Candidate asked a question mid-interview; defer to closing.",
            tags=["deferred"],
        )

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

    simple = SIMPLE_INTENT_ACTIONS.get(intent)
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

    # Cannot answer — hint ladder first (Slice 11, v2); when it does not apply
    # the helper falls through to the v1 behaviour (record insufficient evidence
    # and advance).
    if intent is StudentIntent.CANNOT_ANSWER:
        return _cannot_answer_decision(inputs)

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
    # Rich closing continuation (Slice 13, v2). Once the phase is CLOSING and the
    # feature is on, the closing sub-sequence owns every turn (self-reflection →
    # invite questions → answer/​sign-off), regardless of intent — so an end
    # request during closing simply signs off. Off → falls through to v1.
    if inputs.rich_closing_enabled and inputs.phase is InterviewPhase.CLOSING:
        rich = _rich_closing_step(inputs)
        if rich is not None:
            return rich

    request_decision = _decide_from_intent_request(inputs)
    if request_decision is not None:
        return request_decision

    analysis = inputs.analysis
    # From here the intent is a genuine (partial) answer → we CAN record evidence.
    # 9. Time low → stop probing, advance / close.
    if _below_closing_threshold(inputs):
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

    # Self-correction (Slice 15, v2): the candidate noticed and fixed their own
    # mistake this turn. When enabled, don't point a RESOLVE_CONTRADICTION probe
    # at what they already resolved — that would feel like the interviewer wasn't
    # listening. Off → self_corrected is ignored (byte-for-byte v1).
    self_corrected = (
        inputs.self_correction_enabled and analysis is not None and analysis.self_corrected
    )

    # 11. Probe when the analysis recommends it AND we're allowed to.
    probe = analysis.recommended_probe_type if analysis is not None else ProbeType.NONE
    if self_corrected and probe is ProbeType.RESOLVE_CONTRADICTION:
        probe = ProbeType.NONE
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

    # 11.5 Depth probe on a strong answer (Slice 8, v2).
    depth_probe = _depth_probe(inputs, followups_exhausted=followups_exhausted, time_low=time_low)
    if depth_probe is not None:
        return depth_probe

    # 11.6 Confident-but-wrong forced challenge (Slice 16, v2).
    challenge = _confident_wrong_challenge(
        inputs, followups_exhausted=followups_exhausted, time_low=time_low
    )
    if challenge is not None:
        return challenge

    # 11.7 Rambling redirect (Slice 17, v2).
    redirect = _rambling_redirect(
        inputs, followups_exhausted=followups_exhausted, time_low=time_low
    )
    if redirect is not None:
        return redirect

    # 12. Otherwise advance (recording this answer's evidence first).
    reason = _advance_reason(inputs, followups_exhausted=followups_exhausted)

    if inputs.all_required_outcomes_covered and not time_low:
        # Everything required is covered → begin closing rather than pad.
        decision = _begin_closing_decision(
            inputs,
            ReasonCode.ALL_REQUIRED_OUTCOMES_COVERED,
            "All required outcomes covered; begin closing.",
        )
        decision.should_record_academic_evidence = True
        _reward_self_correction(decision, self_corrected=self_corrected)
        return decision

    decision = _advance_or_close(inputs, reason)
    decision.should_record_academic_evidence = True
    _reward_self_correction(decision, self_corrected=self_corrected)
    return decision


__all__ = [
    "DEFAULT_MAX_FOLLOWUPS_PER_QUESTION",
    "DEFAULT_MAX_TOTAL_FOLLOWUPS",
    "MAX_CANNOT_ANSWER_HINTS",
    "AcknowledgementStyle",
    "DecisionInputs",
    "InterviewerActionType",
    "InterviewerDecision",
    "ReasonCode",
    "decide_next_action",
]
