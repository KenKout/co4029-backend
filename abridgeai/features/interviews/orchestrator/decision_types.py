"""Interviewer decision enums (extracted from decision.py).

Pure value enums with no dependency on the decision *logic* — factored into
their own module so ``decision.py`` stays under the feature-wide 800-LOC "no god
files" ceiling as the v2 slices add action types / reason codes. ``decision.py``
re-exports every name here, so existing ``from ...decision import ReasonCode``
imports keep working unchanged.
"""

from __future__ import annotations

from enum import Enum

from abridgeai.features.interviews.orchestrator.intent import StudentIntent


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
    # Rich closing sub-steps (Slice 13, v2). Deterministic, brief, answer-safe
    # turns that run during the CLOSING phase before the final sign-off: prompt
    # the candidate to self-reflect, then invite their questions, then close.
    PROMPT_SELF_REFLECTION = "prompt_self_reflection"
    INVITE_CANDIDATE_QUESTIONS = "invite_candidate_questions"
    ANSWER_CANDIDATE_QUESTION = "answer_candidate_question"
    # Frustration de-escalation (Slice 19A, v2): acknowledge the candidate's
    # frustration and resume the SAME question — tone/flow only, never scored.
    DEESCALATE = "deescalate"
    # Mid-interview question deferral (Slice 19B, v2): briefly defer a candidate
    # question asked outside closing and resume the current question.
    DEFER_CANDIDATE_QUESTION = "defer_candidate_question"


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
    # Confident-but-wrong forced challenge (Slice 16, v2).
    CONFIDENT_BUT_WRONG_CHALLENGE = "confident_but_wrong_challenge"
    # Rambling redirect (Slice 17, v2): steer a long, on-topic, low-substance ramble.
    RAMBLING_REDIRECT = "rambling_redirect"
    TIME_RUNNING_LOW = "time_running_low"
    ALL_REQUIRED_OUTCOMES_COVERED = "all_required_outcomes_covered"
    FOLLOWUP_LIMIT_REACHED = "followup_limit_reached"
    STUDENT_REQUESTED_END = "student_requested_end"
    TECHNICAL_ISSUE = "technical_issue"
    CLOSING_REQUIRED = "closing_required"
    OFF_TOPIC_REDIRECT = "off_topic_redirect"
    CANNOT_ANSWER_TRANSITION = "cannot_answer_transition"
    # Assistance laddering on "I don't know" (Slice 11, v2): a hint was offered
    # on the SAME question instead of abandoning it on the first non-answer.
    CANNOT_ANSWER_HINT_OFFERED = "cannot_answer_hint_offered"
    # End-confirmation gate (Slice 4).
    END_CONFIRMATION_REQUESTED = "end_confirmation_requested"
    END_CONFIRMED = "end_confirmed"
    END_CANCELLED = "end_cancelled"
    # Rich closing sub-steps (Slice 13, v2).
    CLOSING_SELF_REFLECTION = "closing_self_reflection"
    CLOSING_INVITE_QUESTIONS = "closing_invite_questions"
    CLOSING_ANSWERED_QUESTION = "closing_answered_question"
    # Frustration de-escalation (Slice 19A, v2).
    CANDIDATE_FRUSTRATED = "candidate_frustrated"
    # Mid-interview question deferral (Slice 19B, v2).
    CANDIDATE_QUESTION_DEFERRED = "candidate_question_deferred"


# Non-academic intents map to a fixed action that NEVER scores. Each entry is
# (action, reason_code, internal_rationale). Handled before any answer analysis.
SIMPLE_INTENT_ACTIONS: dict[StudentIntent, tuple[InterviewerActionType, ReasonCode, str]] = {
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


__all__ = [
    "SIMPLE_INTENT_ACTIONS",
    "AcknowledgementStyle",
    "InterviewerActionType",
    "ReasonCode",
]
