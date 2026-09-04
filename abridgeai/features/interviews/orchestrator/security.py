"""Typed prompt-injection security contracts for interview runtime turns.

The types in this module deliberately contain no free-form model reasoning.
They are safe to persist as compact audit/state data and are shared by REST,
hybrid, and LiveKit voice interviews.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

SECURITY_POLICY_VERSION = "2026-07-19"
# 1.3.0 — hardening phases 1.2 (internal analysis-contract fields) and 1.3
# (encoding canonicalisation + the hexadecimal false positive it removed). Bumped
# because sessions stamp this per attempt, so a verdict stays attributable to the
# ruleset that produced it.
SECURITY_RULES_VERSION = "1.3.0"
SECURITY_PROMPT_VERSION = "1.1.0"
OUTPUT_GUARD_VERSION = "1.0.0"


class SecurityCategory(str, Enum):  # noqa: UP042 -- preserve persisted values
    BENIGN = "benign"
    FUTURE_QUESTION_REQUEST = "future_question_request"
    ANSWER_KEY_REQUEST = "answer_key_request"
    RUBRIC_EXFILTRATION = "rubric_exfiltration"
    SYSTEM_PROMPT_REQUEST = "system_prompt_request"
    INSTRUCTION_OVERRIDE = "instruction_override"
    ROLEPLAY_BYPASS = "roleplay_bypass"
    GRADING_MANIPULATION = "grading_manipulation"
    HIDDEN_STATE_REQUEST = "hidden_state_request"
    ENCODED_EXFILTRATION = "encoded_exfiltration"
    CROSS_SESSION_DATA_REQUEST = "cross_session_data_request"


class SecurityAction(str, Enum):  # noqa: UP042 -- preserve API/audit values
    ALLOW = "allow"
    REPEAT_CURRENT_QUESTION = "repeat_current_question"
    CLARIFY_CURRENT_QUESTION = "clarify_current_question"
    EXPLAIN_CURRENT_TERM = "explain_current_term"
    HINT_CURRENT_QUESTION = "hint_current_question"
    REFUSE_AND_REDIRECT = "refuse_and_redirect"
    WARN_AND_REDIRECT = "warn_and_redirect"
    END_AND_FLAG = "end_and_flag"


class SecurityResponsePolicy(str, Enum):  # noqa: UP042
    CONTINUE_AND_LOG = "continue_and_log"
    WARN_AND_CONTINUE = "warn_and_continue"
    END_AND_FLAG = "end_and_flag"


@dataclass(frozen=True)
class SecurityAssessment:
    """A bounded semantic verdict with no chain-of-thought or raw content."""

    category: SecurityCategory
    detected: bool
    confidence: float
    should_block: bool
    should_record_academic_evidence: bool
    response_key: str | None
    normalized_fingerprint: str | None
    source: str = "rules"
    classifier_failed: bool = False


@dataclass(frozen=True)
class ProtectedContent:
    """One server-side protected phrase considered by the output guard."""

    category: str
    text: str
    content_id: str | None = None


@dataclass(frozen=True)
class OutputLeakageAssessment:
    """Deterministic output-guard result; never contains matched secret text."""

    blocked: bool
    protected_content_category: str | None = None
    normalized_fingerprint: str | None = None
    match_method: str | None = None


def confidence_band(confidence: float) -> str:
    if confidence >= 0.9:
        return "high"
    if confidence >= 0.65:
        return "medium"
    return "low"


__all__ = [
    "OUTPUT_GUARD_VERSION",
    "SECURITY_POLICY_VERSION",
    "SECURITY_PROMPT_VERSION",
    "SECURITY_RULES_VERSION",
    "OutputLeakageAssessment",
    "ProtectedContent",
    "SecurityAction",
    "SecurityAssessment",
    "SecurityCategory",
    "SecurityResponsePolicy",
    "confidence_band",
]
