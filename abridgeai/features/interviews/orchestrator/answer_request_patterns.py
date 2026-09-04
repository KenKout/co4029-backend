"""Answer-solicitation cue patterns for the interview security rules.

Split out of ``security_logic`` to keep that module under the orchestrator's
800-line "no god files" ceiling. Pattern data only.

Two tiers live here, and the distinction matters:

* ``DIRECT_ANSWER_REQUEST`` / ``ANSWER_REFERENCE_REQUEST`` are FINAL — the shape
  is unambiguous ("give me the correct answer", "nhắc lại đáp án"), so the rules
  classify on them directly.
* ``ANSWER_REQUEST_CUE`` / ``ANSWER_WORK_CUE`` are concept-level ROUTING signals,
  never verdicts. A broad request needs both a delegation cue and an academic-work
  cue before it is even considered ambiguous, and the semantic classifier then
  decides. Keeping them deliberately loose is safe only because they gate the
  ambiguity path rather than blocking on their own.
"""

from __future__ import annotations

DIRECT_ANSWER_REQUEST = (
    r"\b(?:(?:please\s+)?|(?:(?:can|could|would|will)\s+you\s+))"
    r"(?:give|tell|show|provide)\s+me\s+(?:the\s+|an?\s+)?(?:correct\s+)?answers?\b|"
    r"^what(?:'|’)s\s+(?:the\s+)?(?:correct\s+)?answer\b|"
    r"^what\s+is\s+(?:the\s+)?(?:correct\s+)?answer\b|"
    r"^(?:please\s+)?answer\s+(?:this|the|the\s+current|current)\s+question\s+for\s+me\b|"
    r"^(?:(?:hãy|vui\s+lòng|xin)\s+)?(?:cho|nói|cung\s+cấp)\s+"
    r"(?:tôi|mình|em)\s+(?:đáp\s*án|câu\s*trả\s*lời)\b|"
    r"^(?:(?:hãy|vui\s+lòng|xin)\s+)?trả\s+lời\s+"
    r"(?:câu\s+hỏi\s+)?(?:này|hiện\s+tại)\s+cho\s+(?:tôi|mình|em)\b"
)
ANSWER_REFERENCE_REQUEST = (
    r"\b(?:repeat|restate|recite|say)\s+(?:the\s+)?(?:correct\s+)?answer\b|"
    r"\b(?:nhắc\s+lại|lặp\s+lại|nói)\s+(?:câu\s+trả\s+lời|đáp\s*án)\b"
)
# These are concept-level routing signals, not final classifications. A broad
# request must contain both a request/delegation cue and an academic-work cue;
# the dedicated security model then decides the category semantically.
ANSWER_REQUEST_CUE = (
    r"^(?:(?:please|kindly|just)\s+)?"
    r"(?:tell|show|give|provide|write|draft|compose|solve|answer|respond|complete|"
    r"do|help|teach|walk|take\s+over)\b|"
    r"\b(?:please|kindly|just)\s+"
    r"(?:tell|show|give|provide|write|draft|compose|solve|answer|respond|complete|"
    r"do|help|teach|walk|take\s+over)\b|"
    r"\b(?:can|could|would|will)\s+you\b|"
    r"\b(?:what|how)\s+(?:should|would|could)\s+"
    r"(?:i|you|my|the|a\s+candidate)\b|"
    r"\b(?:for\s+me|on\s+my\s+behalf|as\s+if\s+you\s+were\s+me)\b|"
    r"\bi(?:'|’)m\s+stuck\b.{0,40}\btake\s+over\b|"
    r"^(?:(?:hãy|xin|vui\s+lòng)\s+)?"
    r"(?:cho|nói|viết|soạn|giải|làm|trả\s+lời|giúp)\b|"
    r"\b(?:bạn\s+)?có\s+thể\b|"
    r"\b(?:tôi|mình|em)\s+nên\b|"
    r"\b(?:hộ|giúp|cho)\s+(?:tôi|mình|em)\b"
)
ANSWER_WORK_CUE = (
    r"\b(?:answer|response|solution|question|solve|respond|work\s+out|"
    r"say|write|draft|compose|submit|submission|sample|example|exemplar|"
    r"candidate|take\s+over|do\s+(?:it|this|the\s+question))\b|"
    r"\b(?:đáp\s*án|câu\s+trả\s+lời|câu\s+hỏi|giải|làm|trả\s+lời|"
    r"nói|viết|soạn|nộp|bài\s+làm|ví\s+dụ|bài\s+mẫu)\b"
)

def protected_ask_alternation(*other_patterns: str) -> str:
    """One alternation covering every ask for protected material.

    Takes the caller's protected-topic patterns (future questions, system prompt,
    rubric, answer key) and appends ``DIRECT_ANSWER_REQUEST``, so the
    control-action recogniser can refuse an "explain this term" request that is
    really after the answer — without either module duplicating the other's
    patterns.
    """
    return "|".join((*other_patterns, DIRECT_ANSWER_REQUEST))


__all__ = [
    "ANSWER_REFERENCE_REQUEST",
    "ANSWER_REQUEST_CUE",
    "ANSWER_WORK_CUE",
    "DIRECT_ANSWER_REQUEST",
    "protected_ask_alternation",
]
