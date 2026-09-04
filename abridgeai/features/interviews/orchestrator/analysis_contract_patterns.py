"""Phase 1.2 patterns: the interviewer's internal analysis contract.

Split out of ``security_logic`` to keep that module under the orchestrator's
800-line "no god files" ceiling. Pattern data only — no I/O, no model calls.

The analysis stage fills these fields while judging an answer. A candidate naming
one is reaching past the conversation into the grading structures, which is why
matching the bare identifier is safe: someone answering a question says
"evidence", not ``expected_evidence``; "score", not ``provisional_score``.

Two verdicts, deliberately distinguished so the audit row says what was actually
attempted:

* writing to a field, or declaring every outcome covered → grading manipulation
* reading a field, or the hidden outcome set → rubric exfiltration
"""

from __future__ import annotations

# Internal field names from the analysis contract. snake_case on purpose — these
# are identifiers, not vocabulary a candidate would use in prose.
ANALYSIS_FIELD = (
    r"expected_evidence|provisional_score|evidence_type|outcome_coverage|"
    r"candidate_question_scores|coverage_delta|rubric_weights?"
)

# Verbs that would WRITE a verdict rather than ask about one.
ANALYSIS_WRITE = (
    r"\b(?:mark|set|record|assign|award|force|make|update|override|"
    r"đánh\s*dấu|đặt|ghi\s*nhận|cho|ghi\s+là)\b"
)

# "all learning outcomes" / "every outcome" — the blanket-claim shape.
ANALYSIS_ALL_OUTCOMES = (
    r"\b(?:all|every|each)\s+(?:the\s+)?(?:learning\s+)?outcomes?\b|"
    r"\bfor\s+(?:all|every|each)\s+outcomes?\b|"
    r"(?:tất\s*cả|mọi|toàn\s*bộ)\s+(?:các\s+)?chuẩn\s*đầu\s*ra"
)

# The state a manipulation attempt wants those outcomes moved to. Paired with
# ANALYSIS_WRITE *and* ANALYSIS_ALL_OUTCOMES so an ordinary sentence like "this
# outcome is met because the index stays balanced" cannot trip it.
ANALYSIS_COVERED = (
    r"\b(?:covered|complete|met|satisfied|passed?)\b|(?:đã\s*)?(?:đạt|hoàn\s*thành|bao\s*phủ)"
)

# "which other outcomes are you secretly scoring me on" — reading the hidden set.
ANALYSIS_HIDDEN_OUTCOMES = (
    r"(?:other|remaining|hidden|secret(?:ly)?)\s+(?:learning\s+)?outcomes?|"
    r"outcomes?\s+(?:are\s+)?you\s+(?:secretly\s+)?(?:scoring|grading|tracking)|"
    r"chuẩn\s*đầu\s*ra\s+(?:khác|còn\s*lại|ẩn)"
)

__all__ = [
    "ANALYSIS_ALL_OUTCOMES",
    "ANALYSIS_COVERED",
    "ANALYSIS_FIELD",
    "ANALYSIS_HIDDEN_OUTCOMES",
    "ANALYSIS_WRITE",
]
