"""Internal-material markers and the disclosure-frame regex for the output guard.

Split out of ``security_logic`` to keep that module under the orchestrator's
800-line "no god files" ceiling. Pattern data only — no I/O, no DB, no model
calls — so it stays trivially reviewable: these literals decide what counts as
the assistant handing internal material to a candidate.
"""

from __future__ import annotations

import re

# Protected concepts that must never be disclosed to a candidate. Compared as
# whole phrases inside a disclosure frame (see the regex below), NOT as bare
# substrings — see ``_DISCLOSURE_ONLY_RATIONALE``.
INTERNAL_MARKERS = (
    "system prompt",
    "developer prompt",
    "tool definitions",
    "tool arguments",
    "candidate question scores",
    "internal decision rationale",
    "expected evidence",
    "common misconceptions",
    "security_policy_version",
    "security_rules_version",
    "security_prompt_version",
    "output_guard_version",
)

_DISCLOSURE_ONLY_RATIONALE = """
A marker is only a LEAK when the utterance actually discloses it. A bare mention
is legitimate interview speech — "how would you design a system prompt?" is a
real question in an LLM course, and "what are common misconceptions about RAG?"
is a real answer. Matching these as bare substrings (or lowering the corpus
length threshold in ``assess_output_leakage`` so they join the substring loop)
would penalise candidates for being on topic. So the corpus thresholds stay put
and this regex carries the entire marker path.
"""

# Verbs/punctuation that turn a mention into a disclosure.
_DISCLOSE_VERB = r"(?:is|are|was|were|says?|contains?|includes?|included|:)"

# Lead-ins that disclose without a following verb ("here are my tool definitions").
_DISCLOSE_LEAD = (
    r"(?:here\s+(?:is|are)|let\s+me\s+(?:share|list|show|give)|"
    r"i(?:'|’)?(?:ll|\s+will)?\s*(?:share|list|show|give))"
)

_INTERNAL_CONCEPTS = (
    r"(?:(?:my|the|our|your)\s+)?(?:system|developer)\s+prompts?|"
    r"tool\s+(?:definitions?|arguments?)|candidate\s+question\s+scores?|"
    r"internal\s+decision\s+rationale|expected\s+evidence|common\s+misconceptions"
)

HIGH_CONFIDENCE_INTERNAL_OUTPUT_RE = re.compile(
    # <concept> <disclosure verb> ...
    rf"(?:{_INTERNAL_CONCEPTS})\s+{_DISCLOSE_VERB}|"
    # "here are my tool definitions" / "let me share the expected evidence"
    rf"{_DISCLOSE_LEAD}\s+(?:the\s+|my\s+|our\s+|your\s+)?(?:{_INTERNAL_CONCEPTS})|"
    r"(?:rubric|scoring|grading|outcome).{0,30}(?:weights?|threshold).{0,20}(?:is|are|:|\d)|"
    # Version identifiers are internal tokens: a bare mention IS the leak.
    r"security_(?:policy|rules|prompt)_version|output_guard_version",
    re.IGNORECASE,
)

__all__ = [
    "HIGH_CONFIDENCE_INTERNAL_OUTPUT_RE",
    "INTERNAL_MARKERS",
]
