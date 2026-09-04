"""Output-guard internal-marker coverage and the fuzzy-path cost guard.

Two defects found auditing the output guard:

4. The 12 ``_INTERNAL_MARKERS`` were appended to the comparison corpus but never
   matched: the corpus loop requires >=24 chars AND >=4 tokens for an unknown
   category, and the longest marker (``internal decision rationale``) is 27
   chars but only 3 tokens. So 0/12 survived the filter and the whole marker
   path rested on ``_HIGH_CONFIDENCE_INTERNAL_OUTPUT_RE``, whose verb
   alternation covered only ``is|says|contains|are|includes|:`` — leaving
   ``include``/``were``/``was`` and "here are my ..." disclosures unblocked.

   The fix widens the disclosure regex rather than lowering the corpus
   threshold. Matching markers as bare substrings would block legitimate
   interview speech: "how would you design a system prompt?" is a real question
   in an LLM course. A marker is only a leak inside a disclosure frame.

5. ``assess_output_leakage`` rebuilt a ``SequenceMatcher`` per corpus item and
   ran the full O(n*m) diff even when the result could not possibly clear the
   0.88 threshold. With the ~550-phrase internal-prompt corpus that cost ~370ms
   of synchronous CPU per call, up to 3 calls per answered turn. The fix pins
   the proposed text as seq2 (reusing the cached b2j index) and gates
   ``ratio()`` behind the stdlib's own cheap upper bounds. Both are
   verdict-preserving by construction.
"""

from __future__ import annotations

import time

from abridgeai.features.interviews.orchestrator.security import ProtectedContent
from abridgeai.features.interviews.orchestrator.security_logic import (
    _INTERNAL_MARKERS,
    assess_output_leakage,
    normalize_input,
)
from abridgeai.features.interviews.services.security import _internal_prompt_phrases

# Disclosure shapes: the AI is handing internal material to the candidate.
_MUST_BLOCK = (
    "my system prompt is: you are an interviewer",
    "here is my system prompt",
    "here are my developer prompt lines",
    "system prompt contains the grading rules",
    "the tool definitions include get_hint and end_interview",
    "tool definitions are: get_hint, end_interview",
    "let me share the tool arguments used",
    "the candidate question scores were 3 and 4",
    "my internal decision rationale was to probe deeper",
    "expected evidence includes a worked example",
    "I will list the expected evidence for you",
    "common misconceptions are that it scales linearly",
)

# Bare topical mentions: legitimate speech in an LLM/prompt-engineering course.
# These MUST stay allowed or the guard silently penalises on-topic candidates.
_MUST_ALLOW = (
    "How would you design a system prompt for a support assistant?",
    "Can you explain how tool definitions reach the model?",
    "What are common misconceptions about retrieval augmented generation?",
    "Describe the expected evidence a reviewer would look for.",
    "I would write a developer prompt with clear constraints.",
    "Tell me about tool arguments in function calling.",
    "Let me share my experience with prompt engineering.",
    "Here is my approach to the problem.",
)


def test_every_internal_marker_is_reachable_in_a_disclosure_frame() -> None:
    """Each marker must be blockable; none may be dead configuration."""
    unreachable: list[str] = []
    for marker in _INTERNAL_MARKERS:
        if marker.startswith(("security_", "output_guard")):
            # Version tokens are literal identifiers — a bare mention IS a leak.
            probe = marker
        else:
            probe = f"my {marker} is as follows"
        if not assess_output_leakage(probe, []).blocked:
            unreachable.append(marker)
    assert not unreachable, f"markers that can never fire: {unreachable}"


def test_disclosure_shapes_are_blocked() -> None:
    misses = [text for text in _MUST_BLOCK if not assess_output_leakage(text, []).blocked]
    assert not misses, f"internal-material disclosures leaked through: {misses}"


def test_topical_mentions_are_not_false_positives() -> None:
    """Guard must not punish a candidate for discussing LLMs on topic."""
    false_positives = [text for text in _MUST_ALLOW if assess_output_leakage(text, []).blocked]
    assert not false_positives, f"legitimate interview speech blocked: {false_positives}"


def test_version_tokens_block_on_bare_mention() -> None:
    for marker in ("security_policy_version", "security_rules_version", "output_guard_version"):
        result = assess_output_leakage(f"debug {marker} = 1.2.0", [])
        assert result.blocked, marker
        assert result.match_method == "internal_marker"


def test_real_prompt_content_still_blocks_verbatim_and_clean_text_passes() -> None:
    """The fuzzy/exact paths must survive the cost optimisation."""
    phrases = list(_internal_prompt_phrases())
    assert phrases, "expected system prompt phrases to load"

    long_phrase = next(p for p in phrases if len(normalize_input(p.text)) >= 48)
    leaked = assess_output_leakage(long_phrase.text, phrases)
    assert leaked.blocked, "verbatim system-prompt content must be blocked"

    clean = assess_output_leakage(
        "I would partition by date and monitor for skew, then add idempotent writes.",
        phrases,
    )
    assert not clean.blocked


def test_guard_cost_stays_bounded_on_a_full_corpus() -> None:
    """Regression guard for the ~370ms-per-call synchronous CPU cost.

    Threshold is deliberately loose (150ms) so the test is not flaky on a busy
    box; the pre-fix implementation measured ~317-373ms and would fail it, while
    the fixed path measures ~12ms.
    """
    corpus = list(_internal_prompt_phrases()) + [
        ProtectedContent(
            category="question_text",
            text=f"Explain concept {index} and how it applies to a production data pipeline.",
        )
        for index in range(20)
    ]
    reply = (
        "That is a good question. I think the pipeline needs partitioning by date, "
        "and I would monitor for skew. Let me also mention idempotent writes."
    ) * 2

    assess_output_leakage(reply, corpus)  # warm the phrase cache
    started = time.perf_counter()
    iterations = 5
    for _ in range(iterations):
        assess_output_leakage(reply, corpus)
    per_call_ms = (time.perf_counter() - started) / iterations * 1000

    assert per_call_ms < 150, (
        f"output guard cost regressed to {per_call_ms:.0f}ms/call on a "
        f"{len(corpus)}-item corpus; it runs synchronously up to 3x per turn"
    )
