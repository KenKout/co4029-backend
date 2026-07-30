"""Quarantined claim extraction — pure types + parser.

This is the *untrusted-side* half of answer analysis. The stage that renders
these types (:mod:`orchestrator.extraction_logic`) is the only runtime LLM call
that sees a student's raw answer alongside nothing else worth stealing: no
rubric text, no outcome list, no expected evidence, no model answer. It reduces
the answer to a small set of bounded claims, and :mod:`orchestrator.matching`
maps those claims onto the rubric without ever seeing the raw text.

Why the split exists
--------------------
Previously one call held the raw answer *and* the full rubric surface, so a
prompt injection that survived the security guard sat in the same context as
the thing worth exfiltrating or manipulating. Separating them means an attack
can only travel forward as a claim string that has passed a schema with hard
caps and a deterministic rules screen (:mod:`orchestrator.claim_filter`).

This narrows the channel; it does not close it. A claim is still
attacker-influenced text. The honest guarantee is "bounded and screened",
not "clean" — see the module docstring of ``claim_filter`` for what that
screening can and cannot catch.

Design mirrors :mod:`orchestrator.analysis`: pure types + a permissive parser
here, I/O in the ``_logic`` sibling, so this module is trivially unit-testable.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TypeVar

from abridgeai.features.interviews.orchestrator.analysis import (
    Completeness,
    Relevance,
    Specificity,
)
from abridgeai.features.interviews.orchestrator.intent import (
    IntentClassification,
    StudentIntent,
    parse_intent_response,
)

_E = TypeVar("_E", bound=Enum)

# Hard caps. Enforced at parse time and re-enforced by the claim filter, so a
# model that ignores the prompt's limits still cannot widen the channel between
# the untrusted answer and the rubric-bearing stage.
MAX_CLAIMS = 6
MAX_CLAIM_CHARS = 200


class ClaimKind(str, Enum):  # noqa: UP042 -- match codebase convention
    """What kind of move the student made. Coarse on purpose.

    Finer taxonomies invite the extractor to editorialise; the matcher only
    needs enough structure to weigh a definition differently from an aside.
    """

    ASSERTION = "assertion"
    DEFINITION = "definition"
    EXAMPLE = "example"
    REASONING = "reasoning"
    QUALIFIER = "qualifier"


@dataclass(frozen=True)
class Claim:
    """One atomic thing the student said, restated plainly.

    ``text`` is a paraphrase, not a quote — the prompt asks for subject-matter
    content in the extractor's own words, which is what keeps verbatim
    instruction text from riding through. Length is capped at parse time.
    """

    text: str
    kind: ClaimKind = ClaimKind.ASSERTION


@dataclass
class AnswerClaims:
    """Bounded projection of a student turn, safe(r) to pair with the rubric.

    ``intent`` is produced here too: intent classification needs exactly the
    same inputs (question + utterance) and no protected content, so folding it
    in keeps the per-turn LLM call count unchanged by the split. The value is
    still parsed by :func:`parse_intent_response` and still overridden by the
    caller's deterministic rules, so intent semantics are unchanged.
    """

    intent: IntentClassification
    claims: list[Claim] = field(default_factory=list)
    relevance: Relevance = Relevance.PARTIALLY_RELEVANT
    completeness: Completeness = Completeness.PARTIAL
    specificity: Specificity = Specificity.GENERAL
    #: Did the candidate catch and fix their own error *within this turn*?
    #: Only observable in the raw text, so the matcher cannot recover it from
    #: claims — the extractor is the authority and the matcher's value for this
    #: field is overridden by it.
    self_corrected: bool = False
    #: Claims removed by the deterministic filter. Recorded for the shadow
    #: metric; a non-zero value means an attack was caught at the boundary.
    dropped_claim_count: int = 0
    source: str = "llm"


def fallback_claims(
    intent: IntentClassification | None = None,
) -> AnswerClaims:
    """Safe verdict when extraction is unavailable.

    No claims means the matcher gets nothing to attribute, which yields a
    not-assessable analysis downstream. That is the correct failure direction:
    a broken extractor must never manufacture evidence for an outcome.
    """
    return AnswerClaims(
        intent=intent
        or IntentClassification(
            intent=StudentIntent.ANSWER,
            confidence=0.3,
            rationale="Extractor unavailable; defaulting to answer.",
            source="fallback",
        ),
        claims=[],
        relevance=Relevance.PARTIALLY_RELEVANT,
        completeness=Completeness.INSUFFICIENT,
        specificity=Specificity.VAGUE,
        source="fallback",
    )


def parse_answer_claims(payload: Mapping[str, Any] | None) -> AnswerClaims | None:
    """Coerce gateway JSON into :class:`AnswerClaims`.

    Contract::

        {"intent": "<StudentIntent>", "confidence": 0.0-1.0, "rationale": "...",
         "relevance": "...", "completeness": "...", "specificity": "...",
         "claims": [{"text": "...", "kind": "assertion"}]}

    Returns ``None`` when the payload is unusable so the caller applies its own
    deterministic fallback rather than trusting a guessed structure — same
    contract as :func:`parse_intent_response`.
    """
    if not isinstance(payload, Mapping):
        return None
    intent = parse_intent_response(payload)
    if intent is None:
        return None
    return AnswerClaims(
        intent=intent,
        claims=_parse_claims(payload.get("claims")),
        relevance=_enum(Relevance, payload.get("relevance"), Relevance.OFF_TOPIC),
        completeness=_enum(Completeness, payload.get("completeness"), Completeness.INSUFFICIENT),
        specificity=_enum(Specificity, payload.get("specificity"), Specificity.VAGUE),
        self_corrected=payload.get("self_corrected") is True,
        source="llm",
    )


def _enum(enum_cls: type[_E], value: object, default: _E) -> _E:  # noqa: UP047 -- match analysis.py
    """Unknown values fall back to the safest bucket rather than raising.

    Local copy of the identical helper in :mod:`orchestrator.analysis`; kept
    here so this module does not reach into another module's private surface.
    """
    try:
        return enum_cls(str(value).strip().lower())
    except (ValueError, TypeError, AttributeError):
        return default


def _parse_claims(raw: object) -> list[Claim]:
    """Take at most :data:`MAX_CLAIMS` well-formed claims, truncating each.

    Silently drops malformed rows rather than raising: a partially usable
    extraction is still better evidence than none, and the caller has no way to
    repair the payload.
    """
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return []
    out: list[Claim] = []
    for item in raw:
        if len(out) >= MAX_CLAIMS:
            break
        if not isinstance(item, Mapping):
            continue
        text = item.get("text")
        if not isinstance(text, str):
            continue
        cleaned = " ".join(text.split())[:MAX_CLAIM_CHARS].strip()
        if not cleaned:
            continue
        out.append(
            Claim(
                text=cleaned,
                kind=_enum(ClaimKind, item.get("kind"), ClaimKind.ASSERTION),
            )
        )
    return out


__all__ = [
    "MAX_CLAIMS",
    "MAX_CLAIM_CHARS",
    "AnswerClaims",
    "Claim",
    "ClaimKind",
    "fallback_claims",
    "parse_answer_claims",
]
