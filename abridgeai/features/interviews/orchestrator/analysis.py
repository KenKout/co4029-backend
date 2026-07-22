"""Structured answer analysis (Phase 3).

For utterances classified as an actual answer, the orchestrator analyzes the
response against the current question, its linked outcome, expected evidence,
and misconceptions — producing a structured :class:`AnswerAnalysis` used to
(a) update provisional outcome coverage and (b) pick the next interviewer
action (probe for an example, challenge reasoning, resolve a contradiction…).

Design (mirrors follow-up / intent conventions):
* Pure types + permissive parser here; the LLM call lives in
  :mod:`orchestrator.analysis_logic` so this stays I/O-free and unit-testable.
* Strict-but-tolerant parsing: unknown enum values fall back to the safest
  bucket (``not_assessable`` / ``off_topic`` / ``none``) rather than raising,
  so a malformed model response degrades gracefully instead of 500ing the
  student's turn.
* Evidence is provisional ONLY. The post-session evaluator re-judges the
  transcript independently and is never bound by these values.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, TypeVar

_E = TypeVar("_E", bound=Enum)


class Relevance(str, Enum):  # noqa: UP042 -- match codebase convention
    RELEVANT = "relevant"
    PARTIALLY_RELEVANT = "partially_relevant"
    OFF_TOPIC = "off_topic"


class Completeness(str, Enum):  # noqa: UP042 -- match codebase convention
    INSUFFICIENT = "insufficient"
    PARTIAL = "partial"
    COMPLETE = "complete"


class Correctness(str, Enum):  # noqa: UP042 -- match codebase convention
    INCORRECT = "incorrect"
    MIXED = "mixed"
    MOSTLY_CORRECT = "mostly_correct"
    CORRECT = "correct"
    NOT_ASSESSABLE = "not_assessable"


class Specificity(str, Enum):  # noqa: UP042 -- match codebase convention
    VAGUE = "vague"
    GENERAL = "general"
    SPECIFIC = "specific"


class ProbeType(str, Enum):  # noqa: UP042 -- match codebase convention
    NONE = "none"
    CLARIFICATION = "clarification"
    ASK_FOR_EXAMPLE = "ask_for_example"
    PROBE_REASONING = "probe_reasoning"
    CHALLENGE_ASSUMPTION = "challenge_assumption"
    EXPLORE_TRADEOFF = "explore_tradeoff"
    RESOLVE_CONTRADICTION = "resolve_contradiction"
    # Depth probes (Slice 8, v2): dig into a STRONG answer to find the
    # candidate's ceiling instead of advancing. EXTEND_STRONG asks them to
    # generalize/extend; PROBE_EDGE_CASE pushes on boundaries/failure modes.
    EXTEND_STRONG = "extend_strong"
    PROBE_EDGE_CASE = "probe_edge_case"


class EvidenceType(str, Enum):  # noqa: UP042 -- match codebase convention
    SUPPORTS = "supports"
    PARTIALLY_SUPPORTS = "partially_supports"
    CONTRADICTS = "contradicts"
    INSUFFICIENT = "insufficient"


@dataclass
class OutcomeEvidence:
    """A single piece of provisional evidence tied to a learning outcome.

    ``turn_id`` MUST reference the originating student turn (message id) so the
    final evaluator can trace and independently re-verify the evidence.
    """

    outcome_id: str
    turn_id: str
    evidence_type: EvidenceType = EvidenceType.INSUFFICIENT
    criterion_id: str | None = None
    summary: str = ""
    transcript_excerpt: str | None = None
    provisional_score: float | None = None
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["evidence_type"] = self.evidence_type.value
        return d

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, default_turn_id: str) -> OutcomeEvidence:
        return cls(
            outcome_id=str(data.get("outcome_id", "")),
            turn_id=str(data.get("turn_id") or default_turn_id),
            evidence_type=_enum(EvidenceType, data.get("evidence_type"), EvidenceType.INSUFFICIENT),
            criterion_id=_opt_str(data.get("criterion_id")),
            summary=_str(data.get("summary"), limit=1000),
            transcript_excerpt=_opt_str(data.get("transcript_excerpt"), limit=1000),
            provisional_score=_opt_float(data.get("provisional_score")),
            confidence=_confidence(data.get("confidence")),
        )


@dataclass
class AnswerAnalysis:
    relevance: Relevance = Relevance.OFF_TOPIC
    completeness: Completeness = Completeness.INSUFFICIENT
    correctness: Correctness = Correctness.NOT_ASSESSABLE
    specificity: Specificity = Specificity.VAGUE
    has_concrete_example: bool = False

    identified_concepts: list[str] = field(default_factory=list)
    missing_concepts: list[str] = field(default_factory=list)
    misconceptions: list[str] = field(default_factory=list)
    contradictions: list[str] = field(default_factory=list)

    evidence: list[OutcomeEvidence] = field(default_factory=list)

    recommended_probe_type: ProbeType = ProbeType.NONE
    provisional_quality_score: float | None = None
    confidence: float = 0.0

    # Self-correction (Slice 15, v2). True when the candidate noticed and fixed
    # their OWN mistake within the answer ("actually, it's X, not Y"). A positive
    # signal: it earns a POSITIVE acknowledgement and suppresses a contradiction
    # probe pointing at what they already resolved. Defaults False.
    self_corrected: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "relevance": self.relevance.value,
            "completeness": self.completeness.value,
            "correctness": self.correctness.value,
            "specificity": self.specificity.value,
            "has_concrete_example": self.has_concrete_example,
            "identified_concepts": list(self.identified_concepts),
            "missing_concepts": list(self.missing_concepts),
            "misconceptions": list(self.misconceptions),
            "contradictions": list(self.contradictions),
            "evidence": [e.to_dict() for e in self.evidence],
            "recommended_probe_type": self.recommended_probe_type.value,
            "provisional_quality_score": self.provisional_quality_score,
            "confidence": self.confidence,
            "self_corrected": self.self_corrected,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None, *, default_turn_id: str) -> AnswerAnalysis:
        data = data or {}
        evidence_raw = data.get("evidence", []) or []
        evidence = [
            OutcomeEvidence.from_dict(e, default_turn_id=default_turn_id)
            for e in evidence_raw
            if isinstance(e, Mapping)
        ]
        return cls(
            relevance=_enum(Relevance, data.get("relevance"), Relevance.OFF_TOPIC),
            completeness=_enum(Completeness, data.get("completeness"), Completeness.INSUFFICIENT),
            correctness=_enum(Correctness, data.get("correctness"), Correctness.NOT_ASSESSABLE),
            specificity=_enum(Specificity, data.get("specificity"), Specificity.VAGUE),
            has_concrete_example=bool(data.get("has_concrete_example", False)),
            identified_concepts=_str_list(data.get("identified_concepts")),
            missing_concepts=_str_list(data.get("missing_concepts")),
            misconceptions=_str_list(data.get("misconceptions")),
            contradictions=_str_list(data.get("contradictions")),
            evidence=evidence,
            recommended_probe_type=_enum(
                ProbeType, data.get("recommended_probe_type"), ProbeType.NONE
            ),
            provisional_quality_score=_opt_float(data.get("provisional_quality_score")),
            confidence=_confidence(data.get("confidence")),
            self_corrected=bool(data.get("self_corrected", False)),
        )


def parse_analysis_response(
    payload: Mapping[str, Any] | None, *, default_turn_id: str
) -> AnswerAnalysis | None:
    """Coerce the gateway JSON into an :class:`AnswerAnalysis`.

    Returns None when the payload is not a mapping at all, so the caller can
    apply a deterministic fallback analysis rather than a fabricated one. Any
    mapping (even sparse) yields a valid, safely-defaulted analysis.
    """
    if not isinstance(payload, Mapping):
        return None
    return AnswerAnalysis.from_dict(payload, default_turn_id=default_turn_id)


def fallback_analysis(default_turn_id: str) -> AnswerAnalysis:
    """Safe analysis used when the LLM is unavailable or returns garbage.

    Marks the answer not-assessable with no evidence and no probe recommended,
    so the orchestrator advances neutrally instead of inventing a score. The
    turn is still recorded; the final evaluator will judge it from the
    transcript independently.
    """
    del default_turn_id  # no evidence attached in the fallback
    return AnswerAnalysis(
        relevance=Relevance.PARTIALLY_RELEVANT,
        completeness=Completeness.PARTIAL,
        correctness=Correctness.NOT_ASSESSABLE,
        specificity=Specificity.GENERAL,
        recommended_probe_type=ProbeType.NONE,
        confidence=0.0,
    )


# ── coercion helpers ─────────────────────────────────────────────────────────


def _enum(enum_cls: type[_E], value: object, default: _E) -> _E:  # noqa: UP047 -- PEP 695 syntax needs py3.12; target is 3.11
    try:
        return enum_cls(str(value).strip().lower())
    except (ValueError, TypeError, AttributeError):
        return default


def _str(value: object, *, limit: int = 500) -> str:
    if isinstance(value, str):
        return value.strip()[:limit]
    return ""


def _opt_str(value: object, *, limit: int = 500) -> str | None:
    if isinstance(value, str):
        cleaned = value.strip()
        if cleaned:
            return cleaned[:limit]
    return None


def _str_list(value: object) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        return []
    out: list[str] = []
    for item in value:
        if isinstance(item, str):
            cleaned = item.strip()
            if cleaned:
                out.append(cleaned[:300])
    return out


def _opt_float(value: object) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _confidence(value: object) -> float:
    f = _opt_float(value)
    if f is None:
        return 0.0
    return max(0.0, min(1.0, f))


__all__ = [
    "AnswerAnalysis",
    "Completeness",
    "Correctness",
    "EvidenceType",
    "OutcomeEvidence",
    "ProbeType",
    "Relevance",
    "Specificity",
    "fallback_analysis",
    "parse_analysis_response",
]
