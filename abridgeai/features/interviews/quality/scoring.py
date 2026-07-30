"""Scoring + aggregation for the LLM-judged quality metrics.

Pure: no LLM, no DB. The judges' network calls live in :mod:`judges`; this
module owns the shapes and the aggregation rules so they can be unit-tested
without touching a gateway.

Design decisions worth stating, both learned from reading SparkMe's harness:

* **A judge that fails to parse is EXCLUDED, not scored 1.** SparkMe's
  ``eval_flow`` assigns ``score=1`` on a parse error, which silently mixes
  infrastructure failure into the quality signal and biases every average
  downward. Here an unparseable verdict is recorded in ``errors`` and left out of
  the mean, so a broken judge looks like missing data instead of a bad
  interviewer.
* **"No follow-ups at all" is a finding, not missing data.** An interviewer that
  only reads the question bank never produces a contingent probe. Reporting that
  as ``None`` would hide the exact failure mode the metric exists to catch, so it
  is surfaced explicitly via :attr:`ContingencyReport.no_followups`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Scores below these are worth a human's attention. Not enforcement — these
# metrics never gate a grade or a deployment; they are diagnostics.
CONTINGENCY_CONCERN_BELOW = 3.0
LEADING_CONCERN_BELOW = 3.0


@dataclass(frozen=True)
class JudgedTurn:
    """One judged interviewer utterance."""

    message_id: str
    score: int
    explanation: str = ""
    # Contingency: what the follow-up picked up from the answer.
    grounded_in: str = ""
    # Contingency: the follow-up asserted something the student never said.
    fabricated_premise: bool = False
    # Leading: the content the question handed to the student.
    leaked_content: str = ""
    # Leading: the student's reply mostly echoed the question back.
    answer_echoes_question: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "message_id": self.message_id,
            "score": self.score,
            "explanation": self.explanation,
            "grounded_in": self.grounded_in,
            "fabricated_premise": self.fabricated_premise,
            "leaked_content": self.leaked_content,
            "answer_echoes_question": self.answer_echoes_question,
        }


def _mean(values: list[int]) -> float | None:
    return (sum(values) / len(values)) if values else None


@dataclass(frozen=True)
class ContingencyReport:
    """Aggregate contingency over a session's follow-up questions."""

    session_id: str
    turns: list[JudgedTurn] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    # True when the interview produced no follow-ups at all to judge.
    no_followups: bool = False

    @property
    def mean_score(self) -> float | None:
        return _mean([t.score for t in self.turns])

    @property
    def weak_turns(self) -> list[JudgedTurn]:
        return [t for t in self.turns if t.score < CONTINGENCY_CONCERN_BELOW]

    @property
    def fabricated_premises(self) -> list[JudgedTurn]:
        """Follow-ups that invented something the student never said."""
        return [t for t in self.turns if t.fabricated_premise]

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": "contingency",
            "session_id": self.session_id,
            "no_followups": self.no_followups,
            "judged": len(self.turns),
            "mean_score": self.mean_score,
            "weak_count": len(self.weak_turns),
            "fabricated_premise_count": len(self.fabricated_premises),
            "errors": list(self.errors),
            "turns": [t.to_dict() for t in self.turns],
        }


@dataclass(frozen=True)
class LeadingReport:
    """Aggregate leading-question cleanliness over a session."""

    session_id: str
    turns: list[JudgedTurn] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def mean_score(self) -> float | None:
        return _mean([t.score for t in self.turns])

    @property
    def leading_turns(self) -> list[JudgedTurn]:
        """Questions clean enough to worry about — the actionable list."""
        return [t for t in self.turns if t.score < LEADING_CONCERN_BELOW]

    @property
    def contaminated_turns(self) -> list[JudgedTurn]:
        """Worst case: the question leaked content AND the answer echoed it."""
        return [t for t in self.turns if t.answer_echoes_question]

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": "leading_questions",
            "session_id": self.session_id,
            "judged": len(self.turns),
            "mean_score": self.mean_score,
            "leading_count": len(self.leading_turns),
            "contaminated_count": len(self.contaminated_turns),
            "errors": list(self.errors),
            "turns": [t.to_dict() for t in self.turns],
        }


def parse_verdict(
    payload: dict[str, Any] | None, *, message_id: str
) -> JudgedTurn | None:
    """Coerce one judge response into a :class:`JudgedTurn`.

    Returns ``None`` when the payload is unusable (missing/!int/out-of-range
    score) so the caller can record an error and EXCLUDE the turn rather than
    scoring it 1 — see the module docstring for why that matters.
    """
    if not isinstance(payload, dict):
        return None
    raw = payload.get("score")
    if isinstance(raw, bool):  # bool is an int subclass; never a score
        return None
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw.isdigit():
            return None
        raw = int(raw)
    if not isinstance(raw, int):
        if isinstance(raw, float) and raw.is_integer():
            raw = int(raw)
        else:
            return None
    if not 1 <= raw <= 5:
        return None
    return JudgedTurn(
        message_id=message_id,
        score=raw,
        explanation=_text(payload.get("explanation")),
        grounded_in=_text(payload.get("grounded_in")),
        fabricated_premise=bool(payload.get("fabricated_premise", False)),
        leaked_content=_text(payload.get("leaked_content")),
        answer_echoes_question=bool(payload.get("answer_echoes_question", False)),
    )


def _text(value: object, *, limit: int = 500) -> str:
    return str(value).strip()[:limit] if isinstance(value, str) else ""


__all__ = [
    "CONTINGENCY_CONCERN_BELOW",
    "LEADING_CONCERN_BELOW",
    "ContingencyReport",
    "JudgedTurn",
    "LeadingReport",
    "parse_verdict",
]
