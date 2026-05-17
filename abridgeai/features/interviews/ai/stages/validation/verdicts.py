"""Verdict types for the interview VALIDATION stage (T6.6).

Mirrors the dataclass + Enum playbook used by the quiz validation
stage (T5.7) but with an interview-specific 5-criterion partition.
Each :class:`Verdict` carries the *list* of criteria the question
failed (rather than a single defect code) because three of the checks
run deterministically in Python and two come from one LLM round-trip
— so a question can fail multiple criteria at once.

A question is ``accepted`` iff ``failed_criteria`` is empty. We keep
``accepted`` as an explicit field so callers do not have to compute
it (and to leave room for non-criteria-driven rejections in future).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ValidationCriterion(str, Enum):  # noqa: UP042 -- StrEnum changes value coercion; preserve verbatim per plan §3051
    """The five quality gates each generated interview question must pass.

    The values are stable wire identifiers and feed back into
    ``ai_pipeline_runs.config_json`` for traceability — do not rename
    without a migration coordinator.
    """

    GROUNDED = "grounded"
    DIFFICULTY_COHERENT = "difficulty_coherent"
    TYPE_MATCHES_CONFIG = "type_matches_config"
    NOT_LEADING = "not_leading"
    LENGTH_REASONABLE = "length_reasonable"


@dataclass(frozen=True)
class Verdict:
    """One reviewer decision for a single generated interview question.

    Attributes
    ----------
    question_index
        0-based offset into the input ``drafts`` list. Stays parallel
        to the drafts so :func:`zip(drafts, verdicts)` always lines
        up.
    accepted
        ``True`` when the draft passes every validation criterion.
        Equal to ``not failed_criteria`` today; kept explicit to allow
        future rejections that are not tied to a criterion.
    failed_criteria
        Subset of :class:`ValidationCriterion` the draft did not pass,
        in declaration order. Empty when ``accepted`` is true.
    rationale
        Human-readable explanation aimed at teachers reviewing
        rejections. Free-form prose; never parsed by callers.
    """

    question_index: int
    accepted: bool
    failed_criteria: list[ValidationCriterion] = field(default_factory=list)
    rationale: str = ""


__all__ = ["ValidationCriterion", "Verdict"]
