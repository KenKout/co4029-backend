"""Declarative criterion catalogs per scenario capability.

A ``Criterion`` is a (id, description) pair the judge LLM scores 1-5.
No execution lives here — ``judge.py`` consumes these and renders prompts.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Criterion:
    id: str
    description: str


quiz_criteria: tuple[Criterion, ...] = (
    Criterion(
        id="groundedness",
        description=(
            "The question and its answer are factually supported by the source "
            "material. No hallucinated claims, no fabricated numbers, no quotes "
            "the source did not contain."
        ),
    ),
    Criterion(
        id="answerability",
        description=(
            "The question has a single unambiguous correct answer derivable "
            "from the source material. It is not vague, multi-part without "
            "structure, or dependent on outside knowledge."
        ),
    ),
    Criterion(
        id="difficulty_alignment",
        description=(
            "The cognitive demand matches the requested difficulty (e.g. "
            "recall for 'easy', application/analysis for 'medium', synthesis "
            "for 'hard'). A trivia question labeled 'hard' scores low."
        ),
    ),
    Criterion(
        id="distractor_quality",
        description=(
            "For multiple-choice items, distractors are plausible (a learner "
            "could plausibly pick them), mutually exclusive, and roughly "
            "parallel in length and style. Trivially-wrong distractors score low."
        ),
    ),
)


interview_criteria: tuple[Criterion, ...] = (
    Criterion(
        id="groundedness",
        description=(
            "Each question is anchored in the source material; no questions "
            "depend on knowledge outside the provided context."
        ),
    ),
    Criterion(
        id="open_endedness",
        description=(
            "Questions invite explanation, reasoning, or comparison — not "
            "yes/no answers or single-fact recall."
        ),
    ),
    Criterion(
        id="type_mix",
        description=(
            "The set as a whole covers a healthy mix of conceptual, "
            "applied, and reflective question types rather than repeating "
            "one mode."
        ),
    ),
    Criterion(
        id="difficulty_progression",
        description=(
            "Questions progress from accessible warm-ups to deeper "
            "synthesis, rather than being uniformly easy or uniformly hard."
        ),
    ),
)


gap_report_criteria: tuple[Criterion, ...] = (
    Criterion(
        id="actionability",
        description=(
            "Each identified gap is paired with a concrete next step the "
            "learner can take. Vague advice ('study more') scores low."
        ),
    ),
    Criterion(
        id="resource_relevance",
        description=(
            "Linked resources are directly tied to the named gap, not "
            "tangential. A resource on linear algebra for a calculus gap "
            "scores low."
        ),
    ),
    Criterion(
        id="diagnostic_accuracy",
        description=(
            "Per-criterion claims about what the learner missed are "
            "consistent with the underlying session data. The report does "
            "not invent gaps that the data does not support."
        ),
    ),
)


_BY_CAPABILITY: dict[str, tuple[Criterion, ...]] = {
    "quiz_generation": quiz_criteria,
    "interview_generation": interview_criteria,
    "gap_report": gap_report_criteria,
}


def criteria_for_capability(capability: str) -> tuple[Criterion, ...]:
    try:
        return _BY_CAPABILITY[capability]
    except KeyError as exc:
        raise ValueError(
            f"unknown scenario capability: {capability!r}; expected one of {sorted(_BY_CAPABILITY)}"
        ) from exc
