"""Draft dataclass for the interview GENERATION stage (T6.5).

Lifecycle: the generation stage produces :class:`InterviewQuestionDraft`
instances from raw LLM JSON (see :mod:`.parsers`); they are validated
(T6.6) before persistence. A runtime-cheap dataclass is sufficient because
no ORM-bound model exists for drafts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal
from uuid import UUID

InterviewQuestionType = Literal[
    "technical", "behavioral", "situational", "system_design"
]
InterviewDifficulty = Literal["easy", "medium", "hard"]

_VALID_TYPES: frozenset[str] = frozenset(
    {"technical", "behavioral", "situational", "system_design"}
)
_VALID_DIFFICULTIES: frozenset[str] = frozenset({"easy", "medium", "hard"})


@dataclass
class InterviewQuestionDraft:
    """Pre-validation draft of a single generated interview question.

    Attributes
    ----------
    question_type
        Exactly one of ``technical`` / ``behavioral`` / ``situational``.
    prompt_text
        The stem the learner will read. Stands alone (no slide refs).
    difficulty
        One of ``easy`` / ``medium`` / ``hard``. Mapped to the ORM-side
        ``junior`` / ``mid_level`` / ``senior`` set during persistence.
    expected_depth
        Integer 1-5 describing how long an adequate answer should be.
        1 = one-sentence; 5 = multi-paragraph reasoning.
    linked_outcome_id
        UUID of the :class:`InterviewOutcome` this question assesses.
        ``None`` is allowed when no outcomes were provided to the stage.
    logical_question_index
        Model-supplied ordinal used only to correlate the current all-angle
        response. The durable UUID is assigned server-side after parsing.
    variant_group_id
        Server-assigned UUID shared by peers probing the same logical
        problem. NULL for legacy, manual, imported, and role-only rows.
    source_refs
        UUIDs of the source chunks the question is grounded in.
    rationale
        Internal explanation of why this question was generated.
        Never shown to the learner.
    model_answer
        Teacher-facing reference/model answer for this question. Authoring
        aid only — never shown to the learner and never used to auto-grade
        (live sessions are scored against :class:`InterviewOutcome` rubric
        criteria, not string/semantic matching against this field).
    """

    question_type: InterviewQuestionType
    prompt_text: str
    difficulty: InterviewDifficulty
    expected_depth: int
    linked_outcome_id: UUID | None
    logical_question_index: int | None = None
    variant_group_id: UUID | None = None
    source_refs: list[UUID] = field(default_factory=list)
    rationale: str = ""
    model_answer: str = ""
