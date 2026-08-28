"""Interview generation stage orchestrator (T6.5).

Ports ``backend/app/ai/haystack/pipelines/interview_generation.py`` to
the feature-first layout. Composes the T2.1 :class:`LLMGateway`
(``LLMRole.INTERVIEW_GENERATION``, ``stage_name=\"interview_generation\"``),
the T0.10 Jinja2 prompt loader, and the sibling :mod:`parsers` module.
Config-resolution helpers (type mix, question count) live in
:mod:`.resolve` to keep this module under the LOC budget.

Stages do not implement HTTP / audit logic themselves: the gateway writes
one ``ai_model_calls`` row per call and rolls audit up to the parent
``GenerationRun.id``.

Type-mix policy
---------------
Default is 60% technical, 30% behavioural, 10% situational. Teachers
override via ``InterviewConfig.supplementary_instructions`` using the
``rubric_weights`` JSON shape::

    {"rubric_weights": {"technical": 70, "behavioral": 20, "situational": 10}}

Weights are normalised to sum to 100; missing types default to zero.

Difficulty progression
----------------------
The prompt asks for ``easy`` first, then ``medium``, then ``hard`` with
position-based bands (1-3 / 4-7 / 8+). The LLM produces the labels;
``expected_depth`` (1-5) is the numeric companion the VALIDATION stage
(T6.6) and SCORING stage (T6.9) consume.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol
from uuid import UUID

from abridgeai.ai.llm import LLMGateway, LLMRole
from abridgeai.ai.prompts import render_prompt
from abridgeai.features.interviews.ai.stages.evaluation.rubric import (
    resolve_supplementary_notes,
)
from abridgeai.features.interviews.ai.stages.generation.parsers import (
    InterviewQuestionDraft,
    parse_generation_response,
)
from abridgeai.features.interviews.ai.stages.generation.resolve import (
    VARIANT_ANGLES,
    resolve_question_count,
    resolve_type_mix,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from abridgeai.ai.retrieval import ChunkWithDistance
    from abridgeai.features.interviews.models import InterviewConfig, InterviewOutcome


class GenerationRunRef(Protocol):
    """Structural reference to ``GenerationRun`` — keeps this stage off the
    ``features.quizzes`` import path (upholds the "features-are-independent"
    import-linter contract). ``config_json`` carries the teacher's generation
    request, incl. the ``question_count`` typed in the form.
    """

    id: UUID
    config_json: dict[str, Any]


_STAGE_NAME = "interview_generation"


class InterviewRetrievalContext:
    """Forward-ref placeholder; the real type lives in T6.4 retrieval.

    Imported via :data:`TYPE_CHECKING` so this module does not pull a
    runtime dep on the retrieval stage. Tests construct it ad-hoc.
    """

    chunks: list[ChunkWithDistance]


async def generate_interview_questions(
    db: AsyncSession,
    *,
    run: GenerationRunRef,
    config: InterviewConfig,
    context: InterviewRetrievalContext,
    outcomes: list[InterviewOutcome],
    gateway: LLMGateway | None = None,
    override_question_count: int | None = None,
    avoid_prompts: list[str] | None = None,
    variant_strategy: str | None = None,
    role_type: str | None = None,
) -> list[InterviewQuestionDraft]:
    """Run one INTERVIEW_GENERATION LLM call and return parsed drafts.

    The caller (T6.10 pipeline) persists accepted drafts as
    :class:`InterviewQuestion` rows after the VALIDATION stage (T6.6).
    ``run.id`` is threaded into ``ai_model_calls`` for cost attribution and
    ``run.config_json`` supplies the teacher's ``question_count``. ``outcomes``
    may be empty (prompt emits ``null`` ``linked_outcome_id`` then). Bad LLM
    rows are dropped; an empty list means every question failed parsing.

    ``override_question_count`` lets the pipeline's backfill loop (T6.10)
    ask for exactly the number of questions still missing after validation
    dropped some drafts, instead of re-requesting the full original count.
    ``avoid_prompts`` (already-accepted prompt texts) is surfaced to the LLM
    on backfill calls so it does not repeat a question that already passed.
    """

    type_mix = resolve_type_mix(config.supplementary_instructions)
    question_count = override_question_count or resolve_question_count(
        run_config_json=getattr(run, "config_json", None),
        supplementary=config.supplementary_instructions,
    )
    persona = config.persona or "neutral"

    # Variant mode: ``question_count`` is the TOTAL number of rows requested.
    # ``all_angles`` asks the LLM for whole LOGICAL questions (each spawning one
    # variant per angle), so round the logical count up and produce
    # ``logical_count x len(angles)`` rows (the parser caps at that total).
    logical_count = question_count
    effective_total = question_count
    if variant_strategy == "all_angles":
        logical_count = (question_count + len(VARIANT_ANGLES) - 1) // len(VARIANT_ANGLES)
        effective_total = logical_count * len(VARIANT_ANGLES)

    user_prompt = render_prompt(
        "prompts/user.j2",
        title=config.title,
        persona=persona,
        question_count=logical_count,
        variant_strategy=variant_strategy,
        role_type=role_type,
        angles=list(VARIANT_ANGLES),
        technical_pct=type_mix["technical"],
        behavioral_pct=type_mix["behavioral"],
        situational_pct=type_mix["situational"],
        # Only the prose part: when the field holds structured JSON (rubric,
        # type mix, question count) the raw blob must NOT reach the prompt.
        supplementary_instructions=resolve_supplementary_notes(config.supplementary_instructions),
        outcomes=_outcomes_for_prompt(outcomes),
        chunks_block=_render_chunks(context),
        avoid_prompts=list(avoid_prompts or []),
    )
    system_prompt = render_prompt("prompts/system.j2")

    client = gateway or LLMGateway()
    result = await client.generate_json(
        role=LLMRole.INTERVIEW_GENERATION,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        db=db,
        stage_name=_STAGE_NAME,
        pipeline_run_id=run.id,
        parent_run_id=run.id,
    )
    drafts = parse_generation_response(
        result.content_json,
        max_questions=effective_total,
        require_logical_question_index=variant_strategy == "all_angles",
    )
    return _link_outcomes_round_robin(drafts, outcomes)


def _outcomes_for_prompt(outcomes: list[InterviewOutcome]) -> list[dict[str, Any]]:
    return [
        {
            "id": str(outcome.id),
            "outcome_text": outcome.outcome_text,
            "outcome_type": outcome.outcome_type,
            "importance_weight": outcome.importance_weight,
        }
        for outcome in outcomes
    ]


def _render_chunks(context: InterviewRetrievalContext) -> str:
    """Render retrieved chunks as a labelled prompt block."""
    chunks = list(getattr(context, "chunks", None) or [])
    if not chunks:
        return "No indexed chunks were found."
    rendered: list[str] = []
    for chunk in chunks:
        rendered.append(f"[{chunk.chunk_id}]\ntext: {chunk.content}")
    return "\n\n".join(rendered)


def _link_outcomes_round_robin(
    drafts: list[InterviewQuestionDraft],
    outcomes: list[InterviewOutcome],
) -> list[InterviewQuestionDraft]:
    """Fill missing outcome links while preserving all-angle group coherence."""
    if not outcomes:
        return drafts
    cursor = 0
    assigned_by_group: dict[UUID, UUID] = {}
    for draft in drafts:
        if draft.variant_group_id is not None and draft.linked_outcome_id is not None:
            assigned_by_group.setdefault(draft.variant_group_id, draft.linked_outcome_id)
    for draft in drafts:
        if draft.linked_outcome_id is not None:
            continue
        if draft.variant_group_id is not None:
            outcome_id = assigned_by_group.get(draft.variant_group_id)
            if outcome_id is None:
                outcome_id = outcomes[cursor % len(outcomes)].id
                cursor += 1
                assigned_by_group[draft.variant_group_id] = outcome_id
            draft.linked_outcome_id = outcome_id
        else:
            draft.linked_outcome_id = outcomes[cursor % len(outcomes)].id
            cursor += 1
    return drafts


__all__ = [
    "GenerationRunRef",
    "InterviewRetrievalContext",
    "generate_interview_questions",
    "resolve_question_count",
]
