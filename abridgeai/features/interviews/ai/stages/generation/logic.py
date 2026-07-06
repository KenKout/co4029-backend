"""Interview generation stage orchestrator (T6.5).

Ports ``backend/app/ai/haystack/pipelines/interview_generation.py`` to
the feature-first layout. Composes the T2.1 :class:`LLMGateway`
(``LLMRole.INTERVIEW_GENERATION``, ``stage_name="interview_generation"``),
the T0.10 Jinja2 prompt loader, and the sibling :mod:`parsers` module.

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

import json
from typing import TYPE_CHECKING, Any, Protocol
from uuid import UUID

from abridgeai.ai.llm import LLMGateway, LLMRole
from abridgeai.ai.prompts import render_prompt
from abridgeai.features.interviews.ai.stages.generation.parsers import (
    InterviewQuestionDraft,
    parse_generation_response,
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
_DEFAULT_QUESTION_COUNT = 8
_MIN_QUESTION_COUNT = 1
_MAX_QUESTION_COUNT = 50
_DEFAULT_TYPE_MIX: dict[str, int] = {"technical": 60, "behavioral": 30, "situational": 10}


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
) -> list[InterviewQuestionDraft]:
    """Run one INTERVIEW_GENERATION LLM call and return parsed drafts.

    The caller (T6.10 pipeline) persists accepted drafts as
    :class:`InterviewQuestion` rows after the VALIDATION stage (T6.6).
    ``run.id`` is threaded into ``ai_model_calls`` for cost attribution and
    ``run.config_json`` supplies the teacher's ``question_count``. ``outcomes``
    may be empty (prompt emits ``null`` ``linked_outcome_id`` then). Bad LLM
    rows are dropped; an empty list means every question failed parsing.
    """

    type_mix = _resolve_type_mix(config.supplementary_instructions)
    question_count = _resolve_question_count(
        run_config_json=getattr(run, "config_json", None),
        supplementary=config.supplementary_instructions,
    )
    persona = config.persona or "neutral"

    user_prompt = render_prompt(
        "prompts/user.j2",
        title=config.title,
        persona=persona,
        question_count=question_count,
        technical_pct=type_mix["technical"],
        behavioral_pct=type_mix["behavioral"],
        situational_pct=type_mix["situational"],
        supplementary_instructions=(config.supplementary_instructions or "").strip(),
        outcomes=_outcomes_for_prompt(outcomes),
        chunks_block=_render_chunks(context),
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
    drafts = parse_generation_response(result.content_json, max_questions=question_count)
    return _link_outcomes_round_robin(drafts, outcomes)


def _resolve_type_mix(supplementary: str | None) -> dict[str, int]:
    """Return weights summing to 100 — fall back to the 60/30/10 default."""
    parsed = _try_parse_rubric(supplementary)
    if parsed is None:
        return dict(_DEFAULT_TYPE_MIX)
    raw_weights = parsed.get("rubric_weights")
    if not isinstance(raw_weights, dict):
        return dict(_DEFAULT_TYPE_MIX)
    cleaned: dict[str, int] = {key: 0 for key in _DEFAULT_TYPE_MIX}
    for key, value in raw_weights.items():
        if not isinstance(key, str):
            continue
        normalised_key = key.strip().lower()
        if normalised_key == "behavioural":  # accept BrEng spelling
            normalised_key = "behavioral"
        if normalised_key not in cleaned:
            continue
        try:
            cleaned[normalised_key] = max(0, int(value))
        except (TypeError, ValueError):
            continue
    total = sum(cleaned.values())
    if total <= 0:
        return dict(_DEFAULT_TYPE_MIX)
    return {key: round(value * 100 / total) for key, value in cleaned.items()}


def _resolve_question_count(
    *,
    run_config_json: dict[str, Any] | None,
    supplementary: str | None,
) -> int:
    """Resolve question count, clamped to [1, 50].

    Precedence: form value (``run_config_json["question_count"]``) →
    ``supplementary_instructions`` JSON override → default.
    """
    from_form = _coerce_question_count(
        run_config_json.get("question_count") if isinstance(run_config_json, dict) else None
    )
    if from_form is not None:
        return from_form

    parsed = _try_parse_rubric(supplementary)
    if parsed is not None:
        from_supplementary = _coerce_question_count(parsed.get("question_count"))
        if from_supplementary is not None:
            return from_supplementary

    return _DEFAULT_QUESTION_COUNT


def _coerce_question_count(raw: object) -> int | None:
    """Parse + clamp a raw count to [1, 50]; None if unusable."""
    if raw is None or isinstance(raw, bool):
        return None
    try:
        count = int(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return max(_MIN_QUESTION_COUNT, min(_MAX_QUESTION_COUNT, count))


def _try_parse_rubric(supplementary: str | None) -> dict[str, Any] | None:
    """Best-effort JSON parse of the supplementary-instructions field."""
    if not supplementary:
        return None
    stripped = supplementary.strip()
    if not stripped or not stripped.startswith("{"):
        return None
    try:
        parsed = json.loads(stripped)
    except (TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


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
    """Fill any ``linked_outcome_id is None`` slots round-robin from ``outcomes``."""
    if not outcomes:
        return drafts
    cursor = 0
    for draft in drafts:
        if draft.linked_outcome_id is None:
            draft.linked_outcome_id = outcomes[cursor % len(outcomes)].id
            cursor += 1
    return drafts


__all__ = ["GenerationRunRef", "InterviewRetrievalContext", "generate_interview_questions"]
