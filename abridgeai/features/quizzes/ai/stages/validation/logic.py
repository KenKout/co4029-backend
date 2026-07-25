"""Quiz validation stage orchestrator (T5.7).

Ports ``_run_validation`` from
``backend/app/ai/haystack/pipelines/quiz_generation.py:817-842``. The
stage runs one LLM round-trip with ``LLMRole.VALIDATION`` (a stronger
model class than generation per the role tier map) and returns a
positional list of :class:`Verdict` objects ready for
:func:`apply_verdicts`.

Audit fields
------------
* ``stage_name="validation"`` — required by Reconciliation §B1 so
  ``ai_model_calls`` rows roll up to the validation phase of the run.
* ``pipeline_run_id`` — threaded into the gateway so per-call cost rolls
  up to the parent ``ai_pipeline_runs`` row.
* ``parent_run_id`` (legacy alias) — accepted for backwards compatibility
  with callers still on the old keyword. Prefer ``pipeline_run_id``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import UUID

from abridgeai.ai.llm import LLMGateway, LLMResult, LLMRole
from abridgeai.ai.prompts import render_prompt
from abridgeai.features.quizzes.ai.stages.validation.parsers import (
    Verdict,
    parse_validation_response,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from sqlalchemy.ext.asyncio import AsyncSession


VALIDATION_STAGE_NAME = "validation"


async def validate_questions(
    title: str,
    chunks: Sequence[Any],
    questions: list[dict[str, Any]],
    db: AsyncSession,
    *,
    pipeline_run_id: UUID | None = None,
    parent_run_id: UUID | None = None,
    audit_parent_run_id: UUID | None = None,
    config: Mapping[str, Any] | None = None,
    gateway: LLMGateway | None = None,
) -> tuple[LLMResult, list[Verdict]]:
    """Run the validator over ``questions`` and return positional verdicts.

    Parameters
    ----------
    title
        Quiz title — surfaced in the user prompt so the validator has
        scope context.
    chunks
        Source chunks the questions were drawn from. Each item must
        expose ``id`` (or ``chunk_id``) and ``content``; optional
        ``metadata['section_title']`` is rendered when present.
    questions
        Generation-stage output (normalised dicts). The validator only
        reads ``prompt_text``, ``options``, ``correct_answer`` and
        ``explanation`` — extra keys are passed through untouched.
    db
        Async session — passed to ``LLMGateway`` for the audit write.
    pipeline_run_id
        Pipeline-run row id (FK target on ``ai_pipeline_runs``).
    parent_run_id
        Legacy alias for ``pipeline_run_id``; accepted to ease the port.
    config
        Run config dict — only ``avoid_topics`` is rendered into the
        prompt today (the rejection list is the strict-prompt's main
        knob).
    gateway
        Inject a custom gateway (test seam). Defaults to
        ``LLMGateway()``.

    Returns
    -------
    tuple[LLMResult, list[Verdict]]
        Raw gateway result (kept for callers wanting to introspect
        usage/latency) and the positional verdict list — one entry per
        question in input order, missing entries defaulted to
        ``accept`` per :func:`parse_validation_response`.
    """

    if not questions:
        raise ValueError("validate_questions requires at least one question")

    review_questions = [_question_for_review(question) for question in questions]
    chunk_views = [_chunk_for_prompt(chunk) for chunk in chunks]
    avoid_topics = list((config or {}).get("avoid_topics") or [])

    system_prompt = render_prompt("prompts/system.j2")
    user_prompt = render_prompt(
        "prompts/user.j2",
        title=title,
        chunks=chunk_views,
        questions=review_questions,
        avoid_topics=avoid_topics,
    )

    gateway = gateway or LLMGateway()
    run_id = pipeline_run_id or parent_run_id

    llm_result = await gateway.generate_json(
        role=LLMRole.VALIDATION,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        db=db,
        stage_name=VALIDATION_STAGE_NAME,
        pipeline_run_id=run_id,
        parent_run_id=audit_parent_run_id,
    )

    payload = llm_result.content_json if isinstance(llm_result.content_json, dict) else {}
    verdicts = parse_validation_response(payload, question_count=len(questions))
    return llm_result, verdicts


def _question_for_review(question: dict[str, Any]) -> dict[str, Any]:
    """Project a generated question into the validator's compact view.

    Mirrors ``_question_for_review`` from the legacy pipeline (line
    1259-1277). Options arrive either as the canonical list-of-dicts
    shape from ``mappers/quiz.normalize_quiz_questions`` or as a flat
    ``{key: text}`` dict from older callers; both are flattened to a
    plain dict here so the prompt template stays trivial.
    """

    raw_options = question.get("options") or []
    options: dict[str, str] = {}
    raw_correct = question.get("correct_answer")
    correct: str | None = None
    if isinstance(raw_correct, str):
        correct = raw_correct
    elif isinstance(raw_correct, list):
        # fill_blank stores an ordered list of blanks; render
        # semicolon-separated so the validator can read it without
        # special casing JSON.
        correct = "; ".join(str(item) for item in raw_correct)

    if isinstance(raw_options, list):
        for opt in raw_options:
            if not isinstance(opt, dict):
                continue
            key = opt.get("option_key")
            if isinstance(key, str):
                options[key] = str(opt.get("option_text", ""))
                if correct is None and opt.get("is_correct"):
                    correct = key
    elif isinstance(raw_options, dict):
        options = {str(k): str(v) for k, v in raw_options.items()}

    # Phase 7: numerical / matching / ordering carry their answer on dedicated
    # fields rather than option rows or ``correct_answer``. Flatten those into
    # readable text so the validator can judge groundedness — otherwise it sees
    # an empty answer and rejects every question of these types.
    qtype = question.get("question_type", "multiple_choice")
    if not correct:
        if qtype == "numerical":
            answer = question.get("numeric_answer")
            if answer is not None:
                tolerance = question.get("numeric_tolerance")
                correct = (
                    str(answer)
                    if tolerance is None
                    else f"{answer} (tolerance {tolerance})"
                )
        elif qtype == "matching":
            pairs = question.get("match_pairs")
            if isinstance(pairs, list):
                correct = "; ".join(
                    f"{pair.get('left')} -> {pair.get('right')}"
                    for pair in pairs
                    if isinstance(pair, dict)
                )
        elif qtype == "ordering":
            items = question.get("ordering_sequence")
            if isinstance(items, list):
                correct = "; ".join(
                    f"{index}. {item}" for index, item in enumerate(items, start=1)
                )

    # Multi-select MCQ: report every correct letter, not just the first found.
    if qtype == "multiple_choice" and question.get("single_answer") is False:
        multi_keys = sorted(
            str(opt.get("option_key"))
            for opt in (raw_options if isinstance(raw_options, list) else [])
            if isinstance(opt, dict) and opt.get("is_correct")
        )
        if multi_keys:
            correct = ", ".join(multi_keys)

    return {
        "question_type": question.get("question_type", "multiple_choice"),
        "prompt_text": question.get("prompt_text", ""),
        "options": options,
        "correct_answer": correct or "",
        "explanation": question.get("explanation", ""),
        "bloom_level": question.get("bloom_level", ""),
        "difficulty": question.get("difficulty", ""),
    }


def _chunk_for_prompt(chunk: object) -> dict[str, Any]:
    """Coerce a chunk into the dict shape the user prompt expects."""

    chunk_id = getattr(chunk, "id", None) or getattr(chunk, "chunk_id", None)
    content = getattr(chunk, "content", "") or ""
    metadata = getattr(chunk, "metadata", None) or {}
    section_title = ""
    if isinstance(metadata, dict):
        section_title = str(metadata.get("section_title") or "")
    return {
        "id": str(chunk_id) if chunk_id is not None else "",
        "content": str(content),
        "section_title": section_title,
    }


__all__ = ["VALIDATION_STAGE_NAME", "validate_questions"]
