"""Per-question quiz regeneration pipeline (T5.11).

Ports ``_run_question_regeneration`` from
``backend/app/ai/haystack/pipelines/quiz_generation.py:277-348``.

This is the "narrow" pipeline: rebuild ONE question in place. The
existing question's prompt + bloom/difficulty form the template, so
ideation is **skipped** (plan §5889). Sibling questions in the same
quiz are passed as ``previous_questions`` to the generation stage so
the LLM does not regenerate a near-duplicate of another question. A
dedup pass against the rest of the quiz keeps the regenerated
candidate from colliding with siblings (plan §5890). Persistence is
replace-in-place: the same ``QuizQuestion.id`` survives, ``revision_no``
is bumped, and review state resets to ``pending``.

One ``pipeline_run_id`` is generated here and threaded through every
stage so ``ai_model_calls`` rows roll up to a single
``ai_pipeline_runs`` parent — same audit contract as T5.10.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import uuid4

from sqlalchemy import select

from abridgeai.features.quizzes.ai.stages.dedup import discard_duplicates
from abridgeai.features.quizzes.ai.stages.generation import generate_questions
from abridgeai.features.quizzes.ai.stages.persistence import replace_question_in_place
from abridgeai.features.quizzes.ai.stages.retrieval import retrieve_chunks
from abridgeai.features.quizzes.ai.stages.validation import (
    apply_verdicts,
    validate_questions,
)
from abridgeai.features.quizzes.models import QuizQuestion

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from abridgeai.ai.knowledge_graph.schemas import KGContext
    from abridgeai.ai.retrieval import ChunkWithDistance
    from abridgeai.features.quizzes.models import Quiz


async def run_question_regeneration(
    db: AsyncSession,
    run: Any,  # noqa: ANN401 -- GenerationRun ORM still in legacy backend
    quiz: Quiz,
    question: QuizQuestion,
    chunks: list[ChunkWithDistance],
    kg_context: KGContext,
    config: dict[str, Any],
) -> QuizQuestion:
    """Regenerate ``question`` in place via generation → validation
    → dedup → persistence.

    When ``chunks`` is empty, retrieval is re-run anchored on the
    existing question's prompt so the pipeline is usable standalone
    (mirrors the ``chunks=[]`` shortcut in T5.10's full pipeline).
    """

    pipeline_run_id = uuid4()

    if not chunks:
        chunks, _, _ = await retrieve_chunks(
            db,
            run_id=run.id,
            quiz=quiz,
            config=config,
            question_anchor=question.prompt_text,
            pipeline_run_id=pipeline_run_id,
        )

    if not chunks:
        raise ValueError("Quiz regeneration aborted: retrieval produced zero chunks")

    template: dict[str, Any] = {
        "position": question.position,
        "topic": question.prompt_text,
        "question_type": question.question_type,
        "bloom_level": question.bloom_level or "understand",
        "difficulty": question.difficulty or "medium",
        "source_chunk_ids": _existing_source_chunk_ids(question),
        "rationale": "Replacement for an existing question.",
    }

    previous_questions = await _fetch_sibling_prompts(db, quiz, exclude_id=question.id)

    candidates = await generate_questions(
        title=quiz.title,
        config=config | {"question_count": 1},
        chunks=chunks,
        templates=[template],
        kg_context=kg_context,
        db=db,
        pipeline_run_id=pipeline_run_id,
        previous_questions=previous_questions,
    )
    if not candidates:
        raise ValueError("Per-question regeneration produced no candidate")

    candidate_dicts: list[dict[str, Any]] = [c.model_dump() for c in candidates]

    _, verdicts = await validate_questions(
        title=quiz.title,
        chunks=chunks,
        questions=candidate_dicts,
        db=db,
        pipeline_run_id=pipeline_run_id,
        config=config,
    )
    accepted, rejected, _ = apply_verdicts(candidate_dicts, verdicts)
    if not accepted:
        reason = (
            rejected[0]["reasons"][0] if rejected and rejected[0].get("reasons") else "no reason"
        )
        raise ValueError(f"Validator rejected the regenerated question: {reason}")

    kept, drops = await discard_duplicates(db, quiz, accepted)
    if not kept:
        drop_reason = drops[0].reason if drops else "unknown"
        raise ValueError(f"Quiz regeneration aborted: candidate dropped by dedup ({drop_reason})")

    payload = kept[0]
    payload["source_refs"] = payload.get("source_refs") or payload.get("source_refs_json") or []
    persisted = await replace_question_in_place(
        db,
        run,
        question,
        payload,
        chunks=chunks,
    )

    run.config_json = run.config_json | {
        "pipeline": {
            "stage": "completed",
            "mode": "regenerate_question",
            "pipeline_run_id": str(pipeline_run_id),
            "question_id": str(question.id),
            "validation": {"accepted": len(accepted), "rejected": rejected},
            "dedup": {
                "kept": len(kept),
                "dropped": [{"index": d.index, "reason": d.reason} for d in drops],
            },
        }
    }
    return persisted


async def _fetch_sibling_prompts(
    db: AsyncSession,
    quiz: Quiz,
    *,
    exclude_id: Any,  # noqa: ANN401 -- UUID, but ORM also accepts strings
) -> list[str]:
    stmt = (
        select(QuizQuestion.prompt_text)
        .where(QuizQuestion.quiz_id == quiz.id)
        .where(QuizQuestion.id != exclude_id)
        .where(QuizQuestion.deleted_at.is_(None))
    )
    result = await db.execute(stmt)
    return [row[0] for row in result.all() if row[0]]


def _existing_source_chunk_ids(question: QuizQuestion) -> list[str]:
    refs = question.source_refs or []
    if not isinstance(refs, list):
        return []
    chunk_ids: list[str] = []
    for ref in refs:
        if isinstance(ref, str):
            chunk_ids.append(ref)
        elif isinstance(ref, dict) and isinstance(ref.get("chunk_id"), str):
            chunk_ids.append(ref["chunk_id"])
    return chunk_ids


__all__ = ["run_question_regeneration"]
