"""Full quiz generation pipeline orchestrator (T5.10).

Ports ``_run_full_pipeline`` from
``backend/app/ai/haystack/pipelines/quiz_generation.py:163-275``.

Stage order (plan §5840): retrieval → ideation → generation → validation
→ dedup → persistence. One ``pipeline_run_id`` is generated here and
threaded through every stage so ``ai_model_calls`` rows roll up to a
single ``ai_pipeline_runs`` parent. Stage exceptions propagate; the
caller owns the transaction and run-level failure recording.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import uuid4

from abridgeai.features.quizzes.ai.pipelines._progress import FULL_STAGES, record_stage
from abridgeai.features.quizzes.ai.pipelines._synthetic_outline import resolve_outline_inputs
from abridgeai.features.quizzes.ai.pipelines._telemetry import log_validator_aborted_run
from abridgeai.features.quizzes.ai.stages.dedup import discard_duplicates
from abridgeai.features.quizzes.ai.stages.generation import generate_questions
from abridgeai.features.quizzes.ai.stages.generation.parsers import question_for_review
from abridgeai.features.quizzes.ai.stages.ideation import ideate_for_outline
from abridgeai.features.quizzes.ai.stages.persistence import persist_questions
from abridgeai.features.quizzes.ai.stages.retrieval import (
    retrieval_metadata,
    retrieve_chunks,
)
from abridgeai.features.quizzes.ai.stages.validation import (
    apply_verdicts,
    validate_questions,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from abridgeai.ai.knowledge_graph.schemas import KGContext
    from abridgeai.ai.retrieval import ChunkWithDistance
    from abridgeai.features.quizzes.models import Quiz, QuizQuestion

_IDEATION_OVERSAMPLE = 1.5


async def run_full_pipeline(
    db: AsyncSession,
    run: Any,  # noqa: ANN401 -- GenerationRun ORM still in legacy backend
    quiz: Quiz,
    chunks: list[ChunkWithDistance],
    kg_context: KGContext,
    config: dict[str, Any],
    *,
    outlines: list[Any] | None = None,
    budget: dict[str, int] | None = None,
) -> list[QuizQuestion]:
    """Compose the 6-stage full pipeline for a quiz draft.

    When ``chunks`` is empty, retrieval is re-run so the pipeline is
    usable standalone. When ``outlines``/``budget`` are ``None``, a flat
    single-section outline is synthesised from ``chunks`` so ideation
    has a routing key.
    """

    pipeline_run_id = uuid4()
    requested_count = int(config.get("question_count") or 3)
    template_target = max(
        requested_count,
        int(round(requested_count * _IDEATION_OVERSAMPLE)),
    )

    await record_stage(run.id, stages=FULL_STAGES, current_stage="retrieval")
    if not chunks:
        chunks, primary_embedding, anchors = await retrieve_chunks(
            db,
            run_id=run.id,
            quiz=quiz,
            config=config,
            pipeline_run_id=pipeline_run_id,
        )
        run.config_json = run.config_json | {
            "retrieval": retrieval_metadata(
                chunks,
                anchors=anchors,
                primary_embedding=primary_embedding,
            ),
        }

    if not chunks:
        raise ValueError("Quiz pipeline aborted: retrieval produced zero chunks")

    effective_outlines, effective_budget = resolve_outline_inputs(
        outlines, budget, chunks, template_target
    )

    await record_stage(
        run.id,
        stages=FULL_STAGES,
        current_stage="ideation",
        detail=f"{len(chunks)} chunks retrieved",
    )
    templates = await ideate_for_outline(
        db,
        run,
        title=quiz.title,
        config=config,
        outlines=effective_outlines,
        budget=effective_budget,
        pipeline_run_id=pipeline_run_id,
    )
    template_dicts: list[dict[str, Any]] = [t.model_dump() for t in templates[:requested_count]]

    await record_stage(
        run.id,
        stages=FULL_STAGES,
        current_stage="generation",
        detail=f"{len(template_dicts)} templates",
    )
    candidates = await generate_questions(
        title=quiz.title,
        config=config,
        chunks=chunks,
        templates=template_dicts,
        kg_context=kg_context,
        db=db,
        pipeline_run_id=pipeline_run_id,
        parent_run_id=run.id,
    )
    candidate_dicts: list[dict[str, Any]] = [c.model_dump() for c in candidates]

    if not candidate_dicts:
        raise ValueError("Quiz pipeline aborted: generation produced zero candidates")

    # Project each candidate into the validator's compact view *before*
    # calling the LLM — this surfaces non-MCQ ``correct_answer`` (which
    # lives in ``original_generated_payload``) so the validator can judge
    # groundedness instead of rejecting on "empty correct answer".
    review_dicts = [question_for_review(c) for c in candidates]

    await record_stage(
        run.id,
        stages=FULL_STAGES,
        current_stage="validation",
        detail=f"{len(candidate_dicts)} candidates",
    )
    _, verdicts = await validate_questions(
        title=quiz.title,
        chunks=chunks,
        questions=review_dicts,
        db=db,
        pipeline_run_id=pipeline_run_id,
        audit_parent_run_id=run.id,
        config=config,
    )
    accepted, rejected, _ = apply_verdicts(candidate_dicts, verdicts)

    await record_stage(
        run.id,
        stages=FULL_STAGES,
        current_stage="dedup",
        detail=f"{len(accepted)} accepted",
    )
    kept, drops = await discard_duplicates(db, quiz, accepted)
    if not kept:
        log_validator_aborted_run(
            candidates=candidate_dicts,
            rejected=rejected,
            drops=drops,
            verdicts=verdicts,
            log_prefix="quiz_pipeline_aborted",
        )
        raise ValueError(
            "Quiz pipeline aborted: no questions survived "
            f"(generated={len(candidate_dicts)}, "
            f"rejected_by_validator={len(rejected)}, "
            f"dropped_by_dedup={len(drops)})"
        )

    await record_stage(
        run.id,
        stages=FULL_STAGES,
        current_stage="persistence",
        detail=f"{len(kept)} questions",
    )
    persisted = await persist_questions(db, run, quiz, chunks, kept)

    run.config_json = run.config_json | {
        "pipeline": {
            "stage": "completed",
            "pipeline_run_id": str(pipeline_run_id),
            "ideation": {
                "requested_templates": template_target,
                "received_templates": len(templates),
                "used_templates": len(template_dicts),
            },
            "generation": {
                "requested_questions": requested_count,
                "received_questions": len(candidate_dicts),
            },
            "validation": {"accepted": len(accepted), "rejected": rejected},
            "dedup": {
                "kept": len(kept),
                "dropped": [{"index": d.index, "reason": d.reason} for d in drops],
            },
        }
    }
    return persisted


__all__ = ["run_full_pipeline"]
