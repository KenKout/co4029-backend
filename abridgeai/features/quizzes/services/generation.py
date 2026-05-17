"""Quiz generation pipeline entrypoint (T5.13).

Top-level dispatcher invoked by the ARQ worker for the
``generate_quiz`` job. Reads the :class:`GenerationRun` row, normalizes
:attr:`GenerationRun.config_json`, and routes to one of three pipelines:

* ``run.config_json["question_id"]`` set → :func:`run_regenerate_pipeline`
  (per-question replacement; ports legacy
  ``_run_question_regeneration`` semantics — single question id, no
  ideation, replace-in-place).
* ``run.config_json["generation_mode"] == "coverage"`` →
  :func:`run_coverage_pipeline` (outline-driven section fanout).
* otherwise → :func:`run_full_pipeline` (retrieval → ideation → generation
  → validation → dedup → persistence).

Mirrors legacy ``app.ai.haystack.pipelines.quiz_generation.run_quiz_generation``
(lines 82-156): wrap pipeline invocation in try/except; on success stamp
``status='completed'`` + ``finished_at``; on exception roll the session
back, re-fetch the run, stamp ``status='failed'`` + the failure message
on ``config_json``, then re-raise so ARQ records the job-level failure.

Coverage-mode outline pre-computation
-------------------------------------
:func:`pipelines.coverage.run_coverage_pipeline` requires precomputed
``outlines`` + ``budget`` because ``build_lesson_outline`` /
``allocate_question_budget`` haven't been ported to backend-new yet
(documented in the coverage pipeline docstring). T5.13 dispatches to
coverage with ``outlines=None`` / ``budget=None`` so the pipeline
raises a clear ``ValueError`` until the outline helpers land — matches
the staged port of T5.12.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import UUID

from abridgeai.ai.knowledge_graph.schemas import KGContext
from abridgeai.core.exceptions import NotFoundError
from abridgeai.core.security import utcnow
from abridgeai.features.quizzes.ai.pipelines import (
    coverage as coverage_pipeline,
)
from abridgeai.features.quizzes.ai.pipelines import (
    full as full_pipeline,
)
from abridgeai.features.quizzes.ai.pipelines import (
    regenerate as regenerate_pipeline,
)
from abridgeai.features.quizzes.models import GenerationRun, Quiz, QuizQuestion

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


def _config_uuid(config: dict[str, Any] | None, key: str) -> UUID | None:
    if not config:
        return None
    raw = config.get(key)
    if raw is None:
        return None
    try:
        return UUID(str(raw))
    except (TypeError, ValueError):
        return None


async def run_quiz_generation(db: AsyncSession, generation_run_id: UUID) -> None:
    """ARQ entrypoint: dispatch ``generation_run_id`` to the right pipeline.

    Marks the run ``running`` before invoking the pipeline. On any
    exception inside the pipeline, the session is rolled back, the run
    row is re-fetched, ``status='failed'`` + the failure message are
    stamped on ``config_json``, the failure is committed, and the
    original exception is re-raised so the ARQ retry budget engages.
    """
    run = await db.get(GenerationRun, generation_run_id)
    if run is None:
        raise NotFoundError("Generation run not found")
    run_id = run.id

    target_question_id = _config_uuid(run.config_json, "question_id")
    if target_question_id is not None:
        question = await db.get(QuizQuestion, target_question_id)
        if question is None:
            raise NotFoundError("Quiz question not found for regeneration")
        quiz = await db.get(Quiz, question.quiz_id)
    else:
        question = None
        quiz_id = _config_uuid(run.config_json, "quiz_id")
        if quiz_id is None:
            raise NotFoundError("Generation run is missing quiz_id")
        quiz = await db.get(Quiz, quiz_id)
    if quiz is None:
        raise NotFoundError("Quiz not found for generation run")

    run.status = "running"
    run.started_at = utcnow()
    await db.commit()

    try:
        config = dict(run.config_json or {})
        if question is not None:
            await regenerate_pipeline.run_question_regeneration(
                db=db,
                run=run,
                quiz=quiz,
                question=question,
                chunks=[],
                kg_context=KGContext(),
                config=config,
            )
        else:
            generation_mode = str(config.get("generation_mode") or "topic").strip().lower()
            if generation_mode == "coverage":
                await coverage_pipeline.run_coverage_pipeline(
                    db=db,
                    run=run,
                    quiz=quiz,
                    config=config,
                    outlines=None,
                    budget=None,
                    kg_context=KGContext(),
                )
            else:
                await full_pipeline.run_full_pipeline(
                    db=db,
                    run=run,
                    quiz=quiz,
                    chunks=[],
                    kg_context=KGContext(),
                    config=config,
                )

        run.status = "completed"
        run.finished_at = utcnow()
        await db.commit()
    except Exception as exc:
        await db.rollback()
        run = await db.get(GenerationRun, run_id)
        if run is None:
            raise
        run.status = "failed"
        run.config_json = dict(run.config_json or {}) | {"failure": {"message": str(exc)}}
        run.finished_at = utcnow()
        await db.commit()
        raise


__all__ = ["run_quiz_generation"]
