"""Quiz generation pipeline entrypoint (T5.13).

Top-level dispatcher invoked by the ARQ worker for the
``run_quiz_generation_task`` job. Reads the :class:`GenerationRun`
row, normalizes :attr:`GenerationRun.config_json`, and routes to one
of three pipelines:

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
``outlines`` + ``budget``. Phase 3 of the FR-5 schema port (T5.14)
landed :mod:`abridgeai.features.quizzes.ai.outline`, so this dispatcher
now precomputes both before invoking coverage:

1. Read ``source_lesson_ids`` from ``config_json`` (FR-5 schema field).
2. ``build_lesson_outline(db, lesson_ids, slides_per_section=...)``
   reads ``document_chunks`` and groups them — pure SQL + Python, no
   LLM calls, cheap to run inside the worker tick.
3. ``allocate_question_budget(outlines, total=question_count, ...)``
   computes the per-section question count using the
   ``coverage_options`` block from the schema.
4. Pass both into :func:`run_coverage_pipeline`.

If ``source_lesson_ids`` is empty or no chunks exist, coverage falls
back to ``ValueError`` (legacy behaviour). The schema layer rejects
malformed coverage config at the HTTP boundary so this dispatcher only
sees validated payloads.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING, Any
from uuid import UUID

from abridgeai.ai.knowledge_graph.schemas import KGContext
from abridgeai.ai.models import GenerationRun
from abridgeai.core.exceptions import NotFoundError
from abridgeai.core.security import utcnow
from abridgeai.features.quizzes.ai.outline import (
    allocate_question_budget,
    build_lesson_outline,
)
from abridgeai.features.quizzes.ai.pipelines import (
    coverage as coverage_pipeline,
)
from abridgeai.features.quizzes.ai.pipelines import (
    full as full_pipeline,
)
from abridgeai.features.quizzes.ai.pipelines import (
    regenerate as regenerate_pipeline,
)
from abridgeai.features.quizzes.models import Quiz, QuizQuestion
from abridgeai.features.quizzes.services.completion_notify import (
    notify_quiz_generation_outcome,
)

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


def _config_uuid_list(config: dict[str, Any], key: str) -> list[UUID]:
    """Parse a ``list[str]`` of UUIDs from ``config_json`` into ``list[UUID]``.

    Silently drops malformed entries — the schema layer already rejects
    non-UUID strings at the HTTP boundary, so this is just defence in
    depth for hand-crafted runs. Empty list when the key is missing.
    """
    raw = config.get(key) or []
    if not isinstance(raw, list):
        return []
    out: list[UUID] = []
    for item in raw:
        try:
            out.append(UUID(str(item)))
        except (TypeError, ValueError):
            continue
    return out


async def _resolve_outcome_texts(db: AsyncSession, config: dict[str, Any]) -> list[str]:
    """Resolve ``target_outcome_ids`` → ordered outcome text strings (§LO-3).

    Coverage rollup: targeting a parent outcome implicitly targets its
    whole subtree — a recursive CTE walks children so choosing L.O.1 pulls
    in L.O.1.1, L.O.1.1.1, … The dotted display code is rebuilt from the
    parent chain (``string_agg`` over the path) so labels match the
    teacher-facing ``L.O.x.y`` codes.

    Raw SQL against ``course_learning_outcomes`` (quizzes must not import the
    courses ORM — T0.4 feature-independence). Soft-deleted outcomes are
    excluded from both the seed and the walk. Empty list when none targeted.
    """
    outcome_ids = _config_uuid_list(config, "target_outcome_ids")
    if not outcome_ids:
        return []
    from sqlalchemy import text as _text  # noqa: PLC0415

    # subtree: seed = the targeted ids; recurse to every non-deleted child.
    # code_path: walk root→node accumulating positions into a dotted string.
    rows = (
        await db.execute(
            _text(
                """
                WITH RECURSIVE subtree AS (
                    SELECT id, parent_id, position, outcome_text
                    FROM course_learning_outcomes
                    WHERE id = ANY(:ids) AND deleted_at IS NULL
                    UNION
                    SELECT c.id, c.parent_id, c.position, c.outcome_text
                    FROM course_learning_outcomes c
                    JOIN subtree s ON c.parent_id = s.id
                    WHERE c.deleted_at IS NULL
                ),
                coded AS (
                    SELECT id, parent_id, position, outcome_text,
                           position::text AS code
                    FROM course_learning_outcomes
                    WHERE parent_id IS NULL AND deleted_at IS NULL
                    UNION ALL
                    SELECT c.id, c.parent_id, c.position, c.outcome_text,
                           coded.code || '.' || c.position::text
                    FROM course_learning_outcomes c
                    JOIN coded ON c.parent_id = coded.id
                    WHERE c.deleted_at IS NULL
                )
                SELECT DISTINCT coded.code, coded.outcome_text,
                       string_to_array(coded.code, '.')::int[] AS sort_key
                FROM coded
                JOIN subtree ON subtree.id = coded.id
                ORDER BY sort_key
                """
            ),
            {"ids": outcome_ids},
        )
    ).all()
    return [f"L.O.{row[0]}: {row[1]}" for row in rows]


def _inject_outcome_guidance(config: dict[str, Any], outcome_texts: list[str]) -> dict[str, Any]:
    """Merge learning-outcome guidance into ``config['extra_instructions']``.

    Reuses the existing prompt slot (no template/signature change) so the
    generation LLM steers questions toward the targeted outcomes. Returns a
    new config dict; the original is left untouched.
    """
    if not outcome_texts:
        return config
    block = (
        "Target these course learning outcomes — every question must assess "
        "one of them:\n" + "\n".join(f"- {t}" for t in outcome_texts)
    )
    existing = (config.get("extra_instructions") or "").strip()
    merged = f"{existing}\n\n{block}" if existing else block
    return dict(config) | {"extra_instructions": merged}


async def _precompute_coverage_inputs(
    db: AsyncSession, config: dict[str, Any]
) -> tuple[list[Any], dict[str, int]]:
    """Build outlines + allocate budget for coverage mode.

    Returns ``(outlines, budget)``. Raises ``ValueError`` if the run
    config doesn't have ``source_lesson_ids`` or no chunks exist for
    them — same surface as the legacy pipeline so existing test
    expectations carry over.
    """
    lesson_ids = _config_uuid_list(config, "source_lesson_ids")
    if not lesson_ids:
        raise ValueError("coverage mode requires source_lesson_ids in config_json")

    cov_opts = config.get("coverage_options") or {}
    if not isinstance(cov_opts, dict):
        cov_opts = {}

    section_grouping = str(cov_opts.get("section_grouping") or "auto")
    outlines = await build_lesson_outline(
        db,
        lesson_ids,
        slides_per_section=int(cov_opts.get("slides_per_section") or 4),
        force_bundle=(section_grouping == "fixed"),
    )
    if not outlines:
        raise ValueError("coverage mode: no document chunks found for source_lesson_ids")

    question_count = int(config.get("question_count") or 0)
    if question_count <= 0:
        raise ValueError("coverage mode requires positive question_count in config_json")

    section_ids = cov_opts.get("section_ids")
    if section_ids is not None and not isinstance(section_ids, list):
        section_ids = None

    budget = allocate_question_budget(
        outlines,
        total=question_count,
        min_per_section=int(cov_opts.get("min_per_section") or 1),
        max_per_section=int(cov_opts.get("max_per_section") or 5),
        skip_summaries=bool(cov_opts.get("skip_summaries", True)),
        section_ids=section_ids,
    )
    return outlines, budget


async def run_quiz_generation(
    db: AsyncSession,
    generation_run_id: UUID,
    *,
    arq_pool: object | None = None,
) -> None:
    """ARQ entrypoint: dispatch ``generation_run_id`` to the right pipeline.

    Marks the run ``running`` before invoking the pipeline. On any
    exception inside the pipeline, the session is rolled back, the run
    row is re-fetched, ``status='failed'`` + the failure message are
    stamped on ``config_json``, the failure is committed, and the
    original exception is re-raised so the ARQ retry budget engages.

    ``arq_pool`` (the worker's ``ctx['redis']``) is threaded through solely so
    the completion/failure notification can enqueue an email; ``None`` in
    sync/test paths simply skips the email fan-out.
    """
    run = await db.get(GenerationRun, generation_run_id)
    if run is None:
        raise NotFoundError("Generation run not found")
    run_id = run.id
    # Captured for the completion notification (recipient + deep-link target).
    run_requested_by = run.requested_by
    run_course_id = run.course_id

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
            # §LO-3: inject targeted learning-outcome text into the prompt so
            # the generation LLM steers questions toward them. The round-robin
            # binding at persistence guarantees each question is bound; this
            # just improves relevance. No-op when no outcomes were targeted.
            outcome_texts = await _resolve_outcome_texts(db, config)
            config = _inject_outcome_guidance(config, outcome_texts)
            generation_mode = str(config.get("generation_mode") or "topic").strip().lower()
            if generation_mode == "coverage":
                outlines, budget = await _precompute_coverage_inputs(db, config)
                await coverage_pipeline.run_coverage_pipeline(
                    db=db,
                    run=run,
                    quiz=quiz,
                    config=config,
                    outlines=outlines,
                    budget=budget,
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
        # Notify the initiating teacher (best-effort; rides its own commit and
        # never disturbs the generation transaction above).
        await notify_quiz_generation_outcome(
            db,
            recipient_user_id=run_requested_by,
            quiz_id=quiz.id,
            course_id=run_course_id,
            quiz_title=quiz.title,
            succeeded=True,
            arq_pool=arq_pool,
        )
        await db.commit()
    except asyncio.CancelledError:
        # Worker shutdown / restart / job_timeout cancels the coroutine.
        # CancelledError is a BaseException (not Exception) so it bypasses the
        # failure handler below — we MUST still roll back here, otherwise the
        # open transaction is orphaned as a Postgres backend "idle in
        # transaction" that keeps holding the generation_runs row lock,
        # wedging the run (and any recovery) until the connection is killed.
        # The reaper will re-enqueue/fail the run row; our job is just to
        # release the lock. Re-raise so ARQ sees the cancellation.
        with contextlib.suppress(Exception):
            await db.rollback()
        raise
    except Exception as exc:
        await db.rollback()
        run = await db.get(GenerationRun, run_id)
        if run is None:
            raise
        run.status = "failed"
        run.config_json = dict(run.config_json or {}) | {"failure": {"message": str(exc)}}
        run.finished_at = utcnow()
        await db.commit()
        # Notify the initiating teacher of the failure (best-effort).
        await notify_quiz_generation_outcome(
            db,
            recipient_user_id=run_requested_by,
            quiz_id=quiz.id,
            course_id=run_course_id,
            quiz_title=quiz.title,
            succeeded=False,
            error_message=str(exc),
            arq_pool=arq_pool,
        )
        await db.commit()
        raise


__all__ = ["run_quiz_generation"]
