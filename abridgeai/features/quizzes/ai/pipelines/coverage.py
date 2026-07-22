"""Outline-driven quiz generation pipeline (T5.12).

Ports ``_run_coverage_pipeline`` from
``backend/app/ai/haystack/pipelines/quiz_generation.py:351-562``.

Coverage mode walks a lesson outline, allocates a per-section question
budget, runs **one** ideation call across the whole outline, then fans
out generation per template (one template ↔ one section, one
generation LLM call) under a bounded semaphore. Validation, dedup, and
persistence run **once** on the aggregated questions, not per section.

Why this layout differs from :mod:`.full`
-----------------------------------------
* No global retrieval. Chunks come straight from
  ``OutlineSection.chunk_ids`` so per-template work is just an
  ``id IN (...)`` load.
* Generation parallelism is bounded by ``asyncio.Semaphore``: free-running
  fanout would either DoS the LLM provider or queue behind its rate limit.
* Each generation task gets its **own** ``AsyncSession`` so the gateway's
  per-call audit insert does not cross-talk between concurrent tasks. A
  failure inside a task rolls back its own session and returns ``None``;
  the parent pipeline keeps going (legacy semantics — see commit body).

Audit roll-up
-------------
One ``pipeline_run_id`` is generated at the top of the pipeline and
threaded through every gateway call (ideation + N×generation +
validation), so every ``ai_model_calls`` row created by this run shares
a parent ``ai_pipeline_runs`` id (Reconciliation §B1).

Outline / budget plumbing
-------------------------
``build_lesson_outline`` and ``allocate_question_budget`` from the legacy
``app.ai.haystack.components.outline`` module are **not yet ported** to
backend-new (out-of-scope for T5.12; see notepad). Until they land,
callers (T5.13 services entrypoint) must precompute outlines and budget
and pass them in as keyword arguments — this mirrors the ``outlines`` /
``budget`` plumbing in :func:`.full.run_full_pipeline` so both pipelines
share an outline contract.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from sqlalchemy import bindparam, text
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.dialects.postgresql import UUID as PG_UUID

from abridgeai.core.db import get_sessionmaker
from abridgeai.features.quizzes.ai.pipelines._progress import COVERAGE_STAGES, record_stage
from abridgeai.features.quizzes.ai.pipelines._telemetry import log_validator_aborted_run
from abridgeai.features.quizzes.ai.stages.dedup import discard_duplicates
from abridgeai.features.quizzes.ai.stages.generation import generate_questions
from abridgeai.features.quizzes.ai.stages.ideation import ideate_for_outline
from abridgeai.features.quizzes.ai.stages.persistence import persist_questions
from abridgeai.features.quizzes.ai.stages.validation import (
    apply_verdicts,
    validate_questions,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from abridgeai.ai.knowledge_graph.schemas import KGContext
    from abridgeai.ai.retrieval import ChunkWithDistance
    from abridgeai.features.quizzes.models import Quiz, QuizQuestion

logger = logging.getLogger(__name__)


# Default fanout when neither config nor settings supply one. Mirrors the
# legacy ``Settings.quiz_coverage_parallelism`` default of 6 conservatively
# halved (4) until that setting is ported in the Phase 5 cleanup wave —
# see notepad. Callers can override via ``coverage_options.parallelism``.
_DEFAULT_PARALLELISM = 4

# Default per-template attempt budget. ``2`` = one retry on transient
# failure (gateway 5xx, parse error, empty candidates). Mirrors the
# ``CoverageOptions.max_attempts`` schema default; callers override via
# ``coverage_options.max_attempts``.
_DEFAULT_MAX_ATTEMPTS = 2


async def run_coverage_pipeline(  # noqa: C901 -- pre-existing linear stage sequence (was 15>12 at HEAD); progress checkpoints add statements, not branches
    db: AsyncSession,
    run: Any,  # noqa: ANN401 -- GenerationRun ORM still in legacy backend
    quiz: Quiz,
    config: dict[str, Any],
    *,
    outlines: list[Any] | None = None,
    budget: dict[str, int] | None = None,
    kg_context: KGContext | None = None,
) -> list[QuizQuestion]:
    """Compose the coverage pipeline: outline ideation → fanout generation
    → aggregate validation/dedup/persistence.

    ``outlines`` / ``budget`` are required (see module docstring); a
    ``ValueError`` is raised if either is missing or empty. The legacy
    pipeline computed both inline from ``config['source_lesson_ids']``;
    that helper isn't ported yet, so the caller must precompute.
    """

    pipeline_run_id = uuid4()

    if not outlines:
        raise ValueError(
            "coverage mode requires precomputed outlines — "
            "build_lesson_outline is not yet ported to backend-new"
        )
    if not budget:
        raise ValueError("coverage mode: budget allocation produced no eligible sections")

    flat_sections = [s for o in outlines for s in o.sections]

    # Stash budget on the run for observability — same shape as legacy.
    run.config_json = run.config_json | {
        "coverage": {
            "section_count": len(flat_sections),
            "eligible_count": len(budget),
            "budget": dict(budget),
            "skipped_sections": [s.id for s in flat_sections if s.id not in budget],
        }
    }

    await record_stage(
        run.id,
        stages=COVERAGE_STAGES,
        current_stage="outline",
        detail=f"{len(flat_sections)} sections, {len(budget)} eligible",
    )
    # Step 1 — outline-aware ideation (one LLM call across the whole outline).
    await record_stage(run.id, stages=COVERAGE_STAGES, current_stage="ideation")
    templates = await ideate_for_outline(
        db,
        run,
        title=quiz.title,
        config=config,
        outlines=outlines,
        budget=budget,
        pipeline_run_id=pipeline_run_id,
    )
    if not templates:
        raise ValueError("coverage mode: ideation returned no templates")

    template_dicts: list[dict[str, Any]] = [t.model_dump() for t in templates]

    await record_stage(
        run.id,
        stages=COVERAGE_STAGES,
        current_stage="generation",
        detail=f"{len(template_dicts)} sections",
    )
    # Step 2 — per-template fanout under a bounded semaphore.
    parallelism = _resolve_parallelism(config)
    semaphore = asyncio.Semaphore(max(1, parallelism))
    max_attempts = _resolve_max_attempts(config)
    sessionmaker_ = get_sessionmaker()

    async def _generate_for_template(
        template: dict[str, Any],
    ) -> dict[str, Any] | None:
        chunk_uuids = _coerce_chunk_uuids(template.get("source_chunk_ids"))
        if not chunk_uuids:
            return None
        for attempt in range(max_attempts):
            async with semaphore, sessionmaker_() as task_db:
                try:
                    section_chunks = await _load_chunks_by_id(task_db, chunk_uuids)
                    if not section_chunks:
                        return None
                    candidates = await generate_questions(
                        title=quiz.title,
                        config=config | {"question_count": 1},
                        chunks=section_chunks,
                        templates=[template],
                        kg_context=kg_context,
                        db=task_db,
                        pipeline_run_id=pipeline_run_id,
                        parent_run_id=run.id,
                        previous_questions=None,
                    )
                    await task_db.commit()
                    if candidates:
                        return candidates[0].model_dump()
                except Exception as exc:  # noqa: BLE001 — one bad template
                    await task_db.rollback()
                    if attempt < max_attempts - 1:
                        logger.info(
                            "coverage: template position=%s attempt %s failed, retrying: %s",
                            template.get("position"),
                            attempt + 1,
                            exc,
                        )
                        continue
                    logger.warning(
                        "coverage: template position=%s generation failed: %s",
                        template.get("position"),
                        exc,
                    )
                    return None
        return None

    # ``return_exceptions=False`` matches legacy: per-template ``except``
    # already swallowed task-local failures into ``None``, so gather only
    # ever sees ``dict | None`` results — no exceptions to ferry back.
    fanout = await asyncio.gather(
        *[_generate_for_template(t) for t in template_dicts],
        return_exceptions=False,
    )
    questions: list[dict[str, Any]] = [q for q in fanout if q is not None]
    if not questions:
        raise ValueError("coverage mode: per-template generation produced no candidates")

    # Step 3 — collect every chunk any template referenced so the
    # validator has actual source content to ground judgments against.
    # Legacy passed [] here and got 100% UNGROUNDED rejects (see fix
    # in legacy line 510-518); preserve that fix on the port.
    all_chunk_ids: set[UUID] = set()
    for template in template_dicts:
        for cid in _coerce_chunk_uuids(template.get("source_chunk_ids")):
            all_chunk_ids.add(cid)
    all_chunks = await _load_chunks_by_id(db, list(all_chunk_ids))

    # Step 4 — validation across the whole batch (one LLM call).
    await record_stage(
        run.id,
        stages=COVERAGE_STAGES,
        current_stage="validation",
        detail=f"{len(questions)} candidates",
    )
    _, verdicts = await validate_questions(
        title=quiz.title,
        chunks=all_chunks,
        questions=questions,
        db=db,
        pipeline_run_id=pipeline_run_id,
        audit_parent_run_id=run.id,
        config=config,
    )
    accepted, rejected, _reasons = apply_verdicts(questions, verdicts)

    # Step 5 — dedup against the existing quiz + within-batch.
    await record_stage(
        run.id,
        stages=COVERAGE_STAGES,
        current_stage="dedup",
        detail=f"{len(accepted)} accepted",
    )
    kept, drops = await discard_duplicates(db, quiz, accepted)
    if not kept:
        log_validator_aborted_run(
            candidates=questions,
            rejected=rejected,
            drops=drops,
            verdicts=verdicts,
            log_prefix="coverage_pipeline_aborted",
        )
        raise ValueError(
            "coverage mode: no questions survived "
            f"(generated={len(questions)}, "
            f"rejected_by_validator={len(rejected)}, "
            f"dropped_by_dedup={len(drops)})"
        )

    # Step 6 — persistence (one batched insert).
    await record_stage(
        run.id,
        stages=COVERAGE_STAGES,
        current_stage="persistence",
        detail=f"{len(kept)} questions",
    )
    persisted = await persist_questions(db, run, quiz, all_chunks, kept)

    run.config_json = run.config_json | {
        "pipeline": {
            "stage": "completed",
            "mode": "coverage",
            "pipeline_run_id": str(pipeline_run_id),
            "ideation": {"received_templates": len(templates)},
            "generation": {
                "requested_questions": sum(budget.values()),
                "received_questions": len(questions),
                "parallelism": parallelism,
                "max_attempts": max_attempts,
            },
            "validation": {"accepted": len(accepted), "rejected": rejected},
            "dedup": {
                "kept": len(kept),
                "dropped": [{"index": d.index, "reason": d.reason} for d in drops],
            },
        }
    }
    return persisted


def _resolve_parallelism(config: dict[str, Any]) -> int:
    """Pick the per-template fanout limit from config / coverage_options.

    Resolution order:
      1. ``config['coverage_options']['parallelism']`` if set.
      2. Module default (:data:`_DEFAULT_PARALLELISM`).

    A server-wide ``Settings.quiz_coverage_parallelism`` knob was
    intentionally not ported (tracked in
    ``.sisyphus/notepads/backend-restructure/issues.md``). Callers that
    need ops-level tuning pass ``coverage_options.parallelism`` in the
    run config; the fallback default is conservative and safe.
    """
    coverage_options = config.get("coverage_options") or {}
    raw = coverage_options.get("parallelism")
    if raw is None:
        raw = _DEFAULT_PARALLELISM
    try:
        return int(raw)
    except (TypeError, ValueError):
        return _DEFAULT_PARALLELISM


def _resolve_max_attempts(config: dict[str, Any]) -> int:
    """Pick the per-template attempt budget from coverage_options.

    Resolution order:
      1. ``config['coverage_options']['max_attempts']`` if set.
      2. Module default (:data:`_DEFAULT_MAX_ATTEMPTS`).

    Bad values (non-int, < 1) silently fall back to the default rather
    than failing the whole run — schema validation already rejects
    out-of-bounds inputs at the API surface, so this branch only
    triggers on direct programmatic misuse.
    """
    coverage_options = config.get("coverage_options") or {}
    raw = coverage_options.get("max_attempts")
    if raw is None:
        return _DEFAULT_MAX_ATTEMPTS
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return _DEFAULT_MAX_ATTEMPTS
    return value if value >= 1 else _DEFAULT_MAX_ATTEMPTS


def _coerce_chunk_uuids(raw_ids: object) -> list[UUID]:
    """Best-effort cast of mixed ``str``/``UUID`` lists to ``list[UUID]``.

    Mirrors the per-template loop in legacy lines 461-466: bad ids are
    silently dropped (a single malformed id should not poison the rest
    of the section's chunks).
    """
    if not isinstance(raw_ids, list):
        return []
    out: list[UUID] = []
    for raw in raw_ids:
        if isinstance(raw, UUID):
            out.append(raw)
            continue
        try:
            out.append(UUID(str(raw)))
        except (TypeError, ValueError):
            continue
    return out


async def _load_chunks_by_id(
    db: AsyncSession,
    chunk_ids: list[UUID],
) -> list[ChunkWithDistance]:
    """Load DocumentChunk rows by id and adapt them to ``ChunkWithDistance``.

    Coverage mode skips retrieval entirely; the outline already pinned the
    relevant chunk ids per section. ``distance=0.0`` is a sentinel meaning
    "anchored selection, not similarity-ranked" — downstream stages only
    use ``content`` / ``id`` / ``metadata`` so the distance value is moot.

    Uses raw SQL (not the ``DocumentChunk`` ORM model) so the quizzes
    feature does not import from ``features.materials`` — see
    ``import-linter`` "Features are independent" contract.
    """
    if not chunk_ids:
        return []

    from abridgeai.ai.retrieval import ChunkWithDistance

    stmt = text(
        "SELECT id, material_version_id, course_id, lesson_id, content "  # noqa: S608
        "FROM document_chunks "
        "WHERE id = ANY(CAST(:chunk_ids AS uuid[]))"
    ).bindparams(bindparam("chunk_ids", type_=ARRAY(PG_UUID(as_uuid=True))))

    result = await db.execute(stmt, {"chunk_ids": list(chunk_ids)})
    rows = result.mappings().all()
    return [
        ChunkWithDistance(
            chunk_id=row["id"],
            material_version_id=row["material_version_id"],
            course_id=row["course_id"],
            lesson_id=row["lesson_id"],
            content=row["content"],
            distance=0.0,
        )
        for row in rows
    ]


__all__ = ["run_coverage_pipeline"]
