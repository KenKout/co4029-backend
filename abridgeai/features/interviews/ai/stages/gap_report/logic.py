"""Gap Report stage orchestrator (T6.9).

Runs after :func:`abridgeai.features.interviews.ai.stages.evaluation.evaluate_session`
to synthesize aBridgeAI's signature artefact: a personalised report
contrasting the student's THEORY performance (avg quiz score on related
modules) against their PRACTICE performance (interview rubric total).

Persistence boundary
--------------------
This stage **returns** a :class:`GapReportDraft`; it does **not**
persist. T6.11 services owns the write to ``gap_reports`` (with
``student_id``, ``course_id``, ``module_id``, ``source_quiz_attempt_id``,
``source_interview_session_id``, ``student_summary``,
``teacher_summary``, ``report_json``).

Cross-feature decoupling
------------------------
The ``quiz_attempts`` argument is typed via :class:`_QuizAttemptLike`
(local Protocol) so this stage does NOT import
``abridgeai.features.quizzes.models`` — that would break the
"Features are independent" import-linter contract. The session
parameter is typed against :class:`InterviewSession` (same feature, fine).

Audit fields
------------
* ``stage_name="gap_report"`` — required by Reconciliation §B1 so
  ``ai_model_calls`` rows roll up to the gap-report phase of the run.
* ``pipeline_run_id`` — threaded into the gateway for cost attribution.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable
from uuid import UUID

from sqlalchemy import bindparam, text

from abridgeai.ai.llm import LLMGateway, LLMRole
from abridgeai.ai.prompts import render_prompt
from abridgeai.features.interviews.ai.stages.gap_report.parsers import (
    parse_gap_report_response,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from sqlalchemy.ext.asyncio import AsyncSession

    from abridgeai.features.interviews.ai.stages.evaluation.rubric import (
        CriterionScore,
        RubricScores,
    )
    from abridgeai.features.interviews.models import InterviewSession


GAP_REPORT_STAGE_NAME = "gap_report"

_STRENGTH_THRESHOLD = 4.0
_WEAKNESS_THRESHOLD = 3.0
_MIN_TOTAL_RESOURCES = 3
_TOP_EVIDENCE_PER_CRITERION = 2


@runtime_checkable
class _QuizAttemptLike(Protocol):
    """Protocol matching the fields this stage reads off a quiz attempt.

    Decouples the stage from ``abridgeai.features.quizzes.models`` so the
    "Features are independent" import-linter contract stays green. Only
    ``score_percent`` is consulted; missing values fall back to 0.
    """

    score_percent: Decimal | float | int | None


@dataclass(frozen=True)
class StudyPlanItem:
    topic: str
    weakness_summary: str
    suggested_lesson_id: UUID | None
    suggested_resource_ids: list[UUID]
    priority: Literal["high", "medium", "low"]


@dataclass(frozen=True)
class GapReportDraft:
    """Stage output. Persisted by T6.11 services into ``gap_reports``."""

    discrepancy_score: float
    theory_score_avg: float
    practice_score: float
    strengths: list[str]
    weaknesses: list[str]
    study_plan: list[StudyPlanItem]
    student_summary: str
    teacher_summary: str
    report_json: dict[str, Any] = field(default_factory=dict)


async def generate_gap_report(
    db: AsyncSession,
    *,
    session: InterviewSession,
    rubric_scores: RubricScores,
    quiz_attempts: Sequence[_QuizAttemptLike],
    course_id: UUID,
    module_id: UUID | None = None,
    pipeline_run_id: UUID | None = None,
    gateway: LLMGateway | None = None,
) -> GapReportDraft:
    """Synthesize a Gap Report draft for one completed interview session.

    Discrepancy formula: ``theory_score_avg − practice_score`` where
    theory is the mean ``score_percent`` across ``quiz_attempts`` (0 when
    the list is empty) and practice is ``rubric_scores.total_score``.
    A positive discrepancy means the student knows the theory but
    underperformed live; negative means they performed strongly live but
    quiz coverage is weak.

    Resource-linking invariant: every :class:`StudyPlanItem` carries
    ``suggested_resource_ids`` of length ≥ 1 whenever the lesson library
    can supply one; the full plan surfaces ≥ 3 distinct resources when
    the library has 3+ resources to draw from. Items where the LLM
    returns no library resources fall back to the lesson's own
    resources.
    """

    theory_score_avg = _theory_average(quiz_attempts)
    practice_score = float(rubric_scores.total_score)
    discrepancy_score = round(theory_score_avg - practice_score, 2)

    library = await _load_lesson_library(db, course_id=course_id, module_id=module_id)

    strengths_seed = _strengths_from_aggregated(rubric_scores.aggregated)
    weaknesses_seed = _weaknesses_from_aggregated(rubric_scores.aggregated)
    evidence_excerpts = _evidence_excerpts(rubric_scores.response_evaluations)

    system_prompt = render_prompt("prompts/system.j2")
    user_prompt = json.dumps(
        {
            "theory_score_avg": theory_score_avg,
            "practice_score": practice_score,
            "discrepancy_score": discrepancy_score,
            "rubric_aggregated": rubric_scores.aggregated,
            "evidence_excerpts": evidence_excerpts,
            "lesson_library": library,
        },
        ensure_ascii=False,
    )

    gateway = gateway or LLMGateway()
    llm_result = await gateway.generate_json(
        role=LLMRole.GAP_REPORT_GENERATION,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        db=db,
        stage_name=GAP_REPORT_STAGE_NAME,
        pipeline_run_id=pipeline_run_id,
        parent_run_id=pipeline_run_id,
    )

    payload = llm_result.content_json if isinstance(llm_result.content_json, dict) else {}
    parsed = parse_gap_report_response(payload)

    strengths = parsed["strengths"] or strengths_seed
    weaknesses = parsed["weaknesses"] or weaknesses_seed
    study_plan = _normalise_study_plan(parsed["study_plan"], library)

    report_json: dict[str, Any] = {
        "theory_score_avg": theory_score_avg,
        "practice_score": practice_score,
        "discrepancy_score": discrepancy_score,
        "rubric_aggregated": dict(rubric_scores.aggregated),
        "strengths": list(strengths),
        "weaknesses": list(weaknesses),
        "study_plan": [_study_plan_item_for_json(item) for item in study_plan],
    }

    return GapReportDraft(
        discrepancy_score=discrepancy_score,
        theory_score_avg=theory_score_avg,
        practice_score=practice_score,
        strengths=list(strengths),
        weaknesses=list(weaknesses),
        study_plan=study_plan,
        student_summary=parsed["student_summary"],
        teacher_summary=parsed["teacher_summary"],
        report_json=report_json,
    )


def _theory_average(quiz_attempts: Sequence[_QuizAttemptLike]) -> float:
    if not quiz_attempts:
        return 0.0
    values: list[float] = []
    for attempt in quiz_attempts:
        score = getattr(attempt, "score_percent", None)
        if score is None:
            continue
        try:
            values.append(float(score))
        except (TypeError, ValueError):
            continue
    if not values:
        return 0.0
    return round(sum(values) / len(values), 2)


def _strengths_from_aggregated(aggregated: dict[str, float]) -> list[str]:
    return [
        f"{criterion}: scored {score:.2f}/5 — strong"
        for criterion, score in aggregated.items()
        if score >= _STRENGTH_THRESHOLD
    ]


def _weaknesses_from_aggregated(aggregated: dict[str, float]) -> list[str]:
    return [
        f"{criterion}: scored {score:.2f}/5 — needs work"
        for criterion, score in aggregated.items()
        if score < _WEAKNESS_THRESHOLD
    ]


def _evidence_excerpts(
    response_evaluations: Sequence[Any],
) -> list[dict[str, Any]]:
    """Surface up to two lowest-scoring justifications per criterion.

    The LLM uses these to anchor ``teacher_summary`` in concrete
    evidence rather than generic platitudes. Caller already passed the
    aggregated criterion means; the per-response justifications are
    where the actionable detail lives.
    """

    by_criterion: dict[str, list[CriterionScore]] = {}
    for evaluation in response_evaluations:
        for cs in getattr(evaluation, "criterion_scores", []):
            by_criterion.setdefault(cs.criterion, []).append(cs)

    excerpts: list[dict[str, Any]] = []
    for criterion, scores in by_criterion.items():
        sorted_scores = sorted(scores, key=lambda cs: cs.score)
        for cs in sorted_scores[:_TOP_EVIDENCE_PER_CRITERION]:
            justification = (cs.justification or "").strip()
            if not justification:
                continue
            excerpts.append(
                {
                    "criterion": criterion,
                    "score": cs.score,
                    "justification": justification[:280],
                }
            )
    return excerpts


async def _load_lesson_library(
    db: AsyncSession,
    *,
    course_id: UUID,
    module_id: UUID | None,
) -> list[dict[str, Any]]:
    """Pull lessons + their student-visible resources for the LLM to pick from.

    Scoped to ``module_id`` when present so suggestions stay close to
    the assessed material; falls back to the whole course otherwise.
    Returns a list of ``{id, title, summary, resources: [...]}`` dicts
    in stable order so prompt diffs are minimal across runs.

    Uses raw SQL via :func:`sqlalchemy.text` rather than importing
    ``abridgeai.features.courses.models`` — that direct cross-feature
    import would break the "Features are independent" import-linter
    contract. The query is read-only and joins ``lessons`` with
    ``modules`` (for course-scope) and ``lesson_resources`` (for
    student-visible attachments).
    """

    lesson_sql = """
        SELECT l.id, l.title, COALESCE(l.summary, '') AS summary
        FROM lessons l
        JOIN modules m ON m.id = l.module_id
        WHERE m.course_id = :course_id
          AND l.deleted_at IS NULL
          AND m.deleted_at IS NULL
    """
    params: dict[str, Any] = {"course_id": course_id}
    if module_id is not None:
        lesson_sql += " AND l.module_id = :module_id"
        params["module_id"] = module_id
    lesson_sql += " ORDER BY l.module_id, l.title"

    lesson_rows = (await db.execute(text(lesson_sql), params)).mappings().all()
    if not lesson_rows:
        return []

    lesson_ids = [row["id"] for row in lesson_rows]
    resource_stmt = text(
        """
        SELECT id, lesson_id, title, resource_type
        FROM lesson_resources
        WHERE lesson_id IN :lesson_ids
          AND visible_to_students IS TRUE
          AND deleted_at IS NULL
        ORDER BY lesson_id, position
        """
    ).bindparams(bindparam("lesson_ids", expanding=True))
    resource_rows = (await db.execute(resource_stmt, {"lesson_ids": lesson_ids})).mappings().all()
    resources_by_lesson: dict[UUID, list[dict[str, Any]]] = {}
    for row in resource_rows:
        resources_by_lesson.setdefault(row["lesson_id"], []).append(
            {
                "id": str(row["id"]),
                "title": row["title"],
                "resource_type": row["resource_type"],
            }
        )

    library: list[dict[str, Any]] = []
    for row in lesson_rows:
        summary_text = row["summary"] or ""
        library.append(
            {
                "id": str(row["id"]),
                "title": row["title"],
                "summary": summary_text[:240],
                "resources": resources_by_lesson.get(row["id"], []),
            }
        )
    return library


def _normalise_study_plan(
    items: list[dict[str, Any]],
    library: list[dict[str, Any]],
) -> list[StudyPlanItem]:
    """Convert parsed dicts to typed items + enforce resource invariants.

    For each item: when the LLM returned zero resource_ids, fall back to
    the resources of the chosen lesson in ``library``. After per-item
    fallback, if the total resource count is below
    :data:`_MIN_TOTAL_RESOURCES` and the library has spare resources,
    distribute them across the existing items in priority order so the
    plan still meets the acceptance criterion.
    """

    valid_resource_ids = _index_library(library)
    typed: list[StudyPlanItem] = []
    for item in items:
        resources = _filter_known_resources(item["suggested_resource_ids"], valid_resource_ids)
        lesson_id = item["suggested_lesson_id"]
        if lesson_id is not None and str(lesson_id) not in {entry["id"] for entry in library}:
            lesson_id = None
        if not resources and lesson_id is not None:
            resources = [UUID(res["id"]) for res in _resources_for_lesson(library, lesson_id)]
        typed.append(
            StudyPlanItem(
                topic=item["topic"],
                weakness_summary=item["weakness_summary"],
                suggested_lesson_id=lesson_id,
                suggested_resource_ids=resources,
                priority=item["priority"],
            )
        )

    return _backfill_resource_floor(typed, library)


def _index_library(library: list[dict[str, Any]]) -> set[UUID]:
    valid: set[UUID] = set()
    for lesson in library:
        for resource in lesson["resources"]:
            try:
                valid.add(UUID(resource["id"]))
            except ValueError:
                continue
    return valid


def _filter_known_resources(candidates: list[UUID], known: set[UUID]) -> list[UUID]:
    return [rid for rid in candidates if rid in known]


def _resources_for_lesson(
    library: list[dict[str, Any]],
    lesson_id: UUID,
) -> list[dict[str, Any]]:
    target = str(lesson_id)
    for lesson in library:
        if lesson["id"] == target:
            resources = lesson["resources"]
            if isinstance(resources, list):
                return resources
            return []
    return []


def _backfill_resource_floor(
    items: list[StudyPlanItem],
    library: list[dict[str, Any]],
) -> list[StudyPlanItem]:
    """Distribute spare library resources so the plan surfaces ≥3 distinct ids.

    Only runs when the library actually has at least
    :data:`_MIN_TOTAL_RESOURCES` distinct resources to draw from — there
    is no point fabricating coverage that the course does not support.
    Items are mutated in priority order (high → medium → low) and each
    receives spare ids appended to its existing list.
    """

    if not items:
        return items

    library_ids = list(_index_library(library))
    if len(library_ids) < _MIN_TOTAL_RESOURCES:
        return items

    distinct = {rid for item in items for rid in item.suggested_resource_ids}
    if len(distinct) >= _MIN_TOTAL_RESOURCES:
        return items

    priority_rank: dict[str, int] = {"high": 0, "medium": 1, "low": 2}
    ordered_indexes = sorted(
        range(len(items)), key=lambda idx: priority_rank.get(items[idx].priority, 3)
    )

    spares = [rid for rid in library_ids if rid not in distinct]
    enriched: list[StudyPlanItem] = list(items)
    cursor = 0
    while spares and len(distinct) < _MIN_TOTAL_RESOURCES:
        target_idx = ordered_indexes[cursor % len(ordered_indexes)]
        spare = spares.pop(0)
        existing = list(enriched[target_idx].suggested_resource_ids)
        if spare not in existing:
            existing.append(spare)
            distinct.add(spare)
            enriched[target_idx] = StudyPlanItem(
                topic=enriched[target_idx].topic,
                weakness_summary=enriched[target_idx].weakness_summary,
                suggested_lesson_id=enriched[target_idx].suggested_lesson_id,
                suggested_resource_ids=existing,
                priority=enriched[target_idx].priority,
            )
        cursor += 1
    return enriched


def _study_plan_item_for_json(item: StudyPlanItem) -> dict[str, Any]:
    return {
        "topic": item.topic,
        "weakness_summary": item.weakness_summary,
        "suggested_lesson_id": (
            str(item.suggested_lesson_id) if item.suggested_lesson_id else None
        ),
        "suggested_resource_ids": [str(rid) for rid in item.suggested_resource_ids],
        "priority": item.priority,
    }


__all__ = [
    "GAP_REPORT_STAGE_NAME",
    "GapReportDraft",
    "StudyPlanItem",
    "generate_gap_report",
]
