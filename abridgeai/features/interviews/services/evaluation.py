"""Interview evaluation + gap-report orchestrator (T6.11).

ARQ entrypoint for the ``evaluate_interview_session_task`` job.
Composes:

* T6.8 :func:`evaluate_session` → :class:`RubricScores` (per-criterion
  + aggregated total).
* T6.9 :func:`generate_gap_report` → :class:`GapReportDraft`
  (theory/practice discrepancy + study plan).

Then persists :class:`InterviewOutcomeEvaluation` rows, updates the
session's ``internal_summary_json`` (canonical home for ``total_score``
+ ``rubric_aggregated`` per the baseline schema — there is no separate
``total_score`` column), inserts the :class:`GapReport` row, and
commits.

Failure path: on exception, the transaction is rolled back, the
session row is re-fetched, ``internal_summary_json['evaluation_failure']``
is stamped with the message, the failure is committed, and the
exception is re-raised so ARQ records the job-level failure for retry.
"""

from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal
from typing import TYPE_CHECKING, Any
from uuid import UUID

from abridgeai.core.db.conflict_mapper import flush_or_conflict
from abridgeai.core.exceptions import NotFoundError
from abridgeai.core.security import utcnow
from abridgeai.features.interviews.ai.stages.evaluation import evaluate_session
from abridgeai.features.interviews.ai.stages.evaluation.rubric import RubricScores
from abridgeai.features.interviews.ai.stages.gap_report import (
    GapReportDraft,
    generate_gap_report,
)
from abridgeai.features.interviews.models import (
    GapReport,
    InterviewOutcome,
    InterviewOutcomeEvaluation,
    InterviewSession,
    InterviewSessionMessage,
)
from abridgeai.features.interviews.queries import authoring as authoring_queries
from abridgeai.features.interviews.queries import sessions as sessions_queries

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


async def evaluate_and_generate_report(db: AsyncSession, session_id: UUID) -> None:
    """Run evaluation + gap-report stages, persist results, commit.

    Side effects (all in a single transaction):

    1. ``InterviewOutcomeEvaluation`` rows — one per outcome with a
       boolean verdict derived from the rubric aggregate.
    2. ``InterviewSession.internal_summary_json`` — gains ``total_score``,
       ``rubric_aggregated``, and ``pass_verdict``.
    3. ``GapReport`` row — student / teacher summary + ``report_json``.

    On exception: rollback, stamp ``internal_summary_json['evaluation_failure']``,
    commit, and re-raise.
    """
    session = await sessions_queries.get_session(db, session_id)
    if session is None:
        raise NotFoundError(f"Interview session {session_id} not found")

    try:
        outcomes = await authoring_queries.list_outcomes_for_config(db, session.interview_config_id)
        questions = await authoring_queries.list_questions_for_config(
            db, session.interview_config_id
        )
        candidate_answers = await _list_candidate_answers(db, session_id)

        rubric_scores = await evaluate_session(
            db,
            session=session,
            outcomes=outcomes,
            questions=questions,
            answers=candidate_answers,
            pipeline_run_id=None,
        )

        course_id, module_id = await _resolve_config_scope(db, session.interview_config_id)
        quiz_attempts = await _load_student_quiz_attempts(
            db, student_id=session.student_id, course_id=course_id, module_id=module_id
        )

        report_draft = await generate_gap_report(
            db,
            session=session,
            rubric_scores=rubric_scores,
            quiz_attempts=quiz_attempts,
            course_id=course_id,
            module_id=module_id,
            pipeline_run_id=None,
        )

        await _persist_outcome_evaluations(
            db, session_id=session_id, outcomes=outcomes, rubric_scores=rubric_scores
        )
        _stamp_session_summary(session, rubric_scores=rubric_scores)
        await _persist_gap_report(
            db,
            session=session,
            course_id=course_id,
            module_id=module_id,
            draft=report_draft,
        )

        await db.commit()
    except Exception as exc:
        await db.rollback()
        fresh = await db.get(InterviewSession, session_id)
        if fresh is not None:
            fresh.internal_summary_json = dict(fresh.internal_summary_json or {}) | {
                "evaluation_failure": {
                    "message": str(exc),
                    "failed_at": utcnow().isoformat(),
                }
            }
            await db.commit()
        raise


async def _list_candidate_answers(
    db: AsyncSession, session_id: UUID
) -> list[InterviewSessionMessage]:
    messages = await sessions_queries.list_session_messages(db, session_id)
    return [m for m in messages if getattr(m, "role", None) == "user"]


async def _resolve_config_scope(db: AsyncSession, config_id: UUID) -> tuple[UUID, UUID]:
    config = await authoring_queries.get_interview_for_authoring(db, config_id)
    if config is None:
        raise NotFoundError(f"Interview config {config_id} not found")
    return config.course_id, config.module_id


async def _load_student_quiz_attempts(
    db: AsyncSession,
    *,
    student_id: UUID,
    course_id: UUID,
    module_id: UUID | None,
) -> list[Any]:
    """Pull quiz_attempt scores via raw SQL.

    Cross-feature read into the ``quiz_attempts`` table — direct ORM
    import would break the ``Features are independent`` import-linter
    contract. The Gap Report stage consumes objects with a
    ``score_percent`` attribute (via :class:`_QuizAttemptLike`
    Protocol); raw rows fit that shape.
    """
    from sqlalchemy import text  # noqa: PLC0415

    sql = (
        "SELECT qa.score_percent FROM quiz_attempts qa "
        "JOIN quizzes q ON q.id = qa.quiz_id "
        "WHERE qa.student_id = :student_id "
        "  AND q.course_id = :course_id "
    )
    params: dict[str, Any] = {"student_id": student_id, "course_id": course_id}
    if module_id is not None:
        sql += "  AND q.module_id = :module_id "
        params["module_id"] = module_id

    rows = (await db.execute(text(sql), params)).mappings().all()
    return [_QuizAttemptRow(score_percent=row["score_percent"]) for row in rows]


class _QuizAttemptRow:
    """Lightweight stand-in for the Gap Report's ``_QuizAttemptLike`` Protocol."""

    __slots__ = ("score_percent",)

    def __init__(self, *, score_percent: Decimal | float | int | None) -> None:
        self.score_percent = score_percent


async def _persist_outcome_evaluations(
    db: AsyncSession,
    *,
    session_id: UUID,
    outcomes: Sequence[InterviewOutcome],
    rubric_scores: RubricScores,
) -> None:
    """Insert one :class:`InterviewOutcomeEvaluation` per outcome.

    Verdict heuristic: ``verdict_met = total_score >= 60``. The
    rubric stage scores 0-100; 60 is the conventional pass cutoff
    (matches the legacy ``interview_outcome_evaluations`` semantics).
    Per-outcome differentiation is intentionally simple here — the
    full LLM-driven per-outcome verdicts land in a future stage.
    """
    pass_threshold = 60.0
    verdict_met = rubric_scores.total_score >= pass_threshold
    for outcome in outcomes:
        db.add(
            InterviewOutcomeEvaluation(
                session_id=session_id,
                outcome_id=outcome.id,
                verdict_met=verdict_met,
                hidden_reasoning=_format_rubric_reasoning(rubric_scores),
                evidence_excerpt=None,
            )
        )
    await flush_or_conflict(db)


def _format_rubric_reasoning(rubric_scores: RubricScores) -> str:
    parts = [f"total={rubric_scores.total_score:.2f}"]
    for criterion, score in rubric_scores.aggregated.items():
        parts.append(f"{criterion}={score:.2f}")
    return ", ".join(parts)


def _stamp_session_summary(session: InterviewSession, *, rubric_scores: RubricScores) -> None:
    pass_threshold = 60.0
    summary: dict[str, Any] = dict(session.internal_summary_json or {})
    summary["total_score"] = float(rubric_scores.total_score)
    summary["rubric_aggregated"] = dict(rubric_scores.aggregated)
    summary["evaluated_at"] = utcnow().isoformat()
    session.internal_summary_json = summary
    session.pass_verdict = bool(rubric_scores.total_score >= pass_threshold)


async def _persist_gap_report(
    db: AsyncSession,
    *,
    session: InterviewSession,
    course_id: UUID,
    module_id: UUID | None,
    draft: GapReportDraft,
) -> None:
    db.add(
        GapReport(
            student_id=session.student_id,
            course_id=course_id,
            module_id=module_id,
            source_quiz_attempt_id=None,
            source_interview_session_id=session.id,
            student_summary=draft.student_summary,
            teacher_summary=draft.teacher_summary,
            report_json=dict(draft.report_json),
        )
    )
    await flush_or_conflict(db)


__all__ = ["evaluate_and_generate_report"]
