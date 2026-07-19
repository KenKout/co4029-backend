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

from decimal import Decimal
from typing import TYPE_CHECKING, Any
from uuid import UUID

from abridgeai.core.db.conflict_mapper import flush_or_conflict
from abridgeai.core.exceptions import NotFoundError
from abridgeai.core.security import utcnow
from abridgeai.features.interviews.ai.stages.evaluation import evaluate_outcomes, evaluate_session
from abridgeai.features.interviews.ai.stages.evaluation.outcome_verdicts import OutcomeVerdicts
from abridgeai.features.interviews.ai.stages.evaluation.rubric import RubricScores
from abridgeai.features.interviews.ai.stages.gap_report import (
    GapReportDraft,
    generate_gap_report,
)
from abridgeai.features.interviews.models import (
    GapReport,
    InterviewOutcomeEvaluation,
    InterviewSession,
    InterviewSessionMessage,
    InterviewSessionQuestion,
)
from abridgeai.features.interviews.orchestrator.security import (
    SecurityAction,
    SecurityAssessment,
    SecurityCategory,
)
from abridgeai.features.interviews.queries import authoring as authoring_queries
from abridgeai.features.interviews.queries import sessions as sessions_queries
from abridgeai.features.interviews.services import security as security_service

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


async def evaluate_and_generate_report(
    db: AsyncSession, session_id: UUID, *, is_final_attempt: bool = False
) -> None:
    """Run evaluation + gap-report stages, persist results, commit.

    Side effects (all in a single transaction):

    1. ``InterviewOutcomeEvaluation`` rows — one per outcome with its OWN
       met/not-met verdict + reasoning + evidence (thesis §4.3), NOT a copied
       session-total.
    2. ``InterviewSession.pass_verdict`` — derived from
       ``met_count >= min_outcomes_to_pass`` (NULL threshold → all outcomes
       must be met). ``internal_summary_json`` also gains the rubric
       ``total_score`` / ``rubric_aggregated`` for teacher diagnostics.
    3. ``GapReport`` row — student / teacher summary + ``report_json``.

    Parameters
    ----------
    is_final_attempt
        True when the caller (ARQ task wrapper) has exhausted
        ``WorkerSettings.max_tries`` on this job. When an exception hits
        on the final attempt, ``InterviewSession.status`` is stamped
        ``'failed'`` in addition to the ``evaluation_failure`` note so the
        student-facing poll (``course-interview.tsx``) can detect the
        terminal failure and stop waiting instead of polling forever
        for a ``pass_verdict`` that will never arrive.

    On exception: rollback, stamp ``internal_summary_json['evaluation_failure']``
    (plus ``status='failed'`` when ``is_final_attempt``), commit, and re-raise.
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

        # Thesis §4.3 gate: per-outcome met/not-met verdicts decide pass/fail.
        outcome_verdicts = await evaluate_outcomes(
            db,
            session=session,
            outcomes=outcomes,
            questions=questions,
            answers=candidate_answers,
            pipeline_run_id=None,
        )

        # Rubric stays as a teacher-facing diagnostic feeding the Gap Report;
        # it no longer gates pass/fail (phase-03).
        rubric_scores = await evaluate_session(
            db,
            session=session,
            outcomes=outcomes,
            questions=questions,
            answers=candidate_answers,
            pipeline_run_id=None,
        )

        config = await authoring_queries.get_interview_for_authoring(
            db, session.interview_config_id
        )
        if config is None:
            raise NotFoundError(f"Interview config {session.interview_config_id} not found")
        min_outcomes_to_pass = getattr(config, "min_outcomes_to_pass", None)
        course_id, module_id = config.course_id, config.module_id

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

        await _persist_outcome_evaluations(db, session_id=session_id, verdicts=outcome_verdicts)
        _stamp_session_summary(
            session,
            rubric_scores=rubric_scores,
            verdicts=outcome_verdicts,
            min_outcomes_to_pass=min_outcomes_to_pass,
        )
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
                    "final_attempt": is_final_attempt,
                }
            }
            # Only stamp the terminal 'failed' status once ARQ has exhausted
            # its retry budget. Marking it failed on attempt 1/3 would tell
            # the student the interview is dead while a retry is still
            # queued — but NOT stamping it on the LAST attempt leaves the
            # session stuck at 'completed' with pass_verdict forever null,
            # so the frontend poll in course-interview.tsx never resolves
            # and the student waits indefinitely (the bug we're fixing).
            if is_final_attempt:
                fresh.status = "failed"
            await db.commit()
        raise


async def _list_candidate_answers(
    db: AsyncSession, session_id: UUID
) -> list[InterviewSessionMessage]:
    messages = await sessions_queries.list_session_messages(db, session_id)
    return [m for m in messages if getattr(m, "role", None) == "user"]


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
    verdicts: OutcomeVerdicts,
) -> None:
    """Insert one :class:`InterviewOutcomeEvaluation` per outcome verdict.

    Each row carries that outcome's OWN met/not-met verdict, hidden reasoning,
    and evidence excerpt — the genuine per-outcome judgement from the §4.3
    verdict stage (not a copied session-total).
    """
    for verdict in verdicts.verdicts:
        db.add(
            InterviewOutcomeEvaluation(
                session_id=session_id,
                outcome_id=verdict.outcome_id,
                verdict_met=verdict.met,
                hidden_reasoning=verdict.reasoning,
                evidence_excerpt=verdict.evidence,
            )
        )
    await flush_or_conflict(db)


def _derive_pass_verdict(verdicts: OutcomeVerdicts, min_outcomes_to_pass: int | None) -> bool:
    """Pass when enough outcomes are met (thesis §4.3).

    ``min_outcomes_to_pass`` is the teacher-configured threshold. When it is
    NULL/unset we require EVERY outcome to be met — the documented-safe
    default (a teacher who configured no threshold has not opted into a
    partial pass). A session with no outcomes cannot pass.
    """
    if verdicts.total == 0:
        return False
    threshold = min_outcomes_to_pass if min_outcomes_to_pass is not None else verdicts.total
    return verdicts.met_count >= threshold


def _stamp_session_summary(
    session: InterviewSession,
    *,
    rubric_scores: RubricScores,
    verdicts: OutcomeVerdicts,
    min_outcomes_to_pass: int | None,
) -> None:
    summary: dict[str, Any] = dict(session.internal_summary_json or {})
    summary["total_score"] = float(rubric_scores.total_score)
    summary["rubric_aggregated"] = dict(rubric_scores.aggregated)
    summary["outcomes_met"] = verdicts.met_count
    summary["outcomes_total"] = verdicts.total
    summary["min_outcomes_to_pass"] = min_outcomes_to_pass
    summary["evaluated_at"] = utcnow().isoformat()
    session.internal_summary_json = summary
    session.pass_verdict = _derive_pass_verdict(verdicts, min_outcomes_to_pass)


async def _persist_gap_report(
    db: AsyncSession,
    *,
    session: InterviewSession,
    course_id: UUID,
    module_id: UUID | None,
    draft: GapReportDraft,
) -> None:
    # Gap-report prose is also learner-facing AI output. Guard the complete
    # learner projection (summary + generated study-plan text) before it can be
    # serialized by REST. Numeric rubric details remain teacher-only.
    from sqlalchemy import select  # noqa: PLC0415

    asked_question_ids = list(
        (
            await db.execute(
                select(InterviewSessionQuestion.interview_question_id).where(
                    InterviewSessionQuestion.session_id == session.id,
                    InterviewSessionQuestion.interview_question_id.is_not(None),
                )
            )
        )
        .scalars()
        .all()
    )
    learner_parts = [draft.student_summary]
    for item in draft.study_plan:
        learner_parts.extend((item.topic, item.weakness_summary))
    assessment = SecurityAssessment(
        category=SecurityCategory.BENIGN,
        detected=False,
        confidence=1.0,
        should_block=False,
        should_record_academic_evidence=False,
        response_key=None,
        normalized_fingerprint=None,
        source="gap_report_boundary",
    )
    guarded = await security_service.guard_student_output(
        db,
        session_id=session.id,
        config_id=session.interview_config_id,
        turn_key=f"gap-report:{session.id}",
        proposed_text="\n".join(part for part in learner_parts if part),
        fallback_text=(
            "Your interview feedback could not be displayed safely. "
            "Please ask your instructor for learning guidance."
        ),
        allowed_question_ids=[qid for qid in asked_question_ids if qid is not None],
        assessment=assessment,
        action=SecurityAction.ALLOW,
        attempt_count=0,
    )
    report_json = dict(draft.report_json)
    student_summary = draft.student_summary
    if guarded.output_fallback_used:
        student_summary = guarded.text
        report_json["study_plan"] = []
        report_json["strengths"] = []
        report_json["weaknesses"] = []
    db.add(
        GapReport(
            student_id=session.student_id,
            course_id=course_id,
            module_id=module_id,
            source_quiz_attempt_id=None,
            source_interview_session_id=session.id,
            student_summary=student_summary,
            teacher_summary=draft.teacher_summary,
            report_json=report_json,
        )
    )
    await flush_or_conflict(db)


__all__ = ["evaluate_and_generate_report"]
