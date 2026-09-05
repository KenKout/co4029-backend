"""Persist one interview's gap report, idempotently and behind the output guard.

Split out of ``services.evaluation`` to keep that module under the 800-LOC gate
(``tests/integration/test_interviews_metric.py``). It is a self-contained write:
guard the learner-facing prose, then upsert exactly one row per session.

Two contracts live here:

* **Output guard** — the student summary and study-plan text are LLM-generated
  and learner-facing, so they pass through ``guard_student_output`` before REST
  can serialize them. The boundary is record-only by product decision: leakage is
  audited as a security event, but the feedback is never swapped for the fallback.
* **One report per session** — the writer is re-runnable (ARQ retries, the
  recovery sweep), so it updates in place when a report already exists. Migration
  0107 added a partial UNIQUE index on ``source_interview_session_id`` to make
  that a database guarantee rather than a read-then-write hope: two concurrent
  evaluations both used to see "no report" and both insert, and since readers take
  the newest row the stale duplicate stayed invisible while the data was wrong.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from abridgeai.core.db.conflict_mapper import flush_or_conflict
from abridgeai.features.interviews.models import GapReport, InterviewSessionQuestion
from abridgeai.features.interviews.orchestrator.security import (
    SecurityAction,
    SecurityAssessment,
    SecurityCategory,
)
from abridgeai.features.interviews.queries import sessions as sessions_queries
from abridgeai.features.interviews.services import security as security_service

if TYPE_CHECKING:
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession

    from abridgeai.features.interviews.ai.stages.gap_report import GapReportDraft
    from abridgeai.features.interviews.models import InterviewSession

_FALLBACK_SUMMARY = (
    "Your interview feedback could not be displayed safely. "
    "Please ask your instructor for learning guidance."
)


async def persist_gap_report(
    db: AsyncSession,
    *,
    session: InterviewSession,
    course_id: UUID,
    module_id: UUID | None,
    draft: GapReportDraft,
) -> None:
    """Write (or refresh) the single gap report for ``session``."""
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
        fallback_text=_FALLBACK_SUMMARY,
        allowed_question_ids=[qid for qid in asked_question_ids if qid is not None],
        assessment=assessment,
        action=SecurityAction.ALLOW,
        attempt_count=0,
        # See module docstring: record-only, never substituted.
        force_record_only=True,
    )
    report_json = dict(draft.report_json)
    student_summary = draft.student_summary
    if guarded.output_fallback_used and guarded.output_leakage_blocked:
        # Unreachable today (force_record_only above) — kept as a defensive
        # backstop in case the boundary is ever switched back to enforcing.
        student_summary = guarded.text
        report_json["study_plan"] = []
        report_json["strengths"] = []
        report_json["weaknesses"] = []

    existing = await sessions_queries.get_gap_report_for_session(db, session.id)
    if existing is not None:
        existing.course_id = course_id
        existing.module_id = module_id
        existing.student_summary = student_summary
        existing.teacher_summary = draft.teacher_summary
        existing.report_json = report_json
    else:
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


__all__ = ["persist_gap_report"]
