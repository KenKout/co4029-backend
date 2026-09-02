"""Interview authoring: session views + session transcript / integrity / gap-report routes (T6.12).

Extracted from ``routers.authoring`` (2026-09-01) to keep that router under
the interviews LOC ratchet. Same ``/teacher`` prefix; view helpers are shared
back with ``routers.authoring`` which imports them from here.
"""

from __future__ import annotations

from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.core.db import get_db
from abridgeai.core.security import CurrentUser
from abridgeai.features.interviews.routers._deps import (
    _not_found,
    require_interview_authoring_access,
    require_session_authoring_access,
)
from abridgeai.features.interviews.schemas import (
    GapReportAuthoringRead,
    GapReportNotesUpdate,
    InterviewIntegrityEvent,
    InterviewIntegrityRead,
    InterviewSessionPublic,
    InterviewSessionSummary,
    InterviewSessionTeacherRead,
    InterviewTranscriptRead,
    InterviewTranscriptTurn,
    SecuritySessionSummary,
)

router = APIRouter(prefix="/teacher", tags=["interviews-authoring"])

_REQUIRE_CONFIG = require_interview_authoring_access()
_REQUIRE_SESSION_AUTHORING = require_session_authoring_access()


def _session_teacher_view(
    session: Any,  # noqa: ANN401  -- ORM row
    config_title: str,
    student_name: str | None,
    security_summary: SecuritySessionSummary | None = None,
) -> InterviewSessionTeacherRead:
    return InterviewSessionTeacherRead(
        session_id=session.id,
        interview_config_id=session.interview_config_id,
        interview_config_title=config_title,
        student_id=session.student_id,
        student_name=student_name,
        attempt_number=session.attempt_number,
        status=session.status,
        input_mode=session.input_mode,
        pass_verdict=session.pass_verdict,
        started_at=session.started_at,
        assessment_started_at=session.assessment_started_at,
        onboarding_stage=session.onboarding_stage,
        interview_language=session.interview_language,
        ended_at=session.ended_at,
        security_summary=security_summary,
    )


async def _security_summary_view(
    db: AsyncSession,
    session: Any,  # noqa: ANN401 -- ORM row
    *,
    enabled: bool,
) -> SecuritySessionSummary | None:
    if not enabled:
        return None
    from abridgeai.features.interviews.services import security as _security  # noqa: PLC0415

    metrics = await _security.get_security_session_metrics(db, session.id)
    return SecuritySessionSummary(
        assessment_count=metrics.assessment_count,
        blocked_attempt_count=metrics.blocked_attempt_count,
        repeated_attempt_count=metrics.repeated_attempt_count,
        output_leakage_prevented=metrics.output_leakage_prevented,
        security_fallback_rate=metrics.security_fallback_rate,
        average_classification_latency_ms=metrics.average_classification_latency_ms,
        session_flagged=bool(session.session_security_flagged),
    )


@router.get(
    "/interview-configs/{config_id}/sessions",
    response_model=list[InterviewSessionSummary],
)
async def list_config_sessions(
    config_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_CONFIG)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[InterviewSessionSummary]:
    """Teacher's per-config attempts list (thesis p77 review surface)."""
    del current_user
    from sqlalchemy import text as _text  # noqa: PLC0415

    from abridgeai.features.interviews.models import InterviewConfig  # noqa: PLC0415
    from abridgeai.features.interviews.queries import sessions as _sessions_q  # noqa: PLC0415

    sessions = await _sessions_q.list_sessions_for_config(db, config_id)
    config = await db.get(InterviewConfig, config_id)
    summaries = {
        s.id: await _security_summary_view(
            db,
            s,
            enabled=bool(config and config.security_incident_summary_enabled),
        )
        for s in sessions
    }
    student_ids = {s.student_id for s in sessions}
    names: dict[UUID, str] = {}
    if student_ids:
        rows = (
            (
                await db.execute(
                    _text(
                        "SELECT u.id, COALESCE(p.display_name, u.primary_email) AS name "
                        "FROM users u "
                        "LEFT JOIN user_profiles p ON p.user_id = u.id "
                        "WHERE u.id = ANY(:ids)"
                    ),
                    {"ids": list(student_ids)},
                )
            )
            .mappings()
            .all()
        )
        names = {row["id"]: row["name"] for row in rows}
    return [
        InterviewSessionSummary(
            session_id=s.id,
            student_id=s.student_id,
            student_name=names.get(s.student_id),
            attempt_number=s.attempt_number,
            status=s.status,
            input_mode=s.input_mode,
            pass_verdict=s.pass_verdict,
            started_at=s.started_at,
            ended_at=s.ended_at,
            security_summary=summaries[s.id],
        )
        for s in sessions
    ]


@router.get(
    "/interview-sessions/{session_id}",
    response_model=InterviewSessionPublic,
)
async def get_session_authoring(
    session_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_SESSION_AUTHORING)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> InterviewSessionPublic:
    """Teacher-scoped session detail (course-owner access via
    ``require_session_authoring_access``).

    Mirrors the learner-side ``GET /interview-sessions/{id}`` (student-
    owner-only), which teachers cannot call. The frontend gap-report page
    was hitting that student endpoint and getting a 403 on every load.
    """
    del current_user
    from abridgeai.features.interviews.queries import sessions as _sessions_q  # noqa: PLC0415

    session = await _sessions_q.get_session(db, session_id)
    if session is None:
        raise _not_found("interview_session", session_id)
    return InterviewSessionPublic(
        session_id=session.id,
        interview_config_id=session.interview_config_id,
        status=session.status,
        input_mode=session.input_mode,
        attempt_number=session.attempt_number,
        started_at=session.started_at,
        ended_at=session.ended_at,
        resume_deadline_at=session.resume_deadline_at,
        current_question_index=None,
        time_remaining_seconds=None,
        pass_verdict=session.pass_verdict,
    )


@router.get(
    "/interview-sessions/{session_id}/transcript",
    response_model=InterviewTranscriptRead,
)
async def get_session_transcript(
    session_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_SESSION_AUTHORING)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> InterviewTranscriptRead:
    """Full ordered Q&A transcript for teacher remediation review (thesis p77)."""
    del current_user
    from abridgeai.features.interviews.models import (  # noqa: PLC0415
        InterviewQuestion,
        InterviewSessionQuestion,
    )
    from abridgeai.features.interviews.queries import sessions as _sessions_q  # noqa: PLC0415

    messages = await _sessions_q.list_session_messages(db, session_id)
    # Resolve each message's question prompt via session_question_id -> question.
    prompts: dict[UUID, str] = {}
    sq_ids = {m.session_question_id for m in messages if m.session_question_id is not None}
    if sq_ids:
        from sqlalchemy import select as _select  # noqa: PLC0415

        rows = (
            await db.execute(
                _select(InterviewSessionQuestion.id, InterviewQuestion.prompt_text)
                .join(
                    InterviewQuestion,
                    InterviewQuestion.id == InterviewSessionQuestion.interview_question_id,
                )
                .where(InterviewSessionQuestion.id.in_(sq_ids))
            )
        ).all()
        prompts = {row[0]: row[1] for row in rows}

    turns = [
        InterviewTranscriptTurn(
            role=m.role,
            question_prompt=prompts.get(m.session_question_id)
            if m.session_question_id is not None
            else None,
            content_text=m.content_text,
            has_audio=m.audio_object_id is not None,
            created_at=m.created_at,
        )
        for m in messages
    ]
    return InterviewTranscriptRead(session_id=session_id, turns=turns)


@router.get(
    "/interview-sessions/{session_id}/integrity-events",
    response_model=InterviewIntegrityRead,
)
async def get_session_integrity_events(
    session_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_SESSION_AUTHORING)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> InterviewIntegrityRead:
    """FR-5.8 proctoring timeline for teacher post-session integrity review.

    Returns the session's ``assessment_integrity_events`` (focus_lost /
    tab_switch / fullscreen_exit / reconnect / disconnect), oldest first.
    Teacher-only (course-scoped authoring access); never exposed to students.
    """
    del current_user
    from abridgeai.features.interviews.queries import sessions as _sessions_q  # noqa: PLC0415

    rows = await _sessions_q.list_integrity_events_for_session(db, session_id)
    return InterviewIntegrityRead(
        session_id=session_id,
        events=[InterviewIntegrityEvent.model_validate(ev) for ev in rows],
    )


@router.get(
    "/interview-sessions/{session_id}/gap-report",
    response_model=GapReportAuthoringRead,
)
async def get_session_gap_report_authoring(
    session_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_SESSION_AUTHORING)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> GapReportAuthoringRead:
    del current_user
    from sqlalchemy import select  # noqa: PLC0415

    from abridgeai.features.interviews.models import (  # noqa: PLC0415
        GapReport,
        InterviewOutcomeEvaluation,
    )

    report_stmt = (
        select(GapReport)
        .where(GapReport.source_interview_session_id == session_id)
        .order_by(GapReport.created_at.desc())
        .limit(1)
    )
    report = (await db.execute(report_stmt)).scalar_one_or_none()
    if report is None:
        raise _not_found("gap_report", session_id)

    eval_stmt = select(InterviewOutcomeEvaluation).where(
        InterviewOutcomeEvaluation.session_id == session_id
    )
    evaluations = list((await db.execute(eval_stmt)).scalars().all())
    raw_evaluation_json: dict[str, Any] = {
        "outcome_evaluations": [
            {
                "id": str(e.id),
                "outcome_id": str(e.outcome_id),
                "verdict_met": e.verdict_met,
                "hidden_reasoning": e.hidden_reasoning,
                "evidence_excerpt": e.evidence_excerpt,
            }
            for e in evaluations
        ]
    }

    # ``GapReport`` (the ORM row) has neither ``generated_at`` (it's
    # ``created_at`` via TimestampMixin) nor ``study_plan``/
    # ``per_criterion_breakdown`` (those live inside ``report_json``
    # JSONB) — model_validate(report) directly would 500 on the
    # required ``generated_at`` field and silently default study_plan
    # to []. Build the DTO explicitly instead, mirroring the student
    # projection (_gap_report_view) plus the teacher-only fields.
    from abridgeai.features.interviews.routers.learner import (  # noqa: PLC0415
        _apply_resource_titles,
        _resolve_resource_titles,
        _study_plan_from_report,
    )

    report_json = report.report_json or {}
    study_plan = _study_plan_from_report(report_json)
    # Resolve resource UUIDs → human titles so the teacher study plan shows real
    # resource names instead of a wall of hex (mirrors the student projection).
    resource_ids = {rid for item in study_plan for rid in item["suggested_resources"]}
    _apply_resource_titles(study_plan, await _resolve_resource_titles(db, resource_ids))
    # FR-5.7: per-criterion mean rubric scores are teacher-only. They live
    # in ``report_json["rubric_aggregated"]`` and are surfaced here (never on
    # the student-facing GapReportRead).
    per_criterion = report_json.get("rubric_aggregated") if isinstance(report_json, dict) else None

    # Resolve human-readable context so the teacher view isn't a wall of UUIDs:
    # the student's display name (falling back to email) and the interview
    # config title. Both are read-only projections.
    from sqlalchemy import text as _text  # noqa: PLC0415

    from abridgeai.features.interviews.models import (  # noqa: PLC0415
        InterviewConfig,
        InterviewSession,
    )

    name_row = (
        (
            await db.execute(
                _text(
                    "SELECT COALESCE(p.display_name, u.primary_email) AS name "
                    "FROM users u "
                    "LEFT JOIN user_profiles p ON p.user_id = u.id "
                    "WHERE u.id = :id"
                ),
                {"id": report.student_id},
            )
        )
        .mappings()
        .first()
    )
    student_name = name_row["name"] if name_row else None

    interview_title: str | None = None
    score_summary: dict[str, Any] = {}
    rubric_weights: dict[str, float] = {}
    persona_adherence: dict[str, Any] = {}
    session_row = await db.get(InterviewSession, session_id)
    if session_row is not None:
        config_row = await db.get(InterviewConfig, session_row.interview_config_id)
        interview_title = config_row.title if config_row is not None else None
        # Quantitative rollup lives in internal_summary_json (teacher-only). Project
        # the numbers that contextualize the per-criterion means: weighted total,
        # outcomes met/total, answered/total/unanswered question counts.
        summary_json = session_row.internal_summary_json or {}
        if isinstance(summary_json, dict):
            score_summary = {
                key: summary_json[key]
                for key in (
                    "total_score",
                    "outcomes_met",
                    "outcomes_total",
                    "questions_total",
                    "questions_answered",
                    "questions_unanswered",
                )
                if key in summary_json
            }
            # Tone-only persona-adherence audit (teacher-only). Absent for
            # sessions evaluated before this shipped or never audited.
            audit = summary_json.get("persona_adherence")
            if isinstance(audit, dict):
                persona_adherence = audit
        # Resolve the per-criterion rubric weights so the teacher sees each
        # criterion's contribution to the weighted total.
        if config_row is not None:
            from abridgeai.features.interviews.ai.stages.evaluation.rubric import (  # noqa: PLC0415
                resolve_rubric_definition,
            )

            # Read the SAME source the grading stage reads
            # (``supplementary_instructions``), so the weights a teacher sees
            # here are the weights their session was actually graded with.
            # This previously probed a non-existent ``config_json`` attribute,
            # which always resolved to None and therefore always displayed the
            # default equal weights regardless of the configured rubric.
            rubric_weights = resolve_rubric_definition(
                config_row.supplementary_instructions
            ).weights

    # Qualitative per-criterion notes (criterion-tagged bullet phrases) already
    # live in report_json; surface them so the teacher sees the "why" per criterion.
    strengths = report_json.get("strengths") if isinstance(report_json, dict) else None
    weaknesses = report_json.get("weaknesses") if isinstance(report_json, dict) else None

    return GapReportAuthoringRead.model_validate(
        {
            "id": report.id,
            "student_id": report.student_id,
            "course_id": report.course_id,
            "module_id": report.module_id,
            "discrepancy_summary": report.student_summary or None,
            "study_plan": study_plan,
            "generated_at": report.created_at,
            "per_criterion_breakdown": (per_criterion if isinstance(per_criterion, dict) else {}),
            "strengths": [str(s) for s in strengths] if isinstance(strengths, list) else [],
            "weaknesses": [str(w) for w in weaknesses] if isinstance(weaknesses, list) else [],
            "score_summary": score_summary,
            "rubric_weights": rubric_weights,
            "persona_adherence": persona_adherence,
            "raw_evaluation_json": raw_evaluation_json,
            "teacher_summary": report.teacher_summary,
            "source_quiz_attempt_id": report.source_quiz_attempt_id,
            "source_interview_session_id": report.source_interview_session_id,
            "student_name": student_name,
            "interview_title": interview_title,
        }
    )


@router.patch(
    "/interview-sessions/{session_id}/gap-report/notes",
    response_model=GapReportAuthoringRead,
)
async def update_session_gap_report_notes(
    session_id: UUID,
    payload: GapReportNotesUpdate,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_SESSION_AUTHORING)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> GapReportAuthoringRead:
    """Persist the teacher-authored note (``teacher_summary``) on the report.

    Course-scoped teacher access only (same gate as the GET). Empty/blank input
    clears the note. Returns the full refreshed authoring projection so the
    client re-renders with the saved value.
    """
    from sqlalchemy import select  # noqa: PLC0415

    from abridgeai.features.interviews.models import GapReport  # noqa: PLC0415

    report_stmt = (
        select(GapReport)
        .where(GapReport.source_interview_session_id == session_id)
        .order_by(GapReport.created_at.desc())
        .limit(1)
    )
    report = (await db.execute(report_stmt)).scalar_one_or_none()
    if report is None:
        raise _not_found("gap_report", session_id)

    cleaned = (payload.teacher_summary or "").strip()
    report.teacher_summary = cleaned or None
    await db.commit()

    return await get_session_gap_report_authoring(
        session_id=session_id, current_user=current_user, db=db
    )


__all__ = [
    "_REQUIRE_CONFIG",
    "_REQUIRE_SESSION_AUTHORING",
    "_security_summary_view",
    "_session_teacher_view",
    "router",
]
