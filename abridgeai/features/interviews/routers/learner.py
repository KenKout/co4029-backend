"""Interviews learner router (T6.12).

Seven endpoints (no router prefix; the legacy paths self-prefix under
``/interview-configs``, ``/interview-sessions``, and ``/me``). Composes
:mod:`features.interviews.services.taking` for the session lifecycle
and :mod:`features.interviews.queries.published` for the take payload.

Voice future-proofing
---------------------
``POST /interview-sessions/{session_id}/respond`` accepts an optional
``audio_storage_object_id`` on the body. The service layer stores it
on the :class:`InterviewSessionMessage` row but performs NO
transcription — the field is the forward-compat hook for the future
voice-mode build out (STT lands separately).

Security invariant
------------------
Every session-scoped endpoint depends on
:func:`features.interviews.routers._deps.require_session_owner_access`
which closes the perimeter at the HTTP boundary. The service layer
also raises :class:`ForbiddenError` on cross-user access — defence in
depth.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Annotated, Any, cast
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.core.db import get_db
from abridgeai.core.security import CurrentUser, get_current_user
from abridgeai.features.courses.api.public import can_view_course_content
from abridgeai.features.interviews.routers._deps import require_session_owner_access
from abridgeai.features.interviews.schemas import (
    GapReportRead,
    InterviewConfigPublic,
    InterviewForTakingPublic,
    InterviewProgressRead,
    InterviewQuestionPublic,
    InterviewSessionHistoryTurn,
    InterviewSessionPublic,
)
from abridgeai.features.interviews.services import (
    learner_progress as learner_progress_service,
)
from abridgeai.features.interviews.services import taking as taking_service

router = APIRouter(tags=["interviews-learner"])

if TYPE_CHECKING:
    from abridgeai.features.interviews.models import (
        InterviewConfig,
        InterviewQuestion,
        InterviewSession,
        InterviewSessionMessage,
        InterviewSessionQuestion,
    )

logger = logging.getLogger(__name__)

_REQUIRE_SESSION_OWNER = require_session_owner_access()


def _not_found(resource: str, resource_id: UUID) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail={"error": "not_found", "resource": resource, "id": str(resource_id)},
    )


def _bad_request(message: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail={"error": "bad_request", "message": message},
    )


def _conflict(message: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_409_CONFLICT,
        detail={"error": "conflict", "message": message},
    )


def _voice_unavailable() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail={"error": "voice_unavailable", "message": "Voice interviews are not enabled"},
    )


async def _ensure_config_course_enrolled(
    db: AsyncSession, current_user: CurrentUser, config: InterviewConfig
) -> None:
    """BR gate: an interview config is a course item — enrollment required.

    A published config is only reachable through a course's curriculum,
    and per BR an unenrolled student must not reach ANY course item. This
    404s (the feature's no-existence-leak convention) unless the caller
    has an ``active`` / ``completed`` enrollment in the owning course or
    holds course-management rights (owner/teacher preview).
    """
    from abridgeai.features.access_control.policies import (  # noqa: PLC0415
        can_manage_course,
    )
    from abridgeai.features.enrollments.api import public as enrollments_api  # noqa: PLC0415

    course_id = config.course_id
    if await enrollments_api.has_active_or_completed_enrollment(
        db, student_id=current_user.user_id, course_id=course_id
    ):
        return
    if await can_manage_course(db, current_user.user_id, course_id):
        return
    raise _not_found("interview_config", config.id)


async def get_arq_pool() -> object | None:
    return None


@router.get("/interview-configs/{config_id}", response_model=InterviewForTakingPublic)
async def get_interview_for_taking(
    config_id: UUID,
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> InterviewForTakingPublic:
    from sqlalchemy import func, select  # noqa: PLC0415

    from abridgeai.features.interviews.models import (  # noqa: PLC0415
        InterviewConfig,
        InterviewOutcome,
        InterviewQuestion,
        InterviewSession,
    )

    config = await db.get(InterviewConfig, config_id)
    if config is None or config.status != "published":
        raise _not_found("interview_config", config_id)
    await _ensure_config_course_enrolled(db, current_user, config)
    if config.max_attempts is not None and config.max_attempts > 0:
        used = (
            await db.execute(
                select(func.count(InterviewSession.id)).where(
                    InterviewSession.interview_config_id == config_id,
                    InterviewSession.student_id == current_user.user_id,
                )
            )
        ).scalar_one()
        if used >= config.max_attempts:
            raise _not_found("interview_config", config_id)

    # Just-in-time reveal: this learner request never materializes the full bank.
    #
    # Pinned to the graded partition. No session exists yet, so there is nothing
    # to read a mode off — and the assessment side is the one that preserves this
    # endpoint's behaviour exactly as it was before the split. A practice run
    # gets its first question from ``start_session``, which does know the mode.
    first_question = (
        await db.execute(
            select(InterviewQuestion)
            .where(
                InterviewQuestion.interview_config_id == config_id,
                InterviewQuestion.review_status == "approved",
            )
            .order_by(InterviewQuestion.position)
            .limit(1)
        )
    ).scalar_one_or_none()
    # Count-only signal: how many rubric criteria this interview assesses. We
    # count rows rather than fetch them so no outcome text / weight / threshold
    # is materialized in the learner path (see InterviewForTakingPublic docs).
    outcome_count = (
        await db.execute(
            select(func.count(InterviewOutcome.id)).where(
                InterviewOutcome.interview_config_id == config_id,
                InterviewOutcome.deleted_at.is_(None),
            )
        )
    ).scalar_one()
    return InterviewForTakingPublic(
        config=InterviewConfigPublic.model_validate(config),
        first_question=(
            InterviewQuestionPublic.model_validate(first_question)
            if first_question is not None
            else None
        ),
        outcome_count=int(outcome_count),
    )


@router.get("/interview-sessions/{session_id}/gap-report", response_model=GapReportRead)
async def get_gap_report(
    session_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_SESSION_OWNER)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> GapReportRead:
    del current_user
    from sqlalchemy import select  # noqa: PLC0415

    from abridgeai.features.interviews.models import GapReport  # noqa: PLC0415

    stmt = (
        select(GapReport)
        .where(GapReport.source_interview_session_id == session_id)
        .order_by(GapReport.created_at.desc())
        .limit(1)
    )
    report = (await db.execute(stmt)).scalar_one_or_none()
    if report is None:
        raise _not_found("gap_report", session_id)
    return await _gap_report_view(db, report)


@router.get("/me/interview-sessions", response_model=list[InterviewSessionPublic])
async def list_my_sessions(
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    config_id: Annotated[UUID | None, Query()] = None,
) -> list[InterviewSessionPublic]:
    from sqlalchemy import select  # noqa: PLC0415

    from abridgeai.features.interviews.models import InterviewConfig  # noqa: PLC0415

    sessions = await taking_service.get_user_sessions(
        db, current_user.user_id, interview_config_id=config_id
    )
    # Batch-fetch the parent config title + course for each distinct config so the
    # student history list is self-describing (the session row only carries the id).
    config_ids = {s.interview_config_id for s in sessions}
    config_meta: dict[UUID, tuple[str, UUID]] = {}
    if config_ids:
        rows = (
            await db.execute(
                select(
                    InterviewConfig.id,
                    InterviewConfig.title,
                    InterviewConfig.course_id,
                ).where(InterviewConfig.id.in_(config_ids))
            )
        ).all()
        config_meta = {row.id: (row.title, row.course_id) for row in rows}
    return [
        InterviewSessionPublic(
            session_id=s.id,
            interview_config_id=s.interview_config_id,
            interview_title=config_meta.get(s.interview_config_id, (None, None))[0],
            course_id=config_meta.get(s.interview_config_id, (None, None))[1],
            status=s.status,
            input_mode=s.input_mode,
            attempt_number=s.attempt_number,
            started_at=s.started_at,
            assessment_started_at=s.assessment_started_at,
            onboarding_stage=s.onboarding_stage,
            interview_language=s.interview_language,
            ended_at=s.ended_at,
            resume_deadline_at=s.resume_deadline_at,
            current_question_index=None,
            time_remaining_seconds=await taking_service.session_time_remaining_seconds(db, s),
            pass_verdict=s.pass_verdict,
        )
        for s in sessions
    ]


def _history_elapsed_seconds(
    assessment_started_at: datetime | None,
    created_at: datetime,
) -> int | None:
    if assessment_started_at is None or created_at < assessment_started_at:
        return None
    return max(0, int((created_at - assessment_started_at).total_seconds()))


def _history_message_kind(message: InterviewSessionMessage) -> str:
    metadata = message.metadata_json if isinstance(message.metadata_json, dict) else {}
    stored_kind = metadata.get("kind")
    if stored_kind in {"clarification", "term_explanation"}:
        return "clarification"
    if stored_kind == "hint":
        return "hint"
    # Standardized between-question transition turns (Natural Interview
    # Transitions spec) map to the existing "transition" history kind so the
    # transcript renders them as their own chronological AI turn.
    if stored_kind == "transition":
        return "transition"
    # Rich-closing sub-steps (Slice 13): self-reflection prompt, invite-candidate-
    # questions, and the answer-safe reply are NON-assessed ceremony that the
    # adaptive path persists attached to the last question. Without this they'd
    # fall through to "followup" below and render under "Question N", so classify
    # them as "closing" — the transcript then groups them in their own wrap-up
    # section, never under a numbered assessed question.
    if metadata.get("action") in {
        "prompt_self_reflection",
        "invite_candidate_questions",
        "answer_candidate_question",
    }:
        return "closing"
    if message.role == "user":
        return "answer"
    ceremony_key = metadata.get("ceremony_key")
    if ceremony_key == "briefing":
        return "briefing"
    if ceremony_key == "ready_transition":
        return "transition"
    if ceremony_key == "closing":
        return "closing"
    if message.session_question_id is not None:
        return "followup"
    return "opening"


def _build_session_history(  # noqa: C901 -- explicit transcript merge is easier to audit
    session: InterviewSession,
    question_rows: list[tuple[InterviewSessionQuestion, InterviewQuestion | None]],
    messages: list[InterviewSessionMessage],
) -> list[InterviewSessionHistoryTurn]:
    """Merge persisted ceremony/messages with every revealed question.

    The ``ready_transition`` handoff is deliberately withheld. Its row still exists
    — ``orchestration_bridge`` reads it as the marker that the REST ceremony already
    introduced the interviewer — but nobody ever SAYS it: the agent owns its own
    opening and speaks the question itself. Replaying it here put a line the
    candidate never heard back on screen every time the page reloaded mid-session.
    """
    history: list[InterviewSessionHistoryTurn] = []
    visible_messages = [
        message
        for message in messages
        if message.role in {"ai", "user"}
        and bool((message.content_text or "").strip())
        and (message.metadata_json or {}).get("ceremony_key") != "ready_transition"
    ]

    def append_message(message: InterviewSessionMessage) -> None:
        kind = _history_message_kind(message)
        history.append(
            InterviewSessionHistoryTurn(
                id=f"message:{message.id}",
                role=cast(Any, message.role),
                content_text=message.content_text,
                kind=cast(Any, kind),
                created_at=message.created_at,
                elapsed_seconds=_history_elapsed_seconds(
                    session.assessment_started_at,
                    message.created_at,
                ),
                is_follow_up=kind == "followup",
            )
        )

    for message in visible_messages:
        if message.session_question_id is None:
            append_message(message)

    messages_by_question: dict[UUID, list[InterviewSessionMessage]] = {}
    for message in visible_messages:
        if message.session_question_id is not None:
            messages_by_question.setdefault(message.session_question_id, []).append(message)

    included_question_ids: set[UUID] = set()
    if session.onboarding_stage == "completed":
        for session_question, question in question_rows:
            included_question_ids.add(session_question.id)
            question_created_at = session_question.asked_at
            if session_question.sequence_no == 1 and session.assessment_started_at is not None:
                question_created_at = session.assessment_started_at
            # The interviewer's own wording is the transcript's source of truth:
            # the native agent paraphrases every question, and replaying the
            # bank's exact prompt after a reload showed words the candidate
            # never heard. The canonical `question:*` identity and metadata ride
            # ON the spoken row; the bank prompt only stands in when the agent's
            # wording was never recorded (routed/REST sessions).
            spoken_messages = messages_by_question.get(session_question.id, [])
            answer_seen = False
            spoken_used = False
            # Without a spoken question (routed/REST sessions, or the spoken row
            # was never recorded) the bank prompt stands in — placed FIRST, the
            # position the question occupied in the live conversation.
            bank_fallback = not any(
                message.role == "ai"
                and (message.metadata_json or {}).get("kind") == "question"
                and (message.content_text or "").strip()
                for message in spoken_messages
            )
            if bank_fallback and question is not None and (question.prompt_text or "").strip():
                history.append(
                    InterviewSessionHistoryTurn(
                        id=f"question:{session_question.id}",
                        role="ai",
                        content_text=question.prompt_text,
                        kind="question",
                        created_at=question_created_at,
                        elapsed_seconds=_history_elapsed_seconds(
                            session.assessment_started_at,
                            question_created_at,
                        ),
                        question_type=question.question_type,
                    )
                )
            for message in spoken_messages:
                if message.role == "user":
                    answer_seen = True
                if (
                    not answer_seen
                    and message.role == "ai"
                    and (message.metadata_json or {}).get("kind") == "question"
                ):
                    if spoken_used:
                        continue
                    spoken_used = True
                    history.append(
                        InterviewSessionHistoryTurn(
                            id=f"question:{session_question.id}",
                            role="ai",
                            content_text=message.content_text,
                            kind="question",
                            created_at=message.created_at or question_created_at,
                            elapsed_seconds=_history_elapsed_seconds(
                                session.assessment_started_at,
                                message.created_at or question_created_at,
                            ),
                            question_type=(
                                question.question_type
                                if question is not None
                                else None
                            ),
                        )
                    )
                    continue
                append_message(message)

    # Defensive recovery for messages whose question was removed after the
    # interview. The prompt may no longer exist, but the candidate's own turns
    # still belong in their history.
    for question_id, question_messages in messages_by_question.items():
        if question_id in included_question_ids:
            continue
        for message in question_messages:
            append_message(message)

    return history


async def _session_history(
    db: AsyncSession,
    session: InterviewSession,
) -> list[InterviewSessionHistoryTurn]:
    from sqlalchemy import select  # noqa: PLC0415

    from abridgeai.features.interviews.models import (  # noqa: PLC0415
        InterviewQuestion,
        InterviewSessionQuestion,
    )
    from abridgeai.features.interviews.queries import sessions as sessions_queries  # noqa: PLC0415

    rows = (
        await db.execute(
            select(InterviewSessionQuestion, InterviewQuestion)
            .outerjoin(
                InterviewQuestion,
                InterviewQuestion.id == InterviewSessionQuestion.interview_question_id,
            )
            .where(InterviewSessionQuestion.session_id == session.id)
            .order_by(InterviewSessionQuestion.sequence_no)
        )
    ).all()
    question_rows = [(row[0], row[1]) for row in rows]
    messages = await sessions_queries.list_session_messages(db, session.id)
    return _build_session_history(session, question_rows, messages)


async def _current_session_question(db: AsyncSession, session_id: UUID) -> object | None:
    """Return the latest revealed question so an active attempt resumes in place."""
    from sqlalchemy import select  # noqa: PLC0415

    from abridgeai.features.interviews.models import (  # noqa: PLC0415
        InterviewQuestion,
        InterviewSessionQuestion,
    )

    sq_stmt = (
        select(InterviewSessionQuestion)
        .where(InterviewSessionQuestion.session_id == session_id)
        .order_by(InterviewSessionQuestion.sequence_no.desc())
        .limit(1)
    )
    sq = (await db.execute(sq_stmt)).scalar_one_or_none()
    if sq is None or sq.interview_question_id is None:
        return None
    return await db.get(InterviewQuestion, sq.interview_question_id)


async def _resolve_resource_titles(
    db: AsyncSession,
    resource_ids: set[str],
) -> dict[str, str]:
    """Map resource UUID → human title for study-plan display.

    The gap report persists only resource UUIDs in ``report_json``; resolving
    them to titles at projection time repairs existing reports WITHOUT a data
    migration. IDs that no longer resolve (deleted resource, malformed id) are
    simply omitted so the UI never shows a raw UUID.
    """
    if not resource_ids:
        return {}
    from sqlalchemy import bindparam  # noqa: PLC0415
    from sqlalchemy import text as _text  # noqa: PLC0415

    valid: list[UUID] = []
    for rid in resource_ids:
        try:
            valid.append(UUID(str(rid)))
        except (ValueError, AttributeError, TypeError):
            continue
    if not valid:
        return {}
    stmt = _text(
        "SELECT id, title FROM lesson_resources WHERE id IN :ids AND deleted_at IS NULL"
    ).bindparams(bindparam("ids", expanding=True))
    rows = (await db.execute(stmt, {"ids": valid})).mappings().all()
    return {str(row["id"]): row["title"] for row in rows}


def _study_plan_from_report(report_json: object) -> list[dict[str, Any]]:
    """Extract the raw study-plan entries (UUID resources) from report_json."""
    raw_plan = report_json.get("study_plan") if isinstance(report_json, dict) else None
    study_plan: list[dict[str, Any]] = []
    if isinstance(raw_plan, list):
        for entry in raw_plan:
            if not isinstance(entry, dict):
                continue
            study_plan.append(
                {
                    "topic": entry.get("topic", ""),
                    "lesson_id": entry.get("suggested_lesson_id"),
                    "suggested_resources": [
                        str(rid) for rid in entry.get("suggested_resource_ids", []) or []
                    ],
                }
            )
    return study_plan


def _apply_resource_titles(
    study_plan: list[dict[str, Any]],
    titles: dict[str, str],
) -> None:
    """Replace each item's UUID resource list with resolved human titles.

    Mutates in place. Unresolvable ids are dropped rather than shown raw, so a
    study-plan item may end up with an empty resource list (the UI already
    hides the resource line when empty).
    """
    for item in study_plan:
        item["suggested_resources"] = [
            titles[rid] for rid in item["suggested_resources"] if rid in titles
        ]


async def _gap_report_view(db: AsyncSession, report: Any) -> GapReportRead:  # noqa: ANN401  -- ORM row, typed via duck shape
    report_json = report.report_json or {}
    study_plan = _study_plan_from_report(report_json)
    resource_ids = {rid for item in study_plan for rid in item["suggested_resources"]}
    _apply_resource_titles(study_plan, await _resolve_resource_titles(db, resource_ids))
    # FR-5.7: student sees pass/fail + qualitative remediation only. The
    # per-criterion mean rubric scores (report_json["rubric_aggregated"]) are
    # deliberately NOT projected here — they are teacher-only via
    # GapReportAuthoringRead. See interviews/schemas/report.py.
    summary_text = report.student_summary or ""
    return GapReportRead.model_validate(
        {
            "id": report.id,
            "student_id": report.student_id,
            "course_id": report.course_id,
            "module_id": report.module_id,
            "discrepancy_summary": summary_text or None,
            "study_plan": study_plan,
            "generated_at": report.created_at,
        }
    )


@router.get(
    "/courses/{course_id}/interview-progress",
    response_model=list[InterviewProgressRead],
)
async def list_my_interview_progress(
    course_id: UUID,
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[InterviewProgressRead]:
    """Per-interview completion state for the calling student in a course.

    Feeds the course-learn curriculum, which had no completion signal for
    interview items at all — they stayed pending forever and a module holding
    one could never auto-collapse, even after the student passed.

    Completed ⟺ at least one non-practice attempt has ``pass_verdict = TRUE``.
    Deliberately stricter than the quiz rule (which also completes on
    failed-and-exhausted): the tag reads as *passed*. See
    :class:`InterviewProgressRead` for the field semantics.

    Gated by :func:`can_view_course_content` — the same org/enrollment
    perimeter the quiz-progress endpoint uses — so a cross-tenant caller gets
    404 with no existence leak.
    """
    if not await can_view_course_content(db, user_id=current_user.user_id, course_id=course_id):
        raise _not_found("course", course_id)
    rows = await learner_progress_service.list_my_interview_progress(
        db, course_id=course_id, user_id=current_user.user_id
    )
    return [InterviewProgressRead.model_validate(r) for r in rows]


__all__ = ["get_arq_pool", "router"]
