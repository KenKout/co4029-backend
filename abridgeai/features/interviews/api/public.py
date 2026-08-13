"""Public, typed cross-feature read API for the interviews feature.

Sibling features (progress dashboards, admin reports) MUST import from
this module rather than reaching into ``models``/``queries``/``services``
directly. The session-summary surface roll-ups outcome evaluations so
consumers do not need to walk the rubric tables themselves.
"""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.features.interviews.models import (
    InterviewConfig,
    InterviewOutcome,
    InterviewOutcomeEvaluation,
    InterviewQuestion,
    InterviewSession,
)

from ._dto import SessionSummaryDTO


async def get_session_summary(
    db: AsyncSession,
    session_id: UUID,
) -> SessionSummaryDTO | None:
    """Return a typed summary of a single interview session.

    Returns ``None`` when the session does not exist. ``outcomes_total``
    counts every persisted evaluation row for the session;
    ``outcomes_met`` counts the subset where ``verdict_met = TRUE``.
    Both default to 0 for an in-progress session with no evaluations
    written yet.
    """
    session_row = await db.get(InterviewSession, session_id)
    if session_row is None:
        return None

    counts_stmt = select(
        func.count(InterviewOutcomeEvaluation.id),
        func.count(InterviewOutcomeEvaluation.id).filter(
            InterviewOutcomeEvaluation.verdict_met.is_(True)
        ),
    ).where(InterviewOutcomeEvaluation.session_id == session_id)
    total, met = (await db.execute(counts_stmt)).one()

    return SessionSummaryDTO(
        id=session_row.id,
        interview_config_id=session_row.interview_config_id,
        student_id=session_row.student_id,
        attempt_number=session_row.attempt_number,
        status=session_row.status,
        input_mode=session_row.input_mode,
        started_at=session_row.started_at,
        ended_at=session_row.ended_at,
        pass_verdict=session_row.pass_verdict,
        outcomes_total=int(total or 0),
        outcomes_met=int(met or 0),
    )


async def deep_clone_interview_config(
    db: AsyncSession,
    *,
    source_config_id: UUID,
    target_module_id: UUID,
    actor_id: UUID,
    title_suffix: str = "",
) -> UUID:
    """Deep-clone an interview config (+ outcomes + questions) into a module.

    Cross-feature entry point for the courses module/item duplicate flow.

    The clone is always ``status='draft'`` with ``published_at`` cleared, and
    every cloned question is forced to ``review_status='pending'`` — a
    duplicate re-enters the review queue and is never published on copy.

    Outcomes are cloned first so their new ids can be mapped; each cloned
    question's ``linked_outcome_id`` is rewritten to point at the corresponding
    cloned outcome (dangling links to a foreign outcome would violate the
    rubric's intent). ``embedding`` is deliberately dropped (set NULL) so the
    dedup shortlist re-embeds the copy rather than trusting a stale vector.

    NOT copied: sessions, evaluations, security/runtime rows, bank items.

    Returns the new config id. Flushes but does not commit.
    """
    source = (
        await db.execute(select(InterviewConfig).where(InterviewConfig.id == source_config_id))
    ).scalar_one_or_none()
    if source is None:
        raise ValueError(f"InterviewConfig {source_config_id} not found")

    clone = InterviewConfig(
        course_id=source.course_id,
        module_id=target_module_id,
        title=f"{source.title}{title_suffix}",
        status="draft",
        max_attempts=source.max_attempts,
        cooldown_hours=source.cooldown_hours,
        min_outcomes_to_pass=source.min_outcomes_to_pass,
        lock_quiz_ef_until_pass=source.lock_quiz_ef_until_pass,
        time_limit_minutes=source.time_limit_minutes,
        persona=source.persona,
        persona_profile_json=(
            dict(source.persona_profile_json)
            if source.persona_profile_json is not None
            else None
        ),
        tts_voice=source.tts_voice,
        supported_modes=source.supported_modes,
        supplementary_instructions=source.supplementary_instructions,
        security_response_policy=source.security_response_policy,
        security_max_consecutive_attempts=source.security_max_consecutive_attempts,
        security_custom_refusal_en=source.security_custom_refusal_en,
        security_custom_refusal_vi=source.security_custom_refusal_vi,
        security_incident_summary_enabled=source.security_incident_summary_enabled,
        published_at=None,
        created_by=actor_id,
        updated_by=actor_id,
    )
    db.add(clone)
    await db.flush()

    # Outcomes first — build old->new id map for question relink.
    outcomes = (
        (
            await db.execute(
                select(InterviewOutcome)
                .where(InterviewOutcome.interview_config_id == source_config_id)
                .where(InterviewOutcome.deleted_at.is_(None))
                .order_by(InterviewOutcome.position)
            )
        )
        .scalars()
        .all()
    )
    outcome_id_map: dict[UUID, UUID] = {}
    for src_o in outcomes:
        o_clone = InterviewOutcome(
            interview_config_id=clone.id,
            position=src_o.position,
            outcome_text=src_o.outcome_text,
            outcome_type=src_o.outcome_type,
            importance_weight=src_o.importance_weight,
            created_by=actor_id,
            updated_by=actor_id,
        )
        db.add(o_clone)
        await db.flush()
        outcome_id_map[src_o.id] = o_clone.id

    questions = (
        (
            await db.execute(
                select(InterviewQuestion)
                .where(InterviewQuestion.interview_config_id == source_config_id)
                .where(InterviewQuestion.deleted_at.is_(None))
                .order_by(InterviewQuestion.position)
            )
        )
        .scalars()
        .all()
    )
    for src_q in questions:
        db.add(
            InterviewQuestion(
                interview_config_id=clone.id,
                linked_outcome_id=(
                    outcome_id_map.get(src_q.linked_outcome_id)
                    if src_q.linked_outcome_id is not None
                    else None
                ),
                position=src_q.position,
                question_type=src_q.question_type,
                prompt_text=src_q.prompt_text,
                difficulty=src_q.difficulty,
                embedding=None,
                model_answer=src_q.model_answer,
                review_status="pending",
                ai_generated=src_q.ai_generated,
                source_refs_json=list(src_q.source_refs_json or []),
                source_module_ids=list(src_q.source_module_ids or []),
                created_by=actor_id,
                updated_by=actor_id,
            )
        )

    await db.flush()
    return clone.id


__all__ = [
    "SessionSummaryDTO",
    "deep_clone_interview_config",
    "get_session_summary",
]
