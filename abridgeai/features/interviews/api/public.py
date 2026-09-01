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

from abridgeai.core.slug import slugify, unique_slug
from abridgeai.features.interviews.models import (
    InterviewConfig,
    InterviewOutcome,
    InterviewOutcomeEvaluation,
    InterviewQuestion,
    InterviewSession,
)

from ._dto import SessionSummaryDTO


def _coerce_uuid(value: object) -> UUID | None:
    """Normalise a JSONB-array item to a UUID for dict-key lookup.

    ``source_module_ids`` is stored as a JSONB array of uuid strings, so the
    remap dict (keyed on ``UUID`` objects) never matches raw items. Accepts
    already-UUID values and ``str``/anything str()-able; returns ``None`` for
    malformed entries.
    """
    if isinstance(value, UUID):
        return value
    try:
        return UUID(str(value))
    except (ValueError, TypeError):
        return None


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
    target_course_id: UUID | None = None,
    module_id_map: dict[UUID, UUID] | None = None,
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

    Cross-course clone (course-level clone) support — all optional, so the
    same-module duplicate flow is unchanged:

    * ``target_course_id`` — set the clone's ``course_id`` to a DIFFERENT
      course (defaults to the source's course for in-place duplicates).
    * ``module_id_map`` — old module id -> new id. Cloned questions'
      ``source_module_ids`` (the modules their source material lives in) are
      rewritten through the map; ids NOT in the map are dropped rather than
      left pointing at modules that do not exist in the target course.

    Returns the new config id. Flushes but does not commit.
    """
    source = (
        await db.execute(select(InterviewConfig).where(InterviewConfig.id == source_config_id))
    ).scalar_one_or_none()
    if source is None:
        raise ValueError(f"InterviewConfig {source_config_id} not found")

    clone_title = f"{source.title}{title_suffix}"
    # Slug: derive from the clone's title; uniqueness is per-module over live
    # rows, so append -1, -2, ... on collision (same policy as create).
    from sqlalchemy import select as _select  # noqa: PLC0415

    # Title must not collide with a live sibling either — authoring rejects
    # duplicate titles per module, and duplicating twice would otherwise leave
    # two identical "X (Copy)" rows that are indistinguishable in the UI. A
    # duplicate action must not FAIL, so uniquify with a counter instead.
    _title_rows = await db.execute(
        _select(InterviewConfig.title).where(
            InterviewConfig.module_id == target_module_id,
            InterviewConfig.deleted_at.is_(None),
        )
    )
    _titles_taken = {" ".join((row[0] or "").split()).casefold() for row in _title_rows.all()}
    if " ".join(clone_title.split()).casefold() in _titles_taken:
        _n = 2
        while " ".join(f"{clone_title} {_n}".split()).casefold() in _titles_taken:
            _n += 1
        clone_title = f"{clone_title} {_n}"

    _base = slugify(clone_title) or "interview"
    _taken_rows = await db.execute(
        _select(InterviewConfig.slug).where(
            InterviewConfig.module_id == target_module_id,
            InterviewConfig.deleted_at.is_(None),
        )
    )
    _taken = {row[0] for row in _taken_rows.all()}
    clone = InterviewConfig(
        course_id=target_course_id if target_course_id is not None else source.course_id,
        module_id=target_module_id,
        title=clone_title,
        slug=unique_slug(_base, _taken),
        status="draft",
        max_attempts=source.max_attempts,
        cooldown_minutes=source.cooldown_minutes,
        min_outcomes_to_pass=source.min_outcomes_to_pass,
        time_limit_minutes=source.time_limit_minutes,
        persona=source.persona,
        persona_profile_json=(
            dict(source.persona_profile_json)
            if source.persona_profile_json is not None
            else None
        ),
        tts_voice=source.tts_voice,
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
                source_module_ids=(
                    [
                        str(new_id)
                        for m in (src_q.source_module_ids or [])
                        if (m_id := _coerce_uuid(m)) is not None
                        and (new_id := module_id_map.get(m_id)) is not None
                    ]
                    if module_id_map is not None
                    else list(src_q.source_module_ids or [])
                ),
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
