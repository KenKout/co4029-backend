"""Session entry-view helpers for the learner start-session paths.

Split from services/taking.py for the 800-line metric gate: the three
"return an existing session" paths in ``start_session`` (keyed idempotent
retry, live-session short-circuit, insert-race backstop) share this
prepare-for-response choreography.
"""

from __future__ import annotations

from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.core.db.conflict_mapper import flush_or_conflict
from abridgeai.features.interviews.models import InterviewSession, InterviewSessionQuestion
from abridgeai.features.interviews.services.ceremony import (
    ensure_ceremony_message,
    onboarding_ceremony_kind,
)


async def _restore_session_entry_view(
    db: AsyncSession, session: InterviewSession, config_id: UUID
) -> None:
    """Prep an existing session for a start response: question #1, mode
    alignment, ceremony row. Shared by start_session's three return-existing
    paths."""
    await _ensure_first_question_attached(db, session.id, config_id)
    if session.input_mode != "hybrid":
        session.input_mode = "hybrid"
        await flush_or_conflict(db)
    await ensure_ceremony_message(
        db,
        session=session,
        kind=onboarding_ceremony_kind(session.onboarding_stage),
        language=session.interview_language,
    )


async def _ensure_first_question_attached(
    db: AsyncSession, session_id: UUID, config_id: UUID
) -> None:
    """Attach question #1 to ``session_id`` if it has none yet.

    Closes the gap where ``start_session`` short-circuited on a stale
    ``in_progress`` row created before any question reached
    ``review_status='approved'``. Without this, approving questions
    after the fact never reaches the learner's existing session.
    """
    from sqlalchemy import func, select  # noqa: PLC0415

    count = (
        await db.execute(
            select(func.count(InterviewSessionQuestion.id)).where(
                InterviewSessionQuestion.session_id == session_id
            )
        )
    ).scalar_one()
    if int(count) > 0:
        return
    # Lazy import: _first_published_question lives in taking.py, which imports
    # this module — a module-level import would be circular.
    from abridgeai.features.interviews.services.taking import (  # noqa: PLC0415
        _first_published_question,
    )

    first_question = await _first_published_question(db, config_id, session_seed=str(session_id))
    if first_question is None:
        return
    db.add(
        InterviewSessionQuestion(
            session_id=session_id,
            interview_question_id=first_question.id,
            sequence_no=1,
        )
    )
    await flush_or_conflict(db)

