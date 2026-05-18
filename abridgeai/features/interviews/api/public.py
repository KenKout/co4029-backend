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
    InterviewOutcomeEvaluation,
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


__all__ = [
    "SessionSummaryDTO",
    "get_session_summary",
]
