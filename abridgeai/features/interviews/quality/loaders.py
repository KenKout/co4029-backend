"""Read-only DB loaders for the post-hoc quality metrics.

Separated from the metric logic so the metrics stay pure and unit-testable, and
so it is obvious by inspection that this feature only ever READS: every function
here issues SELECTs and nothing else. Nothing in ``quality`` writes to the
database or touches a live session.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import UUID

from sqlalchemy import select

from abridgeai.features.interviews.models import (
    InterviewOutcome,
    InterviewOutcomeEvaluation,
    InterviewRuntimeState,
    InterviewSessionMessage,
)
from abridgeai.features.interviews.quality.transcript import TranscriptTurn

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


async def load_transcript(db: AsyncSession, session_id: UUID) -> list[TranscriptTurn]:
    """Ordered transcript for one session.

    Ordered by ``created_at`` then ``id`` — ``created_at`` alone is not a total
    order (an AI turn and the student turn it answers can share a timestamp at
    low resolution), and a stable tiebreak keeps follow-up detection
    deterministic across runs.
    """
    rows = (
        (
            await db.execute(
                select(InterviewSessionMessage)
                .where(InterviewSessionMessage.session_id == session_id)
                .order_by(
                    InterviewSessionMessage.created_at.asc(),
                    InterviewSessionMessage.id.asc(),
                )
            )
        )
        .scalars()
        .all()
    )
    return [
        TranscriptTurn(
            message_id=str(m.id),
            role=m.role,
            text=m.content_text or "",
            session_question_id=(
                str(m.session_question_id) if m.session_question_id is not None else None
            ),
            sequence_no=idx,
        )
        for idx, m in enumerate(rows)
    ]


async def load_runtime_coverage(
    db: AsyncSession, session_id: UUID
) -> dict[str, dict[str, Any]]:
    """The session's per-outcome runtime coverage map, or ``{}``.

    Empty when the adaptive orchestrator never ran for this session (legacy /
    non-adaptive path), which the calibration report treats as "nothing to
    compare" rather than as zero coverage.
    """
    state = (
        await db.execute(
            select(InterviewRuntimeState).where(InterviewRuntimeState.session_id == session_id)
        )
    ).scalar_one_or_none()
    if state is None or not isinstance(state.state_json, dict):
        return {}
    coverage = state.state_json.get("outcome_coverage")
    if not isinstance(coverage, dict):
        return {}
    return {str(k): v for k, v in coverage.items() if isinstance(v, dict)}


async def load_verdicts(db: AsyncSession, session_id: UUID) -> dict[str, bool]:
    """The post-session evaluator's per-outcome verdicts, or ``{}``.

    Empty when the session has not been evaluated yet — calibration then has
    nothing to score against and says so.
    """
    rows = (
        (
            await db.execute(
                select(InterviewOutcomeEvaluation).where(
                    InterviewOutcomeEvaluation.session_id == session_id
                )
            )
        )
        .scalars()
        .all()
    )
    return {str(r.outcome_id): bool(r.verdict_met) for r in rows}


async def load_outcome_labels(db: AsyncSession, config_id: UUID) -> dict[str, str]:
    """``{outcome_id: outcome_text}`` for human-readable reporting."""
    rows = (
        (
            await db.execute(
                select(InterviewOutcome).where(InterviewOutcome.interview_config_id == config_id)
            )
        )
        .scalars()
        .all()
    )
    return {str(r.id): r.outcome_text for r in rows}


__all__ = [
    "load_outcome_labels",
    "load_runtime_coverage",
    "load_transcript",
    "load_verdicts",
]
