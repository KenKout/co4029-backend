"""Quiz audit-event service (Phase 13).

An append-only trail of teacher/student quiz actions. Correctness-bearing events
(regrade, manual grade, override CRUD, question edit, publish, attempt submit)
are written in the SAME transaction as the action — a failed audit insert aborts
the action. ``attempt_started`` and notification side-effects are best-effort.

Layering: owns its own DB writes/reads (precedent: services/taking.py). Never
UPDATEs or DELETEs an event row.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import UUID

from sqlalchemy import select

from abridgeai.core.db.conflict_mapper import flush_or_conflict
from abridgeai.core.security import utcnow
from abridgeai.features.quizzes.models import QuizAuditEvent

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

# Frozen v1 event registry — the only names the recorder accepts.
QUIZ_AUDIT_EVENTS = frozenset(
    {
        "attempt_started",
        "attempt_submitted",
        "attempt_regraded",
        "attempt_manually_graded",
        "override_created",
        "override_updated",
        "override_deleted",
        "question_edited",
        "quiz_published",
    }
)


async def record_event(
    db: AsyncSession,
    *,
    event_name: str,
    quiz_id: UUID,
    actor_user_id: UUID | None = None,
    subject_attempt_id: UUID | None = None,
    subject_question_id: UUID | None = None,
    subject_user_id: UUID | None = None,
    payload: dict[str, Any] | None = None,
) -> QuizAuditEvent:
    """Append one audit event. Participates in the caller's transaction.

    Raises ValueError for an unregistered event name so a typo can't create a
    silent, unqueryable event class.
    """
    if event_name not in QUIZ_AUDIT_EVENTS:
        raise ValueError(f"Unknown quiz audit event: {event_name}")
    row = QuizAuditEvent(
        event_name=event_name,
        quiz_id=quiz_id,
        actor_user_id=actor_user_id,
        subject_attempt_id=subject_attempt_id,
        subject_question_id=subject_question_id,
        subject_user_id=subject_user_id,
        payload_json=payload or {},
        occurred_at=utcnow(),
    )
    db.add(row)
    await flush_or_conflict(db)
    return row


async def list_events_for_quiz(
    db: AsyncSession, quiz_id: UUID, *, limit: int = 100
) -> list[QuizAuditEvent]:
    """Most-recent-first audit trail for a quiz (teacher-facing)."""
    rows = (
        await db.execute(
            select(QuizAuditEvent)
            .where(QuizAuditEvent.quiz_id == quiz_id)
            .order_by(QuizAuditEvent.occurred_at.desc())
            .limit(limit)
        )
    ).scalars().all()
    return list(rows)


__all__ = ["QUIZ_AUDIT_EVENTS", "list_events_for_quiz", "record_event"]
