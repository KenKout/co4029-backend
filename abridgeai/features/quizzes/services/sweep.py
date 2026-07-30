"""Overdue-attempt sweep service (Phase 6).

Finds ``in_progress`` attempts whose (grace-extended) deadline has passed and
applies the parent quiz's ``overdue_handling``:

* ``autosubmit`` (default) / ``graceperiod`` -> grade what was answered
  (:func:`taking._finalize_attempt`).
* ``autoabandon`` -> close with no grade (:func:`taking._expire_attempt`).

Runs from the periodic arq task :func:`workers.timing.sweep_overdue_attempts`.
Each attempt is handled independently so one bad row can't abort the batch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import and_, or_, select

from abridgeai.core.observability import get_logger
from abridgeai.core.security import utcnow
from abridgeai.features.quizzes.models import Quiz, QuizAttempt
from abridgeai.features.quizzes.services import timing as _timing
from abridgeai.features.quizzes.services.taking import (
    _expire_attempt,
    _finalize_attempt,
)

if TYPE_CHECKING:
    from datetime import datetime

    from sqlalchemy.ext.asyncio import AsyncSession

_logger = get_logger(__name__)


async def list_overdue_candidates(
    db: AsyncSession, *, limit: int = 500
) -> list[tuple[QuizAttempt, Quiz]]:
    """``in_progress`` attempts on quizzes that have some deadline mechanism.

    Pre-filters in SQL (time limit OR close date present) to keep the scan cheap;
    the exact deadline/grace math is applied per-attempt by :func:`sweep_overdue`.
    """
    stmt = (
        select(QuizAttempt, Quiz)
        .join(Quiz, Quiz.id == QuizAttempt.quiz_id)
        .where(QuizAttempt.status == "in_progress")
        .where(
            or_(
                and_(
                    Quiz.time_limit_seconds.is_not(None),
                    Quiz.time_limit_seconds > 0,
                ),
                Quiz.available_until.is_not(None),
            )
        )
        .order_by(QuizAttempt.started_at.asc())
        .limit(limit)
    )
    return [(row[0], row[1]) for row in (await db.execute(stmt)).all()]


async def sweep_overdue(db: AsyncSession, *, now: datetime | None = None) -> dict[str, int]:
    """Finalize/expire every overdue in_progress attempt. Returns counts."""
    now = now or utcnow()
    submitted = 0
    expired = 0
    skipped = 0
    for attempt, quiz in await list_overdue_candidates(db):
        eff = _timing.resolve_effective_timing(quiz)
        overdue = _timing.is_overdue(
            attempt.started_at,
            eff,
            grace_period_seconds=(
                quiz.grace_period_seconds if quiz.overdue_handling == "graceperiod" else None
            ),
            now=now,
            due_at=quiz.due_at,
            hard_due=False,
        )
        if not overdue:
            skipped += 1
            continue
        if quiz.overdue_handling == "autoabandon":
            await _expire_attempt(db, attempt, now=now)
            expired += 1
        else:
            await _finalize_attempt(db, attempt, quiz, now=now)
            submitted += 1
    return {"submitted": submitted, "expired": expired, "skipped": skipped}


__all__ = ["list_overdue_candidates", "sweep_overdue"]
