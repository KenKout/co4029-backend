"""DB-read + perception helpers extracted from adaptive.py (Slice 7).

Candidate-pool loading, persisted-question lookups, backend time-fraction, the
end-confirmation override, and small utilities. Extracted verbatim from
``run_adaptive_turn`` to keep ``adaptive.py`` under the orchestrator LOC cap;
behaviour is unchanged. These perform DB reads only — no writes, no save.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID

from abridgeai.features.interviews.models import (
    InterviewOutcome,
    InterviewQuestion,
    InterviewSessionQuestion,
)
from abridgeai.features.interviews.orchestrator.intent import (
    IntentClassification,
    classify_confirmation_reply,
)
from abridgeai.features.interviews.orchestrator.selection import CandidateQuestion

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from abridgeai.features.interviews.models import InterviewConfig, InterviewSession


def time_fraction_remaining(session: InterviewSession, config: InterviewConfig) -> float | None:
    """Authoritative backend time fraction (never trusts the browser clock).

    None when the config is untimed. Clamped to [0, 1].
    """
    limit_min = config.time_limit_minutes
    if not limit_min or limit_min <= 0:
        return None
    started = session.assessment_started_at
    if started is None:
        return 1.0
    if started.tzinfo is None:
        started = started.replace(tzinfo=UTC)
    elapsed = (datetime.now(UTC) - started).total_seconds()
    total = float(limit_min) * 60.0
    return max(0.0, min(1.0, (total - elapsed) / total))


async def load_candidates(
    db: AsyncSession, config_id: UUID
) -> tuple[list[CandidateQuestion], dict[str, InterviewQuestion]]:
    """Load approved questions as scorer candidates + an id→ORM lookup.

    The outcome importance weight is denormalised onto each candidate so the
    scorer stays pure. Questions with no linked outcome default to weight 1.
    """
    questions = await list_questions(db, config_id)
    outcomes = await list_outcomes(db, config_id)
    weight_by_outcome = {str(o.id): int(o.importance_weight) for o in outcomes}

    candidates: list[CandidateQuestion] = []
    orm_by_id: dict[str, InterviewQuestion] = {}
    for q in questions:
        oid = str(q.linked_outcome_id) if q.linked_outcome_id is not None else None
        candidates.append(
            CandidateQuestion(
                question_id=str(q.id),
                linked_outcome_id=oid,
                question_type=q.question_type,
                difficulty=q.difficulty,
                position=q.position,
                importance_weight=weight_by_outcome.get(oid or "", 1),
            )
        )
        orm_by_id[str(q.id)] = q
    return candidates, orm_by_id


async def list_questions(db: AsyncSession, config_id: UUID) -> list[InterviewQuestion]:
    from abridgeai.features.interviews.queries import authoring as authoring_queries

    return await authoring_queries.list_questions_for_config(
        db, config_id, review_status="approved"
    )


async def list_outcomes(db: AsyncSession, config_id: UUID) -> list[InterviewOutcome]:
    from abridgeai.features.interviews.queries import authoring as authoring_queries

    return await authoring_queries.list_outcomes_for_config(db, config_id)


async def persisted_question_ids(db: AsyncSession, session_id: UUID) -> list[UUID]:
    """Return every actual interview question already displayed this session."""
    from sqlalchemy import select  # noqa: PLC0415

    stmt = (
        select(InterviewSessionQuestion.interview_question_id)
        .where(
            InterviewSessionQuestion.session_id == session_id,
            InterviewSessionQuestion.interview_question_id.is_not(None),
        )
        .order_by(InterviewSessionQuestion.sequence_no)
    )
    question_ids = (await db.execute(stmt)).scalars().all()
    return [question_id for question_id in question_ids if question_id is not None]


def confirmation_override(
    intent: IntentClassification, answer_text: str, *, pending: bool
) -> IntentClassification:
    """End-confirmation gate (Slice 4).

    While a confirmation is pending, an unambiguous yes/no is reinterpreted as
    CONFIRM_END / CANCEL_END, overriding the general classifier (a bare
    "yes"/"no" here is a confirmation reply, not an answer). Non-matches fall
    through unchanged; the decision policy treats anything that isn't a confirm
    as a cancel while pending, so this stays safe. When no confirmation is
    pending the intent is returned as-is.
    """
    if not pending:
        return intent
    reply = classify_confirmation_reply(answer_text)
    return reply if reply is not None else intent


async def next_sequence(db: AsyncSession, session_id: UUID) -> int:
    from sqlalchemy import func, select

    row = await db.execute(
        select(func.coalesce(func.max(InterviewSessionQuestion.sequence_no), 0)).where(
            InterviewSessionQuestion.session_id == session_id
        )
    )
    return int(row.scalar_one()) + 1


def no_gateway() -> Any:  # noqa: ANN401
    """Sentinel: when use_llm is False we still pass gateway=None and rely on
    the logic modules' deterministic fallbacks. Kept as a function for clarity.
    """
    return None


def prior_claims_for(data: Any, outcome_id: str | None, *, enabled: bool) -> list[str]:  # noqa: ANN401
    """Prior claims the candidate made about ``outcome_id`` (Slice 9, v2).

    Returns the bounded per-outcome claims list so answer analysis can flag a
    cross-turn contradiction. Empty when the feature is disabled, no outcome is
    linked, or the outcome has no recorded coverage yet — i.e. v1 behaviour.
    """
    if not enabled or outcome_id is None:
        return []
    cov = data.outcome_coverage.get(outcome_id)
    return list(cov.claims) if cov is not None else []


__all__ = [
    "confirmation_override",
    "list_outcomes",
    "list_questions",
    "load_candidates",
    "next_sequence",
    "no_gateway",
    "persisted_question_ids",
    "prior_claims_for",
    "time_fraction_remaining",
]
