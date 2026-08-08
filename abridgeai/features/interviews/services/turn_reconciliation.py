"""Off-turn-path full answer analysis + coverage reconciliation.

The turn path spends one fast :func:`~orchestrator.sufficiency_logic.probe_sufficiency`
call and moves on. This service is the other half: it runs the FULL extraction
(:func:`~orchestrator.analysis_logic.analyze_turn`, unchanged) for that same turn
and reconciles the provisional coverage the probe established against the
authoritative result.

Owns the transaction boundary and the optimistic-lock retry, so the ARQ wrapper
stays thin in the manner of :mod:`interviews.workers.evaluation`. A turn whose
row has vanished, or that carries no answer text, is skipped rather than retried
forever.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from uuid import UUID

from abridgeai.core.config import get_settings
from abridgeai.features.interviews.models import (
    InterviewQuestion,
    InterviewSession,
    InterviewSessionMessage,
)
from abridgeai.features.interviews.orchestrator import repository as state_repo
from abridgeai.features.interviews.orchestrator import turn_perception
from abridgeai.features.interviews.orchestrator.reconciliation import (
    ReconciliationResult,
    reconcile_turn_coverage,
)
from abridgeai.features.interviews.orchestrator.repository import StaleStateError
from abridgeai.features.interviews.orchestrator.sufficiency import SufficiencyVerdict

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from abridgeai.features.interviews.orchestrator.analysis import AnswerAnalysis

logger = logging.getLogger(__name__)

# The state row is written by every live turn, so a reconciliation landing while
# the candidate is still talking can lose the optimistic-lock race. Reloading and
# re-applying the delta is always safe (the fold is commutative), so a small
# bounded budget converges without burning a job-level retry.
_SAVE_ATTEMPTS = 3


@dataclass(frozen=True)
class TurnAnalysisRequest:
    """One turn's coordinates, parsed once at the worker boundary.

    ``turn_id`` is the evidence turn identifier the synchronous path used (the
    session-question id), NOT the message id — the two are distinct and the
    evidence trail must keep citing whichever one the probe cited.
    """

    session_id: UUID
    message_id: UUID
    question_id: UUID | None
    turn_id: str
    probe_verdict: SufficiencyVerdict | None

    @classmethod
    def from_payload(cls, session_id: UUID, payload: dict[str, Any]) -> TurnAnalysisRequest:
        raw_question = payload.get("question_id")
        raw_probe = payload.get("probe_verdict")
        return cls(
            session_id=session_id,
            message_id=UUID(str(payload["message_id"])),
            question_id=UUID(str(raw_question)) if raw_question else None,
            turn_id=str(payload["turn_id"]),
            probe_verdict=(
                SufficiencyVerdict.from_dict(raw_probe) if isinstance(raw_probe, dict) else None
            ),
        )


@dataclass(frozen=True)
class _FullAnalysis:
    """The authoritative analysis plus the ids its evidence may legitimately cite."""

    analysis: AnswerAnalysis
    target_outcome_id: str | None
    allowed_other: tuple[str, ...]


async def reconcile_turn_analysis(
    db: AsyncSession, request: TurnAnalysisRequest
) -> ReconciliationResult | None:
    """Run the full analysis for one turn and reconcile its coverage.

    Returns None when the turn could not be analyzed (session, message, or answer
    text missing) or when lock contention outlasted the retry budget. In both
    cases nothing is changed and the probe's provisional coverage stands — which
    costs at most some question-selection accuracy for the rest of the session,
    because the post-session evaluator re-judges the transcript independently.
    """
    resolved = await _run_full_analysis(db, request)
    if resolved is None:
        return None
    # Commit the analysis's ``ai_model_calls`` audit rows before touching state:
    # a lost optimistic-lock race below rolls back, and the record of what the
    # model was asked and what it answered must survive that.
    await db.commit()
    return await _persist_reconciliation(db, request, resolved)


async def _run_full_analysis(
    db: AsyncSession, request: TurnAnalysisRequest
) -> _FullAnalysis | None:
    session = await db.get(InterviewSession, request.session_id)
    message = await db.get(InterviewSessionMessage, request.message_id)
    if session is None or message is None:
        logger.warning(
            "turn reconciliation skipped: session or message missing",
            extra={"session_id": str(request.session_id), "turn_id": request.turn_id},
        )
        return None
    answer_text = (message.content_text or "").strip()
    if not answer_text:
        return None

    question = (
        await db.get(InterviewQuestion, request.question_id)
        if request.question_id is not None
        else None
    )
    outcome_id = (
        str(question.linked_outcome_id)
        if question is not None and question.linked_outcome_id is not None
        else None
    )
    settings = get_settings()
    emergent_enabled = settings.adaptive_v2_emergent_evidence_enabled
    loaded = await state_repo.load_or_init(db, request.session_id)

    analysis = await turn_perception.analyze_turn_answer(
        db,
        data=loaded.data,
        session=session,
        current_question=question,
        question_text=question.prompt_text if question is not None else "",
        answer_text=answer_text,
        turn_id=request.turn_id,
        outcome_id=outcome_id,
        cross_turn_enabled=settings.adaptive_v2_cross_turn_enabled,
        emergent_evidence_enabled=emergent_enabled,
    )
    other_outcomes = await turn_perception.other_outcomes_for_analysis(
        db,
        session.interview_config_id,
        target_outcome_id=outcome_id,
        enabled=emergent_enabled,
    )
    return _FullAnalysis(
        analysis=analysis,
        target_outcome_id=outcome_id,
        allowed_other=tuple(str(o["id"]) for o in other_outcomes),
    )


async def _persist_reconciliation(
    db: AsyncSession, request: TurnAnalysisRequest, resolved: _FullAnalysis
) -> ReconciliationResult | None:
    for attempt in range(1, _SAVE_ATTEMPTS + 1):
        loaded = await state_repo.load_or_init(db, request.session_id)
        result = reconcile_turn_coverage(
            loaded.data,
            turn_id=request.turn_id,
            probe_verdict=request.probe_verdict,
            analysis=resolved.analysis,
            target_outcome_id=resolved.target_outcome_id,
            allowed_other=resolved.allowed_other,
        )
        loaded.data.last_answer_analysis = resolved.analysis.to_dict()
        try:
            await state_repo.save(
                db, request.session_id, loaded.data, expected_version=loaded.version
            )
        except StaleStateError:
            await db.rollback()
            if attempt == _SAVE_ATTEMPTS:
                logger.warning(
                    "turn reconciliation abandoned after lock contention",
                    extra={"session_id": str(request.session_id), "turn_id": request.turn_id},
                )
                return None
            continue
        await db.commit()
        _log_result(request, result)
        return result
    return None


def _log_result(request: TurnAnalysisRequest, result: ReconciliationResult) -> None:
    """Record the disagreement with ids and counts only — never evidence text."""
    logger.info(
        "interview_turn_reconciled",
        extra={
            "session_id": str(request.session_id),
            "turn_id": request.turn_id,
            "changed": result.changed,
            "revoked_outcome_ids": result.revoked_outcome_ids,
            "granted_outcome_ids": result.granted_outcome_ids,
        },
    )


__all__ = ["TurnAnalysisRequest", "reconcile_turn_analysis"]
