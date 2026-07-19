"""Perception pipeline: classify intent + analyze answer + persist (Phase 2+3).

This is the *perception* half of the orchestrator — it turns a raw student
utterance into structured, persisted signals (intent classification and, for
genuine answers, a structured answer analysis). It does NOT yet decide the
next interviewer action or select a question (Phase 4/5) and is NOT yet wired
into the live ``take_session_step`` path (Phase 17). It exists now so the
perception layer can be built, tested, and verified in isolation before it
drives anything.

Contract
--------
``perceive_turn`` is:

* **Best-effort / never-raises** — every LLM sub-call has its own fallback, so
  a classifier or analyzer failure degrades to a safe default instead of
  breaking the turn.
* **Idempotent-aware** — it takes an optional ``turn_idempotency_key``; a
  duplicate replay (same key already recorded on the runtime-state row) returns
  the *existing* state without re-running the LLMs or bumping the version.
* **Optimistically locked** — the persisted state is saved with the version the
  caller read; a lost race raises :class:`StaleStateError` for the caller to
  handle (reload / treat as duplicate).

Academic-scoring guard: utterances whose intent is non-academic (repeat,
clarification, skip, technical issue, …) are NEVER analyzed or scored — only a
genuine ``answer`` / ``partial_answer`` gets an :class:`AnswerAnalysis`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from abridgeai.features.interviews.orchestrator import repository as state_repo
from abridgeai.features.interviews.orchestrator.analysis import AnswerAnalysis
from abridgeai.features.interviews.orchestrator.analysis_logic import analyze_answer
from abridgeai.features.interviews.orchestrator.intent import (
    NON_ACADEMIC_INTENTS,
    IntentClassification,
    StudentIntent,
)
from abridgeai.features.interviews.orchestrator.intent_logic import classify_intent

if TYPE_CHECKING:
    from collections.abc import Sequence
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession

    from abridgeai.ai.llm import LLMGateway


ACADEMIC_INTENTS: frozenset[StudentIntent] = frozenset(
    {StudentIntent.ANSWER, StudentIntent.PARTIAL_ANSWER}
)


@dataclass(frozen=True)
class PerceptionResult:
    """Outcome of perceiving one student turn.

    ``analysis`` is populated ONLY for academic intents; it is ``None`` for a
    repeat/clarification/skip/etc. ``state_version`` is the new persisted
    version (or the existing one on a duplicate replay). ``was_duplicate`` tells
    the caller this turn was a replay and nothing advanced.
    """

    intent: IntentClassification
    analysis: AnswerAnalysis | None
    state_version: int
    was_duplicate: bool


async def perceive_turn(
    db: AsyncSession,
    *,
    session_id: UUID,
    question_text: str,
    student_utterance: str,
    turn_id: str,
    outcome_id: str | None = None,
    outcome_text: str | None = None,
    expected_evidence: Sequence[str] | None = None,
    common_misconceptions: Sequence[str] | None = None,
    supplementary_instructions: str | None = None,
    turn_idempotency_key: str | None = None,
    pipeline_run_id: UUID | None = None,
    gateway: LLMGateway | None = None,
) -> PerceptionResult:
    """Classify intent, analyze (if academic), and persist into runtime state.

    Does NOT commit — the caller owns the transaction boundary so this joins
    whatever unit of work is in flight (mirrors the repository contract).
    """
    loaded = await state_repo.load_or_init(db, session_id)

    # Idempotency: a duplicate replay of the same turn is a no-op — return the
    # already-persisted intent/analysis without re-running the LLMs.
    if state_repo.is_duplicate_turn(loaded, turn_idempotency_key):
        prior_intent = _intent_from_state(loaded.data.last_student_intent)
        prior_analysis = _analysis_from_state(loaded.data.last_answer_analysis, turn_id)
        return PerceptionResult(
            intent=prior_intent,
            analysis=prior_analysis,
            state_version=loaded.version,
            was_duplicate=True,
        )

    intent = await classify_intent(
        db,
        question_text=question_text,
        student_utterance=student_utterance,
        pipeline_run_id=pipeline_run_id,
        gateway=gateway,
    )

    analysis: AnswerAnalysis | None = None
    if intent.intent in ACADEMIC_INTENTS and intent.intent not in NON_ACADEMIC_INTENTS:
        analysis = await analyze_answer(
            db,
            question_text=question_text,
            student_answer=student_utterance,
            turn_id=turn_id,
            outcome_id=outcome_id,
            outcome_text=outcome_text,
            expected_evidence=expected_evidence,
            common_misconceptions=common_misconceptions,
            supplementary_instructions=supplementary_instructions,
            pipeline_run_id=pipeline_run_id,
            gateway=gateway,
        )

    # Persist the perception into runtime state (version-guarded).
    data = loaded.data
    data.last_student_intent = intent.to_dict()
    data.last_answer_analysis = analysis.to_dict() if analysis is not None else None

    now = datetime.now(UTC).isoformat()
    signals = data.candidate_signals
    signals.requested_repeat = intent.intent is StudentIntent.ASK_TO_REPEAT
    signals.requested_clarification = intent.intent is StudentIntent.ASK_FOR_CLARIFICATION
    signals.requested_skip = intent.intent is StudentIntent.SKIP_QUESTION
    signals.technical_issue_detected = intent.intent is StudentIntent.TECHNICAL_ISSUE
    signals.appeared_uncertain = intent.intent is StudentIntent.CANNOT_ANSWER
    signals.appeared_off_topic = intent.intent is StudentIntent.OFF_TOPIC

    if analysis is not None:
        data.current_outcome_id = outcome_id or data.current_outcome_id
        if analysis.confidence > 0.0:
            for ev in analysis.evidence:
                cov = data.outcome_coverage.get(ev.outcome_id)
                if cov is None:
                    from abridgeai.features.interviews.orchestrator.state import (
                        OutcomeCoverageState,
                    )

                    cov = OutcomeCoverageState(outcome_id=ev.outcome_id)
                    data.outcome_coverage[ev.outcome_id] = cov
                cov.evidence_count += 1
                cov.last_updated_at = now
                if ev.turn_id not in cov.supporting_turn_ids:
                    cov.supporting_turn_ids.append(ev.turn_id)

    new_version = await state_repo.save(
        db,
        session_id,
        data,
        expected_version=loaded.version,
        turn_idempotency_key=turn_idempotency_key,
    )

    return PerceptionResult(
        intent=intent,
        analysis=analysis,
        state_version=new_version,
        was_duplicate=False,
    )


def _intent_from_state(raw: dict[str, object] | None) -> IntentClassification:
    from abridgeai.features.interviews.orchestrator.intent import parse_intent_response

    parsed = parse_intent_response(raw)
    if parsed is not None:
        return parsed
    return IntentClassification(
        intent=StudentIntent.ANSWER,
        confidence=0.0,
        rationale="Replay with no prior intent on record.",
        source="fallback",
    )


def _analysis_from_state(raw: dict[str, object] | None, turn_id: str) -> AnswerAnalysis | None:
    if raw is None:
        return None
    return AnswerAnalysis.from_dict(raw, default_turn_id=turn_id)


__all__ = ["ACADEMIC_INTENTS", "PerceptionResult", "perceive_turn"]
