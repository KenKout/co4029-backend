"""Adaptive interviewer orchestration for the live REST path (Slice 4, Phase 17).

This is the single entry point the shared ``take_session_step`` calls when the
adaptive path is active. It ties the whole pipeline together for ONE student
turn and returns the canonical superset result dict:

    perceive (intent + analysis, in-memory)
        → decide_next_action (deterministic)
        → select_next_question (adaptive, sequential fallback)
        → generate_utterance (LLM phrasing + deterministic fallback)
        → persist ONE AI turn + ONE state-version bump
        → canonical_step_result

Safeguard compliance
--------------------
* #2 nested rollback: the CALLER wraps this in ``db.begin_nested()``; on any
  exception the savepoint rolls back (no orphan AI turn / partial state) and the
  caller runs the legacy fallback. The student-answer row lives in the OUTER
  transaction and survives.
* #3 one version owner: this module calls ``state_repo.save`` EXACTLY once. It
  does NOT call ``perceive_turn`` (which has its own save) — it uses the
  perception logic modules directly and mutates the loaded state in memory.
* #1 idempotency: the caller performs the pre-insert idempotency check; this
  module records ``turn_key`` on the save so a replay is detected next time.
* #4 canonical mapping: the result is built solely by ``canonical_step_result``.

This module performs DB reads + one AI-turn insert + one state save, but does
NOT commit and does NOT open its own savepoint — the caller owns both the
savepoint and the outer commit, so the rollback boundary is unambiguous.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID

from abridgeai.features.interviews.models import (
    InterviewOutcome,
    InterviewQuestion,
    InterviewSessionMessage,
    InterviewSessionQuestion,
)
from abridgeai.features.interviews.orchestrator import repository as state_repo
from abridgeai.features.interviews.orchestrator.analysis import AnswerAnalysis
from abridgeai.features.interviews.orchestrator.analysis_logic import analyze_answer
from abridgeai.features.interviews.orchestrator.decision import (
    DecisionInputs,
    InterviewerActionType,
    ReasonCode,
    decide_next_action,
)
from abridgeai.features.interviews.orchestrator.intent import (
    StudentIntent,
)
from abridgeai.features.interviews.orchestrator.intent_logic import classify_intent
from abridgeai.features.interviews.orchestrator.mapping import canonical_step_result
from abridgeai.features.interviews.orchestrator.security import (
    SecurityAction,
    SecurityAssessment,
    SecurityCategory,
)
from abridgeai.features.interviews.orchestrator.selection import (
    CandidateQuestion,
    SelectionContext,
    select_next_question,
)
from abridgeai.features.interviews.orchestrator.state import (
    InterviewPhase,
    InterviewRuntimeStateData,
    OutcomeCoverageState,
)
from abridgeai.features.interviews.orchestrator.utterance import (
    Utterance,
    build_fallback_utterance,
    persona_from,
)
from abridgeai.features.interviews.orchestrator.utterance_logic import generate_utterance
from abridgeai.features.interviews.services import security as security_service

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from abridgeai.features.interviews.models import InterviewConfig, InterviewSession

logger = logging.getLogger(__name__)

# Provisional coverage is "sufficient" at this many pieces of evidence — used to
# decide which outcomes still need covering (guides selection, not scoring).
_SUFFICIENT_EVIDENCE = 2

ADVANCE_ACTIONS = frozenset(
    {
        InterviewerActionType.ASK_MAIN_QUESTION,
        InterviewerActionType.TRANSITION_TOPIC,
        InterviewerActionType.SKIP_QUESTION,
    }
)


class AdaptiveUnavailable(RuntimeError):  # noqa: N818 -- signal, not an error condition; matches codebase style
    """Raised inside the adaptive attempt to signal 'fall back to legacy'.

    The caller catches this (and any other exception) after the savepoint rolls
    back, then runs the sequential path. Distinct type so the caller can log the
    fallback reason without treating it as an unexpected crash.
    """


@dataclass(frozen=True)
class AdaptiveOutcome:
    """What the adaptive attempt produced (or why it declined)."""

    result: dict[str, Any] | None
    fallback_reason: str | None


def _time_fraction_remaining(session: InterviewSession, config: InterviewConfig) -> float | None:
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


async def _load_candidates(
    db: AsyncSession, config_id: UUID
) -> tuple[list[CandidateQuestion], dict[str, InterviewQuestion]]:
    """Load approved questions as scorer candidates + an id→ORM lookup.

    The outcome importance weight is denormalised onto each candidate so the
    scorer stays pure. Questions with no linked outcome default to weight 1.
    """
    questions = await authoring_list_questions(db, config_id)
    outcomes = await authoring_list_outcomes(db, config_id)
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


async def authoring_list_questions(db: AsyncSession, config_id: UUID) -> list[InterviewQuestion]:
    from abridgeai.features.interviews.queries import authoring as authoring_queries

    return await authoring_queries.list_questions_for_config(
        db, config_id, review_status="approved"
    )


async def authoring_list_outcomes(db: AsyncSession, config_id: UUID) -> list[InterviewOutcome]:
    from abridgeai.features.interviews.queries import authoring as authoring_queries

    return await authoring_queries.list_outcomes_for_config(db, config_id)


async def _persisted_question_ids(db: AsyncSession, session_id: UUID) -> list[UUID]:
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


def _sync_question_history(
    data: InterviewRuntimeStateData,
    persisted_question_ids: list[UUID],
    *,
    current_question: InterviewQuestion | None,
) -> None:
    """Merge the database transcript into lazily-created adaptive state.

    The first session question is attached before runtime state exists. Legacy
    fallbacks can also append questions without updating adaptive state. Unless
    those persisted IDs are merged here, selection treats an already displayed
    question as new and can ask it again immediately after a valid answer.
    """
    known_ids = set(data.asked_question_ids)
    for question_id in persisted_question_ids:
        value = str(question_id)
        if value not in known_ids:
            data.asked_question_ids.append(value)
            known_ids.add(value)

    if current_question is not None:
        current_id = str(current_question.id)
        if current_id not in known_ids:
            data.asked_question_ids.append(current_id)
        data.current_question_id = current_id
        data.current_outcome_id = (
            str(current_question.linked_outcome_id)
            if current_question.linked_outcome_id is not None
            else None
        )


async def run_adaptive_turn(
    db: AsyncSession,
    *,
    session: InterviewSession,
    config: InterviewConfig,
    current_session_question: InterviewSessionQuestion,
    current_question: InterviewQuestion | None,
    answer_text: str,
    turn_key: str,
    language: str,
    use_llm: bool = True,
    security_assessment: SecurityAssessment | None = None,
    security_action: SecurityAction = SecurityAction.ALLOW,
    security_attempt_count: int = 0,
) -> AdaptiveOutcome:
    """Run one adaptive turn. MUST be called inside a caller-owned savepoint.

    Returns an :class:`AdaptiveOutcome` whose ``result`` is the canonical
    superset dict on success. Raises :class:`AdaptiveUnavailable` (or lets other
    exceptions propagate) to make the caller roll back the savepoint and run the
    legacy path — the answer row in the outer transaction is untouched.
    """
    session_id = session.id
    loaded = await state_repo.load_or_init(db, session_id)

    # Idempotency replay: the same turn_key was already processed → return the
    # persisted canonical result WITHOUT re-running anything or bumping version.
    if state_repo.is_duplicate_turn(loaded, turn_key):
        prior = loaded.data.last_answer_analysis or {}
        replay = prior.get("_canonical_result") if isinstance(prior, dict) else None
        if isinstance(replay, dict):
            return AdaptiveOutcome(result=_rehydrate_replay(replay, db), fallback_reason=None)
        # No stored canonical result (shouldn't happen) → treat as fresh.

    data = loaded.data
    _sync_question_history(
        data,
        await _persisted_question_ids(db, session_id),
        current_question=current_question,
    )
    question_text = current_question.prompt_text if current_question is not None else ""
    outcome_id = (
        str(current_question.linked_outcome_id)
        if current_question is not None and current_question.linked_outcome_id is not None
        else None
    )

    # 1. Intent (rules-first, LLM best-effort, never raises).
    intent = await classify_intent(
        db,
        question_text=question_text,
        student_utterance=answer_text,
        gateway=None if use_llm else _no_gateway(),
    )

    # 2. Analysis — only for genuine academic answers.
    analysis: AnswerAnalysis | None = None
    turn_id_placeholder = str(current_session_question.id)
    if intent.intent in (StudentIntent.ANSWER, StudentIntent.PARTIAL_ANSWER):
        outcome_text = None
        if outcome_id is not None:
            oc = await db.get(InterviewOutcome, current_question.linked_outcome_id)  # type: ignore[union-attr]
            outcome_text = oc.outcome_text if oc is not None else None
        analysis = await analyze_answer(
            db,
            question_text=question_text,
            student_answer=answer_text,
            turn_id=turn_id_placeholder,
            outcome_id=outcome_id,
            outcome_text=outcome_text,
            # The author-wide blob may contain full rubric weights or unrelated
            # hidden criteria. Runtime analysis gets only this question and its
            # linked outcome.
            supplementary_instructions=None,
        )

    # 3. Load candidate pool + compute selection context.
    candidates, orm_by_id = await _load_candidates(db, session.interview_config_id)
    asked = frozenset(data.asked_question_ids)
    skipped = frozenset(data.skipped_question_ids)
    evidence_counts = {oid: cov.evidence_count for oid, cov in data.outcome_coverage.items()}

    # Required outcomes not yet sufficiently covered (guides selection priority).
    outcomes = await authoring_list_outcomes(db, session.interview_config_id)
    uncovered_required = frozenset(
        str(o.id) for o in outcomes if evidence_counts.get(str(o.id), 0) < _SUFFICIENT_EVIDENCE
    )
    all_required_covered = len(uncovered_required) == 0 and len(outcomes) > 0

    ctx = SelectionContext(
        asked_question_ids=asked,
        skipped_question_ids=skipped,
        outcome_evidence_counts=evidence_counts,
        uncovered_required_outcome_ids=uncovered_required,
        time_fraction_remaining=_time_fraction_remaining(session, config),
        last_targeted_outcome_id=data.current_outcome_id,
    )
    scored = select_next_question(candidates, ctx)
    has_next = scored is not None

    # 4. Deterministic decision.
    decision = decide_next_action(
        DecisionInputs(
            intent=intent,
            analysis=analysis,
            current_question_follow_up_count=data.current_question_follow_up_count,
            total_follow_up_count=data.total_follow_up_count,
            time_fraction_remaining=ctx.time_fraction_remaining,
            has_next_question=has_next,
            all_required_outcomes_covered=all_required_covered,
        )
    )

    # 5. Resolve the selected question ORM (only for advance actions).
    selected_orm: InterviewQuestion | None = None
    if decision.action in ADVANCE_ACTIONS and scored is not None:
        selected_orm = orm_by_id.get(scored.candidate.question_id)
        decision.target_question_id = scored.candidate.question_id
        decision.target_outcome_id = scored.candidate.linked_outcome_id

    # 6. Natural utterance (LLM phrasing + deterministic fallback).
    persona = persona_from(config.persona)
    probe_or_question_text = (
        selected_orm.prompt_text
        if selected_orm is not None
        else _probe_seed_text(decision, current_question)
    )
    utterance, utt_status = await generate_utterance(
        db,
        decision,
        persona=persona,
        language=language,
        question_text=probe_or_question_text,
        use_llm=use_llm,
    )
    fallback_utterance = build_fallback_utterance(
        decision,
        persona=persona,
        language=language,
        question_text=probe_or_question_text,
    )
    assessment = security_assessment or SecurityAssessment(
        category=SecurityCategory.BENIGN,
        detected=False,
        confidence=1.0,
        should_block=False,
        should_record_academic_evidence=True,
        response_key=None,
        normalized_fingerprint=None,
        source="adaptive",
    )
    allowed_question_ids: list[UUID] = []
    if current_question is not None:
        allowed_question_ids.append(current_question.id)
    if selected_orm is not None:
        allowed_question_ids.append(selected_orm.id)
    guarded = await security_service.guard_student_output(
        db,
        session_id=session_id,
        config_id=session.interview_config_id,
        turn_key=turn_key,
        proposed_text=utterance.ai_turn_text,
        fallback_text=fallback_utterance.ai_turn_text,
        allowed_question_ids=allowed_question_ids,
        assessment=assessment,
        action=security_action,
        attempt_count=security_attempt_count,
    )
    if guarded.output_fallback_used:
        if guarded.text == fallback_utterance.ai_turn_text:
            utterance = fallback_utterance
        else:
            # Even the deterministic selected-question utterance overlapped a
            # protected answer/rubric. Do not attach or serialize that question.
            selected_orm = None
            decision.target_question_id = None
            decision.target_outcome_id = None
            utterance = Utterance("", "", "", guarded.text)
        utt_status = "fallback"
        data.output_leakage_prevented_count += 1

    # 7. Persist ONE AI turn (compact audit in metadata_json — safeguard #8/#11).
    ai_msg = InterviewSessionMessage(
        session_id=session_id,
        session_question_id=current_session_question.id,
        role="ai",
        content_text=utterance.ai_turn_text,
        metadata_json={
            "kind": "adaptive",
            "action": decision.action.value,
            "reason_code": decision.reason_code.value,
            "utterance_status": utt_status,
            "turn_key": turn_key,
            "top_candidates": _compact_scores(scored),
            "output_leakage_blocked": guarded.output_leakage_blocked,
            "output_fallback_used": guarded.output_fallback_used,
            "protected_content_category": guarded.protected_content_category,
        },
    )
    db.add(ai_msg)

    # 8. If advancing, append the selected question as the next session question.
    if selected_orm is not None:
        seq = await _next_sequence(db, session_id)
        db.add(
            InterviewSessionQuestion(
                session_id=session_id,
                interview_question_id=selected_orm.id,
                sequence_no=seq,
            )
        )

    await db.flush()  # assign ai_msg.id; surface the idempotency unique-index race

    # 9. Update runtime state (in memory) then SAVE ONCE (the one version owner).
    _apply_state_updates(
        data,
        intent=intent,
        analysis=analysis,
        decision=decision,
        selected_question_id=(str(selected_orm.id) if selected_orm is not None else None),
        target_outcome_id=decision.target_outcome_id,
    )

    canonical = canonical_step_result(
        decision=decision,
        utterance=utterance,
        selected_question=selected_orm,
        language=language,
        state_version=loaded.version + 1,
        ai_turn_id=str(ai_msg.id) if ai_msg.id is not None else None,
        utterance_status=utt_status,
    )

    # Stash a replay-safe copy of the canonical result (ids as strings) so a
    # duplicate turn_key can be answered without re-running the pipeline.
    data.last_answer_analysis = _with_replay(data.last_answer_analysis, canonical, selected_orm)

    await state_repo.save(
        db,
        session_id,
        data,
        expected_version=loaded.version,
        turn_idempotency_key=turn_key,
    )

    return AdaptiveOutcome(result=canonical, fallback_reason=None)


def _apply_state_updates(  # noqa: C901 -- explicit action branches are auditable
    data: Any,  # noqa: ANN401 -- InterviewRuntimeStateData; loose to avoid import churn
    *,
    intent: Any,  # noqa: ANN401
    analysis: AnswerAnalysis | None,
    decision: Any,  # noqa: ANN401
    selected_question_id: str | None,
    target_outcome_id: str | None,
) -> None:
    """Mutate the loaded state in memory (NO save here — caller saves once)."""
    now = datetime.now(UTC).isoformat()
    data.last_student_intent = intent.to_dict()

    # Candidate signals.
    sig = data.candidate_signals
    sig.requested_repeat = intent.intent is StudentIntent.ASK_TO_REPEAT
    sig.requested_clarification = intent.intent is StudentIntent.ASK_FOR_CLARIFICATION
    sig.requested_skip = intent.intent is StudentIntent.SKIP_QUESTION
    sig.technical_issue_detected = intent.intent is StudentIntent.TECHNICAL_ISSUE
    sig.appeared_uncertain = intent.intent is StudentIntent.CANNOT_ANSWER
    sig.appeared_off_topic = intent.intent is StudentIntent.OFF_TOPIC

    # Evidence / coverage from analysis.
    if analysis is not None and analysis.confidence > 0.0:
        for ev in analysis.evidence:
            cov = data.outcome_coverage.get(ev.outcome_id)
            if cov is None:
                cov = OutcomeCoverageState(outcome_id=ev.outcome_id)
                data.outcome_coverage[ev.outcome_id] = cov
            cov.evidence_count += 1
            cov.last_updated_at = now
            if ev.turn_id not in cov.supporting_turn_ids:
                cov.supporting_turn_ids.append(ev.turn_id)

    # Follow-up counters + phase.
    if decision.action in ADVANCE_ACTIONS:
        data.current_question_follow_up_count = 0
        if selected_question_id is not None:
            if selected_question_id not in data.asked_question_ids:
                data.asked_question_ids.append(selected_question_id)
            data.current_question_id = selected_question_id
        if (
            decision.action is InterviewerActionType.SKIP_QUESTION
            and data.current_question_id
            and data.current_question_id not in data.skipped_question_ids
        ):
            data.skipped_question_ids.append(data.current_question_id)
        if data.phase is InterviewPhase.OPENING:
            data.phase = InterviewPhase.CORE
    elif decision.action in (
        InterviewerActionType.BEGIN_CLOSING,
        InterviewerActionType.CLOSE_INTERVIEW,
    ):
        data.phase = InterviewPhase.CLOSING
    elif decision.reason_code in {
        ReasonCode.STUDENT_REQUESTED_REPEAT,
        ReasonCode.STUDENT_REQUESTED_CLARIFICATION,
        ReasonCode.STUDENT_REQUESTED_HINT,
    }:
        # Candidate controls and assistance do not consume the academic probe
        # budget used to prevent assessment loops.
        pass
    else:
        # A probe / clarify / repeat keeps the same question → count the follow-up.
        data.current_question_follow_up_count += 1
        data.total_follow_up_count += 1

    if target_outcome_id:
        data.current_outcome_id = target_outcome_id


def _probe_seed_text(decision: Any, current_question: InterviewQuestion | None) -> str | None:  # noqa: ANN401
    """Text the utterance layer phrases for a non-advance action.

    For a repeat we re-speak the current question; for clarify/probe we let the
    utterance layer supply an answer-safe generic probe (returns None).
    """
    if decision.action is InterviewerActionType.REPEAT_QUESTION and current_question is not None:
        return current_question.prompt_text
    return None


def _compact_scores(scored: Any) -> list[dict[str, Any]]:  # noqa: ANN401
    """At most the top candidate's compact score (safeguard #8 — bounded audit)."""
    if scored is None:
        return []
    return [
        {
            "question_id": scored.candidate.question_id,
            "score": round(float(scored.score), 2),
        }
    ]


def _with_replay(
    existing: dict[str, Any] | None,
    canonical: dict[str, Any],
    selected_orm: InterviewQuestion | None,
) -> dict[str, Any]:
    """Store a JSON-safe copy of the canonical result for idempotent replay."""
    base = dict(existing) if isinstance(existing, dict) else {}
    replay = {k: v for k, v in canonical.items() if not k.startswith("_")}
    # next_question is an ORM object — store just its id for rehydration.
    replay["next_question"] = str(selected_orm.id) if selected_orm is not None else None
    base["_canonical_result"] = replay
    return base


def _rehydrate_replay(replay: dict[str, Any], db: AsyncSession) -> dict[str, Any]:  # noqa: ARG001
    """Rebuild a result dict from a stored replay (next_question stays an id).

    The caller (take_session_step) is responsible for turning the id back into
    an ORM row if needed; for the router's purposes the structured fields plus
    the legacy scalar fields are sufficient, and next_question is re-fetched
    there. Here we return the stored dict as-is (next_question = id string).
    """
    return dict(replay)


async def _next_sequence(db: AsyncSession, session_id: UUID) -> int:
    from sqlalchemy import func, select

    row = await db.execute(
        select(func.coalesce(func.max(InterviewSessionQuestion.sequence_no), 0)).where(
            InterviewSessionQuestion.session_id == session_id
        )
    )
    return int(row.scalar_one()) + 1


def _no_gateway() -> Any:  # noqa: ANN401
    """Sentinel: when use_llm is False we still pass gateway=None and rely on
    the logic modules' deterministic fallbacks. Kept as a function for clarity.
    """
    return None


__all__ = ["ADVANCE_ACTIONS", "AdaptiveOutcome", "AdaptiveUnavailable", "run_adaptive_turn"]
