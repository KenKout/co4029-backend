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

Slice 7 note: the pure/near-pure perception + state-application helpers were
extracted into ``turn_perception`` and ``turn_state`` sibling modules to keep
this file under the orchestrator LOC cap. Behaviour is unchanged.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from uuid import UUID

from abridgeai.features.interviews.models import (
    InterviewOutcome,
    InterviewQuestion,
    InterviewSessionMessage,
    InterviewSessionQuestion,
)
from abridgeai.features.interviews.orchestrator import repository as state_repo
from abridgeai.features.interviews.orchestrator import turn_perception, turn_state
from abridgeai.features.interviews.orchestrator.affect import Affect, detect_affect
from abridgeai.features.interviews.orchestrator.analysis import AnswerAnalysis
from abridgeai.features.interviews.orchestrator.analysis_logic import analyze_answer
from abridgeai.features.interviews.orchestrator.coverage import is_provisionally_sufficient
from abridgeai.features.interviews.orchestrator.decision import (
    DEFAULT_MAX_TOTAL_FOLLOWUPS,
    DecisionInputs,
    decide_next_action,
)
from abridgeai.features.interviews.orchestrator.difficulty import (
    target_difficulty_level,
    update_streaks,
)
from abridgeai.features.interviews.orchestrator.intent import StudentIntent
from abridgeai.features.interviews.orchestrator.intent_logic import classify_intent
from abridgeai.features.interviews.orchestrator.mapping import canonical_step_result
from abridgeai.features.interviews.orchestrator.phases import resolve_phase_and_level
from abridgeai.features.interviews.orchestrator.security import (
    SecurityAction,
    SecurityAssessment,
    SecurityCategory,
)
from abridgeai.features.interviews.orchestrator.selection import (
    SelectionContext,
    select_next_question,
)
from abridgeai.features.interviews.orchestrator.turn_state import ADVANCE_ACTIONS
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

# Sufficiency is decided by weighted coverage points — see coverage.py
# (COVERAGE_SUFFICIENT_POINTS / is_provisionally_sufficient), not a raw count.


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


def _resolve_affect(
    data: Any,  # noqa: ANN401 -- InterviewRuntimeStateData; loose to avoid import churn
    *,
    answer_text: str,
    analysis: AnswerAnalysis | None,
    enabled: bool,
) -> Any | None:  # noqa: ANN401 -- Affect | None; loose to avoid import churn
    """Detect candidate affect + record it on state (Slice 10, v2).

    Returns the detected ``Affect`` (and stores its value on candidate_signals)
    when the feature is enabled, else None → v1 tone. Kept as a helper so
    ``run_adaptive_turn`` stays under the complexity cap.
    """
    if not enabled:
        return None
    affect = detect_affect(answer_text=answer_text, analysis=analysis)
    data.candidate_signals.last_affect = affect.value
    return affect


def _is_rambling(
    *,
    answer_text: str,
    analysis: AnswerAnalysis | None,
    enabled: bool,
) -> bool:
    """Whether the candidate is rambling (Slice 17, v2), gated on the flag.

    Uses the same deterministic ``detect_affect`` as the tone layer (single
    source of truth) so the decision-time signal and the tone lead-in never
    disagree. Off → False → the decision is byte-for-byte v1.
    """
    if not enabled:
        return False
    return detect_affect(answer_text=answer_text, analysis=analysis) is Affect.RAMBLING


# Communication-polish thresholds (Slice 20, v2).
_COMMS_TIME_PRESSURE_FRACTION = 0.2  # signal to prioritise below this time fraction
_COMMS_RECOVERY_WEAK_STREAK = 2  # rebuild after this many consecutive weak answers


def _comms_polish_signals(
    *,
    time_fraction_remaining: float | None,
    consecutive_weak_answers: int,
    enabled: bool,
) -> tuple[bool, bool]:
    """Return ``(time_pressure, recovery)`` tone signals (Slice 20, v2).

    ``time_pressure`` fires when little time remains so the interviewer can tell
    the candidate to prioritise; ``recovery`` fires after a weak streak so a
    rattled candidate gets an encouraging, scoped lead-in. TONE ONLY — these
    feed the utterance lead-in, never the decision. Off → (False, False) → v1.
    """
    if not enabled:
        return False, False
    time_pressure = (
        time_fraction_remaining is not None
        and time_fraction_remaining <= _COMMS_TIME_PRESSURE_FRACTION
    )
    recovery = consecutive_weak_answers >= _COMMS_RECOVERY_WEAK_STREAK
    return time_pressure, recovery


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
    phases_enabled: bool = False,
    depth_probe_enabled: bool = False,
    cross_turn_enabled: bool = False,
    affect_enabled: bool = False,
    hint_ladder_enabled: bool = False,
    per_outcome_difficulty_enabled: bool = False,
    rich_closing_enabled: bool = False,
    self_correction_enabled: bool = False,
    confident_wrong_challenge_enabled: bool = False,
    rambling_redirect_enabled: bool = False,
    backtrack_undercovered_enabled: bool = False,
    comms_polish_enabled: bool = False,
    frustration_deescalation_enabled: bool = False,
    question_deferral_enabled: bool = False,
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
            return AdaptiveOutcome(result=turn_state.rehydrate_replay(replay), fallback_reason=None)
        # No stored canonical result (shouldn't happen) → treat as fresh.

    data = loaded.data
    turn_state.sync_question_history(
        data,
        await turn_perception.persisted_question_ids(db, session_id),
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
        gateway=None if use_llm else turn_perception.no_gateway(),
    )

    # 1b. End-confirmation gate (Slice 4) — see turn_perception.confirmation_override.
    intent = turn_perception.confirmation_override(
        intent, answer_text, pending=data.pending_confirmation
    )

    # 2. Analysis — only for genuine academic answers.
    analysis: AnswerAnalysis | None = None
    turn_id_placeholder = str(current_session_question.id)
    if intent.intent in (StudentIntent.ANSWER, StudentIntent.PARTIAL_ANSWER):
        outcome_text = None
        if outcome_id is not None:
            oc = await db.get(InterviewOutcome, current_question.linked_outcome_id)  # type: ignore[union-attr]
            outcome_text = oc.outcome_text if oc is not None else None
        # Cross-turn memory (Slice 9, v2): the candidate's own prior claims about
        # THIS outcome, so the analyzer can flag a cross-turn contradiction.
        prior_claims = turn_perception.prior_claims_for(
            data, outcome_id, enabled=cross_turn_enabled
        )
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
            prior_claims=prior_claims,
        )

    # 2b. Difficulty streaks (Slice 3). Fold THIS answer's quality into the
    # strong/weak streaks BEFORE selection so it shapes the next question's
    # difficulty. Neutral / low-confidence answers leave both streaks untouched.
    # These fields were persisted but never written until now; apply_state_updates
    # deliberately does not touch them, so this is their single write site.
    data.consecutive_strong_answers, data.consecutive_weak_answers = update_streaks(
        consecutive_strong=data.consecutive_strong_answers,
        consecutive_weak=data.consecutive_weak_answers,
        analysis=analysis,
    )
    current_difficulty = current_question.difficulty if current_question is not None else None
    student_level = target_difficulty_level(
        current_difficulty=current_difficulty,
        consecutive_strong=data.consecutive_strong_answers,
        consecutive_weak=data.consecutive_weak_answers,
    )

    # 3. Load candidate pool + compute selection context.
    candidates, orm_by_id = await turn_perception.load_candidates(db, session.interview_config_id)
    asked = frozenset(data.asked_question_ids)
    skipped = frozenset(data.skipped_question_ids)
    # Weighted coverage points (Slice 2) — not the raw evidence count — drive
    # both the "uncovered" signal for scoring and the sufficiency gate below, so
    # a single low-value (partial/contradictory) turn cannot mark an outcome
    # covered the way a flat count once did.
    coverage_points = {oid: cov.coverage_points for oid, cov in data.outcome_coverage.items()}

    # Required outcomes not yet sufficiently covered (guides selection priority).
    outcomes = await turn_perception.list_outcomes(db, session.interview_config_id)
    uncovered_required = frozenset(
        str(o.id)
        for o in outcomes
        if not is_provisionally_sufficient(coverage_points.get(str(o.id), 0))
    )
    all_required_covered = len(uncovered_required) == 0 and len(outcomes) > 0

    time_fraction = turn_perception.time_fraction_remaining(session, config)

    # Phase progression (Slice 7, v2). When the phases feature is enabled we
    # compute the phase the NEXT turn should be in and bias the difficulty
    # target accordingly (warmup eases down, deep-probe pushes up). When the
    # flag is OFF, target_phase stays the current phase and the bias is 0, so
    # ``student_level`` and every downstream decision are byte-for-byte v1.
    target_phase = data.phase
    if phases_enabled:
        target_phase, student_level = resolve_phase_and_level(
            current_phase=data.phase,
            turns_in_phase=data.turns_in_phase,
            warmup_turns_target=data.warmup_turns_target,
            all_required_covered=all_required_covered,
            time_fraction_remaining=time_fraction,
            total_follow_up_count=data.total_follow_up_count,
            max_total_follow_ups=DEFAULT_MAX_TOTAL_FOLLOWUPS,
            student_level=student_level,
        )

    # Per-outcome difficulty calibration (Slice 12, v2): expose each outcome's
    # competence estimate so selection targets question difficulty per topic.
    # Gated: None → selection falls back to the global student level (v1).
    outcome_competence = (
        {oid: cov.competence_estimate for oid, cov in data.outcome_coverage.items()}
        if per_outcome_difficulty_enabled
        else None
    )

    ctx = SelectionContext(
        asked_question_ids=asked,
        skipped_question_ids=skipped,
        outcome_evidence_counts=coverage_points,
        uncovered_required_outcome_ids=uncovered_required,
        student_difficulty_level=student_level,
        time_fraction_remaining=time_fraction,
        last_targeted_outcome_id=data.current_outcome_id,
        outcome_competence=outcome_competence,
        backtrack_undercovered=backtrack_undercovered_enabled,
    )
    scored = select_next_question(candidates, ctx)
    has_next = scored is not None

    # Rambling signal (Slice 17, v2). Computed BEFORE the decision so the policy
    # can steer a long, on-topic, low-substance ramble back to focus. Uses the
    # same deterministic detect_affect as the tone layer (single source), gated
    # on the flag. Off → False → the decision is byte-for-byte v1.
    rambling = _is_rambling(
        answer_text=answer_text, analysis=analysis, enabled=rambling_redirect_enabled
    )

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
            pending_confirmation=data.pending_confirmation,
            depth_probe_enabled=depth_probe_enabled,
            phase=data.phase,
            rich_closing_enabled=rich_closing_enabled,
            closing_step=data.closing_step,
            self_correction_enabled=self_correction_enabled,
            confident_wrong_challenge_enabled=confident_wrong_challenge_enabled,
            rambling=rambling,
            rambling_redirect_enabled=rambling_redirect_enabled,
            frustration_deescalation_enabled=frustration_deescalation_enabled,
            question_deferral_enabled=question_deferral_enabled,
        )
    )

    # 5. Resolve the selected question ORM (only for advance actions).
    selected_orm: InterviewQuestion | None = None
    if decision.action in ADVANCE_ACTIONS and scored is not None:
        selected_orm = orm_by_id.get(scored.candidate.question_id)
        decision.target_question_id = scored.candidate.question_id
        decision.target_outcome_id = scored.candidate.linked_outcome_id

    # 5b. Candidate affect (Slice 10, v2). Read lightweight affect to warm the
    # utterance TONE only — never control flow. Gated: off → None → v1 tone.
    affect = _resolve_affect(
        data, answer_text=answer_text, analysis=analysis, enabled=affect_enabled
    )

    # 6. Natural utterance (LLM phrasing + deterministic fallback).
    persona = persona_from(config.persona)
    # Resolve any teacher per-trait overrides (Phase 3) layered on the preset,
    # so the phrasing LLM speaks in the tuned tone. TONE ONLY — the persona enum
    # above still keys the deterministic fallback tables and the decision path.
    from abridgeai.features.interviews.orchestrator.persona import (  # noqa: PLC0415
        profile_from_config,
    )

    resolved_persona_profile = profile_from_config(
        config.persona,
        getattr(config, "persona_profile_json", None),
    )
    probe_or_question_text = (
        selected_orm.prompt_text
        if selected_orm is not None
        else turn_state.probe_seed_text(decision, current_question)
    )
    # Assistance laddering (Slice 11, v2): render this turn at the CURRENT ladder
    # level; apply_state_updates advances the level AFTER, so a repeated hint /
    # reframe escalates next turn. Gated: off → level 0 → v1 wording.
    hint_level = data.hint_level if hint_ladder_enabled else 0
    reframe_count = data.reframe_count if hint_ladder_enabled else 0
    # Communication polish (Slice 20, v2): TONE-ONLY lead-ins — signal time
    # pressure when little time remains, and an encouraging recovery framing
    # after a weak streak. Gated: off → (False, False) → v1 wording.
    time_pressure, recovery = _comms_polish_signals(
        time_fraction_remaining=time_fraction,
        consecutive_weak_answers=data.consecutive_weak_answers,
        enabled=comms_polish_enabled,
    )
    utterance, utt_status = await generate_utterance(
        db,
        decision,
        persona=persona,
        language=language,
        question_text=probe_or_question_text,
        persona_profile=resolved_persona_profile,
        use_llm=use_llm,
        affect=affect,
        hint_level=hint_level,
        reframe_count=reframe_count,
        time_pressure=time_pressure,
        recovery=recovery,
    )
    fallback_utterance = build_fallback_utterance(
        decision,
        persona=persona,
        language=language,
        question_text=probe_or_question_text,
        affect=affect,
        hint_level=hint_level,
        reframe_count=reframe_count,
        time_pressure=time_pressure,
        recovery=recovery,
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
            "top_candidates": turn_state.compact_scores(scored),
            "output_leakage_blocked": guarded.output_leakage_blocked,
            "output_fallback_used": guarded.output_fallback_used,
            "protected_content_category": guarded.protected_content_category,
        },
    )
    db.add(ai_msg)

    # 8. If advancing, append the selected question as the next session question.
    if selected_orm is not None:
        seq = await turn_perception.next_sequence(db, session_id)
        db.add(
            InterviewSessionQuestion(
                session_id=session_id,
                interview_question_id=selected_orm.id,
                sequence_no=seq,
            )
        )

    await db.flush()  # assign ai_msg.id; surface the idempotency unique-index race

    # 9. Update runtime state (in memory) then SAVE ONCE (the one version owner).
    turn_state.apply_state_updates(
        data,
        intent=intent,
        analysis=analysis,
        decision=decision,
        selected_question_id=(str(selected_orm.id) if selected_orm is not None else None),
        target_outcome_id=decision.target_outcome_id,
        target_phase=target_phase if phases_enabled else None,
        cross_turn_enabled=cross_turn_enabled,
        hint_ladder_enabled=hint_ladder_enabled,
        per_outcome_difficulty_enabled=per_outcome_difficulty_enabled,
    )

    canonical = canonical_step_result(
        decision=decision,
        utterance=utterance,
        selected_question=selected_orm,
        language=language,
        state_version=loaded.version + 1,
        ai_turn_id=str(ai_msg.id) if ai_msg.id is not None else None,
        utterance_status=utt_status,
        pending_confirmation=data.pending_confirmation,
        interaction_state=data.interaction_state.value,
    )

    # Stash a replay-safe copy of the canonical result (ids as strings) so a
    # duplicate turn_key can be answered without re-running the pipeline.
    data.last_answer_analysis = turn_state.with_replay(
        data.last_answer_analysis, canonical, selected_orm
    )

    await state_repo.save(
        db,
        session_id,
        data,
        expected_version=loaded.version,
        turn_idempotency_key=turn_key,
    )

    return AdaptiveOutcome(result=canonical, fallback_reason=None)


__all__ = ["ADVANCE_ACTIONS", "AdaptiveOutcome", "AdaptiveUnavailable", "run_adaptive_turn"]
