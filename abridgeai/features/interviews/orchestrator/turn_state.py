"""State-application + replay helpers extracted from adaptive.py (Slice 7).

Near-pure mutation of the loaded runtime state in memory — NO DB save (the
caller owns the single ``state_repo.save``). Extracted verbatim from
``run_adaptive_turn`` to keep ``adaptive.py`` under the orchestrator LOC cap;
behaviour is unchanged.

``ADVANCE_ACTIONS`` lives here (the lowest layer) so both this module and
``adaptive.py`` can import it without a cycle.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from abridgeai.features.interviews.orchestrator.coverage import apply_evidence_to_coverage
from abridgeai.features.interviews.orchestrator.decision import (
    InterviewerActionType,
    ReasonCode,
)
from abridgeai.features.interviews.orchestrator.difficulty import update_competence
from abridgeai.features.interviews.orchestrator.intent import StudentIntent
from abridgeai.features.interviews.orchestrator.state import (
    InteractionState,
    InterviewPhase,
    InterviewRuntimeStateData,
    OutcomeCoverageState,
)

if TYPE_CHECKING:
    from uuid import UUID

    from abridgeai.features.interviews.models import InterviewQuestion
    from abridgeai.features.interviews.orchestrator.analysis import AnswerAnalysis

# Actions that move the interview forward to a (different) question. Depth
# probes / clarify / repeat are NOT here — they keep the same question and
# consume the follow-up budget.
ADVANCE_ACTIONS = frozenset(
    {
        InterviewerActionType.ASK_MAIN_QUESTION,
        InterviewerActionType.TRANSITION_TOPIC,
        InterviewerActionType.SKIP_QUESTION,
    }
)


def sync_question_history(
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


def apply_state_updates(  # noqa: C901 -- explicit action branches are auditable
    data: Any,  # noqa: ANN401 -- InterviewRuntimeStateData; loose to avoid import churn
    *,
    intent: Any,  # noqa: ANN401
    analysis: AnswerAnalysis | None,
    decision: Any,  # noqa: ANN401
    selected_question_id: str | None,
    target_outcome_id: str | None,
    target_phase: InterviewPhase | None = None,
    cross_turn_enabled: bool = False,
    hint_ladder_enabled: bool = False,
    per_outcome_difficulty_enabled: bool = False,
) -> None:
    """Mutate the loaded state in memory (NO save here — caller saves once).

    ``target_phase`` (Slice 7, v2) is the phase the phase-policy decided the
    next turn should be in. When provided, it AUTHORITATIVELY sets ``data.phase``
    and maintains ``turns_in_phase`` (reset to 0 on a phase change, else +1),
    overriding the v1 hardcoded OPENING→CORE / →CLOSING transitions below. When
    None (v2 phases disabled), the legacy transitions run unchanged — v1 parity.
    """
    now = datetime.now(UTC).isoformat()
    phase_at_entry = data.phase  # captured before v1 transitions mutate data.phase
    data.last_student_intent = intent.to_dict()

    # Candidate signals.
    sig = data.candidate_signals
    sig.requested_repeat = intent.intent is StudentIntent.ASK_TO_REPEAT
    sig.requested_clarification = intent.intent is StudentIntent.ASK_FOR_CLARIFICATION
    sig.requested_skip = intent.intent is StudentIntent.SKIP_QUESTION
    sig.technical_issue_detected = intent.intent is StudentIntent.TECHNICAL_ISSUE
    sig.appeared_uncertain = intent.intent is StudentIntent.CANNOT_ANSWER
    sig.appeared_off_topic = intent.intent is StudentIntent.OFF_TOPIC

    # Evidence / coverage from analysis (weighted — see coverage.py).
    if analysis is not None and analysis.confidence > 0.0:
        for ev in analysis.evidence:
            cov = data.outcome_coverage.get(ev.outcome_id)
            if cov is None:
                cov = OutcomeCoverageState(outcome_id=ev.outcome_id)
                data.outcome_coverage[ev.outcome_id] = cov
            apply_evidence_to_coverage(cov, ev, now=now)

        # Cross-turn memory (Slice 9, v2). Record a short claim summary for each
        # outcome that got evidence this turn, so a LATER turn's analysis can be
        # fed the candidate's own prior words and spot a contradiction. Bounded
        # to the last 3 claims per outcome, each capped in length. The candidate's
        # words only — never rubric/answer content. Gated: off → no claims logged.
        if cross_turn_enabled:
            for ev in analysis.evidence:
                cov = data.outcome_coverage.get(ev.outcome_id)
                if cov is None:
                    continue
                claim = (ev.summary or "").strip()
                if claim:
                    cov.claims.append(claim[:200])
                    del cov.claims[:-3]  # keep only the last 3

        # Per-outcome difficulty calibration (Slice 12, v2). EWMA-fold this
        # answer's quality into the competence estimate of each outcome it
        # touched, so the selector can target question difficulty per topic.
        # A neutral answer leaves the estimate unchanged (see update_competence).
        # Gated: off → estimates never move → v1 behaviour.
        if per_outcome_difficulty_enabled:
            for ev in analysis.evidence:
                cov = data.outcome_coverage.get(ev.outcome_id)
                if cov is None:
                    continue
                cov.competence_estimate = update_competence(
                    prior=cov.competence_estimate, analysis=analysis
                )

    # End-confirmation state (Slice 4). These actions keep the SAME question and
    # never consume the academic probe budget; they only flip the confirmation
    # flag + interaction state, so they are handled before the follow-up logic.
    if decision.action is InterviewerActionType.REQUEST_END_CONFIRMATION:
        data.pending_confirmation = True
        data.interaction_state = InteractionState.CONFIRMING_END
        return
    if decision.action is InterviewerActionType.CANCEL_END:
        data.pending_confirmation = False
        data.interaction_state = InteractionState.AWAITING_ANSWER
        return

    # Any other resolved action clears a pending confirmation (a confirmed end
    # flows into the closing branch below; nothing else should stay pending).
    data.pending_confirmation = False

    # Follow-up counters + phase.
    if decision.action in ADVANCE_ACTIONS:
        data.current_question_follow_up_count = 0
        # Assistance laddering (Slice 11): a new question resets the hint ladder
        # and reframe variety so escalation starts fresh per question.
        data.hint_level = 0
        data.reframe_count = 0
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
        data.interaction_state = InteractionState.AWAITING_ANSWER
    elif decision.action in (
        InterviewerActionType.BEGIN_CLOSING,
        InterviewerActionType.CLOSE_INTERVIEW,
    ):
        data.phase = InterviewPhase.CLOSING
        data.interaction_state = InteractionState.CLOSING
    elif decision.action in (
        InterviewerActionType.PROMPT_SELF_REFLECTION,
        InterviewerActionType.INVITE_CANDIDATE_QUESTIONS,
        InterviewerActionType.ANSWER_CANDIDATE_QUESTION,
    ):
        # Rich closing sub-steps (Slice 13, v2). Non-finishing turns that keep
        # the session in the CLOSING phase and advance the closing_step marker
        # so the sub-sequence progresses: "" → reflection → questions. An
        # ANSWER_CANDIDATE_QUESTION turn keeps the marker at "questions" so the
        # next turn signs off. These never consume the academic follow-up budget.
        data.phase = InterviewPhase.CLOSING
        data.interaction_state = InteractionState.CLOSING
        if decision.action is InterviewerActionType.PROMPT_SELF_REFLECTION:
            data.closing_step = "reflection"
        elif decision.action is InterviewerActionType.INVITE_CANDIDATE_QUESTIONS:
            data.closing_step = "questions"
        # ANSWER_CANDIDATE_QUESTION leaves closing_step == "questions".
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

    # Phase progression (Slice 7, v2). When the phase policy supplied a target
    # phase it is authoritative: it overrides the v1 hardcoded transitions above
    # (which already ran, but the explicit set here wins) and maintains the
    # phase-dwell counter. A phase CHANGE resets turns_in_phase to 0; staying in
    # the same phase increments it, so the policy can measure dwell next turn.
    if target_phase is not None:
        if target_phase is not phase_at_entry:
            data.phase = target_phase
            data.turns_in_phase = 0
        else:
            data.phase = target_phase
            data.turns_in_phase += 1

    # Assistance laddering (Slice 11, v2). AFTER this turn's utterance rendered
    # at the current level, advance the ladder so a REPEATED hint/reframe on the
    # SAME question escalates next time. The advance-reset above (hint_level=0,
    # reframe_count=0) fires first for advance actions, so escalation is
    # per-question. Gated: off → counters never move → v1 behaviour.
    if hint_ladder_enabled and decision.action not in ADVANCE_ACTIONS:
        if decision.action is InterviewerActionType.PROVIDE_NEUTRAL_HINT:
            data.hint_level += 1
        elif decision.action is InterviewerActionType.REFRAME_QUESTION:
            data.reframe_count += 1

    if target_outcome_id:
        data.current_outcome_id = target_outcome_id


def probe_seed_text(decision: Any, current_question: InterviewQuestion | None) -> str | None:  # noqa: ANN401
    """Text the utterance layer phrases for a non-advance action.

    For a repeat we re-speak the current question; for clarify/probe we let the
    utterance layer supply an answer-safe generic probe (returns None).
    """
    if decision.action is InterviewerActionType.REPEAT_QUESTION and current_question is not None:
        return current_question.prompt_text
    return None


def compact_scores(scored: Any) -> list[dict[str, Any]]:  # noqa: ANN401
    """At most the top candidate's compact score (safeguard #8 — bounded audit)."""
    if scored is None:
        return []
    return [
        {
            "question_id": scored.candidate.question_id,
            "score": round(float(scored.score), 2),
        }
    ]


def with_replay(
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


def rehydrate_replay(replay: dict[str, Any]) -> dict[str, Any]:
    """Rebuild a result dict from a stored replay (next_question stays an id).

    The caller (take_session_step) is responsible for turning the id back into
    an ORM row if needed; for the router's purposes the structured fields plus
    the legacy scalar fields are sufficient, and next_question is re-fetched
    there. Here we return the stored dict as-is (next_question = id string).
    """
    return dict(replay)


__all__ = [
    "ADVANCE_ACTIONS",
    "apply_state_updates",
    "compact_scores",
    "probe_seed_text",
    "rehydrate_replay",
    "sync_question_history",
    "with_replay",
]
