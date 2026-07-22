"""Unit tests for the interviewer decision policy (Phase 4) and adaptive
question selection (Phase 5).

Both modules are pure (no DB, no LLM), so these tests exercise the full
decision precedence ladder and the scorer's ranking / fallback semantics
directly with plain objects.
"""

from __future__ import annotations

from abridgeai.features.interviews.orchestrator.analysis import (
    AnswerAnalysis,
    Correctness,
    ProbeType,
    Relevance,
)
from abridgeai.features.interviews.orchestrator.decision import (
    DecisionInputs,
    InterviewerActionType,
    ReasonCode,
    decide_next_action,
)
from abridgeai.features.interviews.orchestrator.intent import (
    IntentClassification,
    StudentIntent,
)
from abridgeai.features.interviews.orchestrator.selection import (
    CandidateQuestion,
    SelectionContext,
    score_candidate,
    select_next_question,
    sequential_pick,
)


def _intent(kind: StudentIntent) -> IntentClassification:
    return IntentClassification(intent=kind, confidence=0.9, rationale="test")


def _analysis(
    *,
    relevance: Relevance = Relevance.RELEVANT,
    correctness: Correctness = Correctness.MOSTLY_CORRECT,
    probe: ProbeType = ProbeType.NONE,
) -> AnswerAnalysis:
    return AnswerAnalysis(
        relevance=relevance,
        correctness=correctness,
        recommended_probe_type=probe,
        confidence=0.7,
    )


def _inputs(**overrides: object) -> DecisionInputs:
    base: dict[str, object] = {
        "intent": _intent(StudentIntent.ANSWER),
        "analysis": _analysis(),
        "current_question_follow_up_count": 0,
        "total_follow_up_count": 0,
        "time_fraction_remaining": 0.8,
        "has_next_question": True,
        "all_required_outcomes_covered": False,
    }
    base.update(overrides)
    return DecisionInputs(**base)  # type: ignore[arg-type]


# ── depth-probe mappings (Slice 8) ───────────────────────────────────────────


def test_depth_probe_types_map_to_actions_and_reason() -> None:
    from abridgeai.features.interviews.orchestrator.decision import (
        _probe_action,
        _probe_reason,
    )

    assert _probe_action(ProbeType.EXTEND_STRONG) is InterviewerActionType.EXTEND_ANSWER
    assert _probe_action(ProbeType.PROBE_EDGE_CASE) is InterviewerActionType.PROBE_EDGE_CASE
    assert _probe_reason(ProbeType.EXTEND_STRONG) is ReasonCode.STRONG_ANSWER_DEPTH_PROBE
    assert _probe_reason(ProbeType.PROBE_EDGE_CASE) is ReasonCode.STRONG_ANSWER_DEPTH_PROBE


def _strong_analysis() -> AnswerAnalysis:
    from abridgeai.features.interviews.orchestrator.analysis import (
        Completeness,
        Specificity,
    )

    return AnswerAnalysis(
        relevance=Relevance.RELEVANT,
        completeness=Completeness.COMPLETE,
        correctness=Correctness.CORRECT,
        specificity=Specificity.SPECIFIC,
        recommended_probe_type=ProbeType.NONE,
        confidence=0.9,
    )


# ── self-correction recognition (Slice 15) ───────────────────────────────────


def _self_corrected_analysis(
    *,
    correctness: Correctness = Correctness.MOSTLY_CORRECT,
    probe: ProbeType = ProbeType.NONE,
) -> AnswerAnalysis:
    from abridgeai.features.interviews.orchestrator.analysis import (
        Completeness,
        Specificity,
    )

    return AnswerAnalysis(
        relevance=Relevance.RELEVANT,
        completeness=Completeness.COMPLETE,
        correctness=correctness,
        specificity=Specificity.SPECIFIC,
        recommended_probe_type=probe,
        confidence=0.8,
        self_corrected=True,
    )


def test_self_correction_earns_positive_ack_when_enabled() -> None:
    from abridgeai.features.interviews.orchestrator.decision import AcknowledgementStyle

    d = decide_next_action(
        _inputs(
            analysis=_self_corrected_analysis(),
            self_correction_enabled=True,
        )
    )
    # A candidate who fixed their own mistake is advanced (nothing else to probe)
    # but explicitly rewarded with a POSITIVE acknowledgement.
    assert d.acknowledgement_style is AcknowledgementStyle.POSITIVE
    assert "self_correction" in d.tags
    # Invariant: still records evidence + advances like any complete answer.
    assert d.should_record_academic_evidence is True


def test_self_correction_suppresses_contradiction_probe_when_enabled() -> None:
    # The candidate already resolved their own contradiction, so we must NOT
    # issue a RESOLVE_CONTRADICTION probe pointing at what they just fixed.
    d = decide_next_action(
        _inputs(
            analysis=_self_corrected_analysis(probe=ProbeType.RESOLVE_CONTRADICTION),
            self_correction_enabled=True,
        )
    )
    assert d.action is not InterviewerActionType.RESOLVE_CONTRADICTION


def test_self_correction_is_inert_when_flag_off() -> None:
    # Flag off → byte-for-byte v1: the self_corrected signal is ignored, so a
    # recommended contradiction probe still fires and the ack is not forced positive.
    d = decide_next_action(
        _inputs(
            analysis=_self_corrected_analysis(probe=ProbeType.RESOLVE_CONTRADICTION),
            self_correction_enabled=False,
        )
    )
    assert d.action is InterviewerActionType.RESOLVE_CONTRADICTION


# ── confident-but-wrong forced challenge (Slice 16) ──────────────────────────


def _confidently_wrong_analysis(*, probe: ProbeType = ProbeType.NONE) -> AnswerAnalysis:
    from abridgeai.features.interviews.orchestrator.analysis import (
        Completeness,
        Specificity,
    )

    return AnswerAnalysis(
        relevance=Relevance.RELEVANT,
        completeness=Completeness.COMPLETE,
        correctness=Correctness.INCORRECT,
        specificity=Specificity.SPECIFIC,
        recommended_probe_type=probe,
        confidence=0.9,
    )


def test_confident_wrong_forces_challenge_when_enabled() -> None:
    # A specific, relevant, confidently-WRONG answer with budget + time left:
    # the interviewer leans in with a challenge instead of quietly advancing.
    d = decide_next_action(
        _inputs(
            analysis=_confidently_wrong_analysis(),
            confident_wrong_challenge_enabled=True,
        )
    )
    assert d.action is InterviewerActionType.CHALLENGE_REASONING
    assert d.reason_code is ReasonCode.CONFIDENT_BUT_WRONG_CHALLENGE
    assert d.should_record_academic_evidence is True
    assert d.should_advance_question is False
    assert "confident_wrong" in d.tags


def test_confident_wrong_inert_when_flag_off() -> None:
    # Flag off → byte-for-byte v1: no forced challenge; a confidently-wrong
    # answer with no recommended probe just advances (records evidence first).
    d = decide_next_action(
        _inputs(
            analysis=_confidently_wrong_analysis(),
            confident_wrong_challenge_enabled=False,
        )
    )
    assert d.action is not InterviewerActionType.CHALLENGE_REASONING


def test_confident_wrong_respects_followup_budget() -> None:
    # Loop protection wins: with the per-question budget exhausted, we must NOT
    # force another challenge — we advance even though the answer was wrong.
    from abridgeai.features.interviews.orchestrator.decision import (
        DEFAULT_MAX_FOLLOWUPS_PER_QUESTION,
    )

    d = decide_next_action(
        _inputs(
            analysis=_confidently_wrong_analysis(),
            confident_wrong_challenge_enabled=True,
            current_question_follow_up_count=DEFAULT_MAX_FOLLOWUPS_PER_QUESTION,
        )
    )
    assert d.action is not InterviewerActionType.CHALLENGE_REASONING
    assert d.should_advance_question is True


def test_confident_wrong_defers_to_explicit_recommended_probe() -> None:
    # If the analyzer already recommends a probe, the normal rule-11 probe path
    # handles it (higher precedence); the forced-challenge rule only fires when
    # nothing else would probe. Here an explicit ask_for_example is honored.
    d = decide_next_action(
        _inputs(
            analysis=_confidently_wrong_analysis(probe=ProbeType.ASK_FOR_EXAMPLE),
            confident_wrong_challenge_enabled=True,
        )
    )
    assert d.action is InterviewerActionType.ASK_FOR_EXAMPLE


# ── rambling redirect (Slice 17) ─────────────────────────────────────────────


def _rambling_analysis(*, probe: ProbeType = ProbeType.NONE) -> AnswerAnalysis:
    # On-topic but meandering: relevant, partial, general (not vague/off-topic),
    # no probe recommended. The affect layer flags the ramble; the decision uses
    # the plain `rambling` signal to steer.
    from abridgeai.features.interviews.orchestrator.analysis import (
        Completeness,
        Specificity,
    )

    return AnswerAnalysis(
        relevance=Relevance.RELEVANT,
        completeness=Completeness.PARTIAL,
        correctness=Correctness.MIXED,
        specificity=Specificity.GENERAL,
        recommended_probe_type=probe,
        confidence=0.7,
    )


def test_rambling_redirect_when_enabled() -> None:
    # A long, on-topic, low-substance ramble with budget + time and no other
    # probe → the interviewer steers back to focus instead of advancing.
    d = decide_next_action(
        _inputs(
            analysis=_rambling_analysis(),
            rambling=True,
            rambling_redirect_enabled=True,
        )
    )
    assert d.action is InterviewerActionType.REDIRECT_TO_TOPIC
    assert d.reason_code is ReasonCode.RAMBLING_REDIRECT
    assert d.should_advance_question is False
    assert d.should_record_academic_evidence is True
    assert "rambling" in d.tags


def test_rambling_redirect_inert_when_flag_off() -> None:
    # Flag off → byte-for-byte v1: the rambling signal is ignored, no redirect.
    d = decide_next_action(
        _inputs(
            analysis=_rambling_analysis(),
            rambling=True,
            rambling_redirect_enabled=False,
        )
    )
    assert d.action is not InterviewerActionType.REDIRECT_TO_TOPIC


def test_rambling_redirect_only_when_actually_rambling() -> None:
    # Feature on but affect is not rambling → no redirect (v1 advance/probe).
    d = decide_next_action(
        _inputs(
            analysis=_rambling_analysis(),
            rambling=False,
            rambling_redirect_enabled=True,
        )
    )
    assert d.action is not InterviewerActionType.REDIRECT_TO_TOPIC


def test_rambling_redirect_respects_followup_budget() -> None:
    # Loop protection: with the per-question budget exhausted, a ramble does not
    # trigger another redirect — we advance.
    from abridgeai.features.interviews.orchestrator.decision import (
        DEFAULT_MAX_FOLLOWUPS_PER_QUESTION,
    )

    d = decide_next_action(
        _inputs(
            analysis=_rambling_analysis(),
            rambling=True,
            rambling_redirect_enabled=True,
            current_question_follow_up_count=DEFAULT_MAX_FOLLOWUPS_PER_QUESTION,
        )
    )
    assert d.action is not InterviewerActionType.REDIRECT_TO_TOPIC
    assert d.should_advance_question is True


def test_rambling_redirect_defers_to_explicit_probe() -> None:
    # An analyzer-recommended probe (rule 11) wins; the rambling redirect only
    # fires when nothing else would probe.
    d = decide_next_action(
        _inputs(
            analysis=_rambling_analysis(probe=ProbeType.ASK_FOR_EXAMPLE),
            rambling=True,
            rambling_redirect_enabled=True,
        )
    )
    assert d.action is InterviewerActionType.ASK_FOR_EXAMPLE


def test_strong_answer_triggers_depth_probe_when_enabled() -> None:
    from abridgeai.features.interviews.orchestrator.state import InterviewPhase

    d = decide_next_action(
        _inputs(
            analysis=_strong_analysis(),
            depth_probe_enabled=True,
            phase=InterviewPhase.DEEP_PROBE,
        )
    )
    assert d.action in (
        InterviewerActionType.EXTEND_ANSWER,
        InterviewerActionType.PROBE_EDGE_CASE,
    )
    assert d.reason_code is ReasonCode.STRONG_ANSWER_DEPTH_PROBE
    assert d.should_record_academic_evidence is True
    assert d.should_advance_question is False


def test_strong_answer_core_phase_uses_extend() -> None:
    from abridgeai.features.interviews.orchestrator.state import InterviewPhase

    d = decide_next_action(
        _inputs(
            analysis=_strong_analysis(),
            depth_probe_enabled=True,
            phase=InterviewPhase.CORE,
        )
    )
    assert d.action is InterviewerActionType.EXTEND_ANSWER


def test_strong_answer_advances_when_flag_off() -> None:
    # v1 parity: with depth_probe_enabled=False a strong answer advances.
    d = decide_next_action(_inputs(analysis=_strong_analysis(), depth_probe_enabled=False))
    assert d.should_advance_question is True
    assert d.action is not InterviewerActionType.EXTEND_ANSWER


def test_depth_probe_respects_followup_cap() -> None:
    # Even a strong answer must not probe once the follow-up budget is exhausted.
    from abridgeai.features.interviews.orchestrator.state import InterviewPhase

    d = decide_next_action(
        _inputs(
            analysis=_strong_analysis(),
            depth_probe_enabled=True,
            phase=InterviewPhase.DEEP_PROBE,
            current_question_follow_up_count=2,
        )
    )
    assert d.should_advance_question is True
    assert d.action is not InterviewerActionType.EXTEND_ANSWER


# ── per-outcome difficulty calibration (Slice 12) ────────────────────────────


def test_per_outcome_competence_biases_difficulty_fit() -> None:
    """A high competence for an outcome favors a HARDER question on it."""
    from abridgeai.features.interviews.orchestrator.selection import _difficulty_fit_score

    # Global student level unknown; per-outcome competence high (0.9 → target senior).
    ctx = SelectionContext(
        asked_question_ids=frozenset(),
        skipped_question_ids=frozenset(),
        outcome_evidence_counts={},
        uncovered_required_outcome_ids=frozenset(),
        student_difficulty_level=None,
        outcome_competence={"o-1": 0.9},
    )
    senior = _difficulty_fit_score(ctx, "senior", outcome_id="o-1")
    junior = _difficulty_fit_score(ctx, "junior", outcome_id="o-1")
    assert senior > junior  # high competence → prefer the harder question


def test_per_outcome_low_competence_favors_easier() -> None:
    from abridgeai.features.interviews.orchestrator.selection import _difficulty_fit_score

    ctx = SelectionContext(
        asked_question_ids=frozenset(),
        skipped_question_ids=frozenset(),
        outcome_evidence_counts={},
        uncovered_required_outcome_ids=frozenset(),
        student_difficulty_level=None,
        outcome_competence={"o-1": 0.1},
    )
    junior = _difficulty_fit_score(ctx, "junior", outcome_id="o-1")
    senior = _difficulty_fit_score(ctx, "senior", outcome_id="o-1")
    assert junior > senior  # low competence → prefer the easier question


def test_no_per_outcome_competence_falls_back_to_global() -> None:
    """Parity: without outcome_competence, fit uses the global student level."""
    from abridgeai.features.interviews.orchestrator.selection import _difficulty_fit_score

    ctx = SelectionContext(
        asked_question_ids=frozenset(),
        skipped_question_ids=frozenset(),
        outcome_evidence_counts={},
        uncovered_required_outcome_ids=frozenset(),
        student_difficulty_level=3,
    )
    # No per-outcome data → global level 3 favors senior.
    assert _difficulty_fit_score(ctx, "senior", outcome_id="o-1") > _difficulty_fit_score(
        ctx, "junior", outcome_id="o-1"
    )


# ── rich closing sub-sequence (Slice 13) ─────────────────────────────────────


def test_rich_closing_off_is_one_shot_begin_closing() -> None:
    """Parity: with rich_closing off, an exhausted pool → v1 one-shot BEGIN_CLOSING."""
    d = decide_next_action(_inputs(has_next_question=False, rich_closing_enabled=False))
    assert d.action is InterviewerActionType.BEGIN_CLOSING


def test_rich_closing_entry_prompts_self_reflection() -> None:
    """Entering closing with rich_closing on emits self-reflection (non-finishing)."""
    from abridgeai.features.interviews.orchestrator.state import InterviewPhase

    d = decide_next_action(
        _inputs(
            has_next_question=False,
            rich_closing_enabled=True,
            closing_step="",
            phase=InterviewPhase.CORE,
        )
    )
    assert d.action is InterviewerActionType.PROMPT_SELF_REFLECTION
    assert d.reason_code is ReasonCode.CLOSING_SELF_REFLECTION


def test_rich_closing_sequence_reflection_then_invite_then_signoff() -> None:
    from abridgeai.features.interviews.orchestrator.state import InterviewPhase

    # reflection → invite questions
    d1 = decide_next_action(
        _inputs(
            rich_closing_enabled=True,
            closing_step="reflection",
            phase=InterviewPhase.CLOSING,
        )
    )
    assert d1.action is InterviewerActionType.INVITE_CANDIDATE_QUESTIONS

    # questions + no interviewer-question → final sign-off (CLOSE_INTERVIEW)
    d2 = decide_next_action(
        _inputs(
            rich_closing_enabled=True,
            closing_step="questions",
            phase=InterviewPhase.CLOSING,
        )
    )
    assert d2.action is InterviewerActionType.CLOSE_INTERVIEW


def test_rich_closing_answers_candidate_question_before_signoff() -> None:
    from abridgeai.features.interviews.orchestrator.state import InterviewPhase

    d = decide_next_action(
        _inputs(
            intent=_intent(StudentIntent.ASK_INTERVIEWER_QUESTION),
            rich_closing_enabled=True,
            closing_step="questions",
            phase=InterviewPhase.CLOSING,
        )
    )
    assert d.action is InterviewerActionType.ANSWER_CANDIDATE_QUESTION
    assert d.reason_code is ReasonCode.CLOSING_ANSWERED_QUESTION


# ── decision precedence ──────────────────────────────────────────────────────


def test_technical_issue_never_scored() -> None:
    d = decide_next_action(_inputs(intent=_intent(StudentIntent.TECHNICAL_ISSUE)))
    assert d.action is InterviewerActionType.HANDLE_TECHNICAL_ISSUE
    assert d.should_record_academic_evidence is False
    assert d.should_advance_question is False


def test_repeat_request_repeats_without_scoring() -> None:
    d = decide_next_action(_inputs(intent=_intent(StudentIntent.ASK_TO_REPEAT)))
    assert d.action is InterviewerActionType.REPEAT_QUESTION
    assert d.reason_code is ReasonCode.STUDENT_REQUESTED_REPEAT
    assert d.should_record_academic_evidence is False
    assert d.should_advance_question is False


def test_clarification_does_not_leak_answer_and_not_scored() -> None:
    d = decide_next_action(_inputs(intent=_intent(StudentIntent.ASK_FOR_CLARIFICATION)))
    assert d.action is InterviewerActionType.CLARIFY_WITHOUT_REVEALING_ANSWER
    assert d.should_record_academic_evidence is False


def test_skip_request_advances_and_marks_skipped() -> None:
    d = decide_next_action(_inputs(intent=_intent(StudentIntent.SKIP_QUESTION)))
    assert d.action is InterviewerActionType.SKIP_QUESTION
    assert d.should_advance_question is True
    assert "skipped" in d.tags


def test_cannot_answer_records_insufficient_and_advances() -> None:
    d = decide_next_action(_inputs(intent=_intent(StudentIntent.CANNOT_ANSWER)))
    assert d.should_advance_question is True
    assert d.should_record_academic_evidence is True
    assert "insufficient_evidence" in d.tags


def test_end_request_asks_for_confirmation_then_closes_on_confirm() -> None:
    # End-confirmation gate (Slice 4): a fresh end request no longer closes
    # immediately — it asks the candidate to confirm.
    fresh = decide_next_action(_inputs(intent=_intent(StudentIntent.END_INTERVIEW)))
    assert fresh.action is InterviewerActionType.REQUEST_END_CONFIRMATION
    assert fresh.reason_code is ReasonCode.END_CONFIRMATION_REQUESTED
    assert fresh.should_advance_question is False
    # With a confirmation pending, the end request (or an explicit confirm) closes.
    confirmed = decide_next_action(
        _inputs(intent=_intent(StudentIntent.END_INTERVIEW), pending_confirmation=True)
    )
    assert confirmed.action is InterviewerActionType.BEGIN_CLOSING
    assert confirmed.reason_code is ReasonCode.END_CONFIRMED


def test_off_topic_redirects_once_then_advances() -> None:
    # First encounter (no follow-ups yet) → redirect.
    first = decide_next_action(
        _inputs(
            intent=_intent(StudentIntent.OFF_TOPIC),
            current_question_follow_up_count=0,
        )
    )
    assert first.action is InterviewerActionType.REDIRECT_TO_TOPIC
    # Persisting off-topic (already redirected once) → advance.
    second = decide_next_action(
        _inputs(
            intent=_intent(StudentIntent.OFF_TOPIC),
            current_question_follow_up_count=1,
        )
    )
    assert second.should_advance_question is True


def test_probe_recommended_probes_and_records_evidence() -> None:
    d = decide_next_action(_inputs(analysis=_analysis(probe=ProbeType.ASK_FOR_EXAMPLE)))
    assert d.action is InterviewerActionType.ASK_FOR_EXAMPLE
    assert d.reason_code is ReasonCode.MISSING_EXAMPLE
    assert d.should_record_academic_evidence is True
    assert d.should_advance_question is False


def test_probe_suppressed_when_followup_limit_reached() -> None:
    d = decide_next_action(
        _inputs(
            analysis=_analysis(probe=ProbeType.PROBE_REASONING),
            current_question_follow_up_count=2,  # == default cap
        )
    )
    # Limit hit → advance instead of probing.
    assert d.should_advance_question is True
    assert d.reason_code is ReasonCode.FOLLOWUP_LIMIT_REACHED


def test_probe_suppressed_when_time_low() -> None:
    d = decide_next_action(
        _inputs(
            analysis=_analysis(probe=ProbeType.PROBE_REASONING),
            time_fraction_remaining=0.15,  # below low-time threshold
        )
    )
    assert d.action is not InterviewerActionType.PROBE_DEEPER


def test_closing_threshold_time_wraps_up() -> None:
    d = decide_next_action(_inputs(time_fraction_remaining=0.05))
    assert d.reason_code is ReasonCode.TIME_RUNNING_LOW


def test_all_required_covered_begins_closing() -> None:
    d = decide_next_action(
        _inputs(
            analysis=_analysis(probe=ProbeType.NONE),
            all_required_outcomes_covered=True,
        )
    )
    assert d.action is InterviewerActionType.BEGIN_CLOSING
    assert d.reason_code is ReasonCode.ALL_REQUIRED_OUTCOMES_COVERED


def test_no_next_question_begins_closing() -> None:
    d = decide_next_action(
        _inputs(analysis=_analysis(probe=ProbeType.NONE), has_next_question=False)
    )
    assert d.action is InterviewerActionType.BEGIN_CLOSING
    assert d.reason_code is ReasonCode.CLOSING_REQUIRED


def test_public_dict_excludes_internal_rationale() -> None:
    d = decide_next_action(_inputs())
    pub = d.to_public_dict()
    assert "internal_rationale" not in pub
    assert "should_record_academic_evidence" not in pub
    assert "action" in pub
    assert "reason_code" in pub
    # Audit dict keeps the internals.
    assert "internal_rationale" in d.to_audit_dict()


# ── adaptive selection ───────────────────────────────────────────────────────


def _q(
    qid: str,
    *,
    outcome: str | None,
    pos: int | None,
    weight: int = 1,
    difficulty: str | None = "mid_level",
) -> CandidateQuestion:
    return CandidateQuestion(
        question_id=qid,
        linked_outcome_id=outcome,
        question_type="conceptual",
        difficulty=difficulty,
        position=pos,
        importance_weight=weight,
    )


def test_selector_prioritises_uncovered_required_outcome() -> None:
    candidates = [
        _q("q1", outcome="o_covered", pos=1, weight=5),
        _q("q2", outcome="o_required", pos=2, weight=1),
    ]
    ctx = SelectionContext(
        asked_question_ids=frozenset(),
        skipped_question_ids=frozenset(),
        outcome_evidence_counts={"o_covered": 3},  # already covered
        uncovered_required_outcome_ids=frozenset({"o_required"}),
    )
    picked = select_next_question(candidates, ctx)
    assert picked is not None
    # q2 wins despite lower weight + later position: it targets the uncovered
    # REQUIRED outcome, while q1's outcome is already amply covered (penalised).
    assert picked.candidate.question_id == "q2"


def test_selector_never_returns_asked_question() -> None:
    candidates = [_q("q1", outcome="o1", pos=1), _q("q2", outcome="o2", pos=2)]
    ctx = SelectionContext(
        asked_question_ids=frozenset({"q1"}),
        skipped_question_ids=frozenset(),
        outcome_evidence_counts={},
        uncovered_required_outcome_ids=frozenset(),
    )
    picked = select_next_question(candidates, ctx)
    assert picked is not None
    assert picked.candidate.question_id == "q2"


def test_selector_skips_skipped_questions() -> None:
    candidates = [_q("q1", outcome="o1", pos=1)]
    ctx = SelectionContext(
        asked_question_ids=frozenset(),
        skipped_question_ids=frozenset({"q1"}),
        outcome_evidence_counts={},
        uncovered_required_outcome_ids=frozenset(),
    )
    assert select_next_question(candidates, ctx) is None


def test_selector_penalises_recent_outcome_fixation() -> None:
    candidates = [
        _q("q_same", outcome="o1", pos=1),
        _q("q_other", outcome="o2", pos=2),
    ]
    ctx = SelectionContext(
        asked_question_ids=frozenset(),
        skipped_question_ids=frozenset(),
        outcome_evidence_counts={},
        uncovered_required_outcome_ids=frozenset(),
        last_targeted_outcome_id="o1",
    )
    picked = select_next_question(candidates, ctx)
    assert picked is not None
    # Both uncovered, but q_same repeats the last outcome → penalised → q_other.
    assert picked.candidate.question_id == "q_other"


def test_empty_pool_returns_none() -> None:
    ctx = SelectionContext(
        asked_question_ids=frozenset(),
        skipped_question_ids=frozenset(),
        outcome_evidence_counts={},
        uncovered_required_outcome_ids=frozenset(),
    )
    assert select_next_question([], ctx) is None


def test_sequential_fallback_preserves_teacher_order() -> None:
    candidates = [
        _q("q3", outcome=None, pos=3),
        _q("q1", outcome=None, pos=1),
        _q("q2", outcome=None, pos=2),
    ]
    # None asked → first by position.
    first = sequential_pick(candidates, frozenset())
    assert first is not None
    assert first.question_id == "q1"
    # q1 asked → q2 next.
    second = sequential_pick(candidates, frozenset({"q1"}))
    assert second is not None
    assert second.question_id == "q2"
    # all asked → None.
    assert sequential_pick(candidates, frozenset({"q1", "q2", "q3"})) is None


def test_score_breakdown_is_populated() -> None:
    ctx = SelectionContext(
        asked_question_ids=frozenset(),
        skipped_question_ids=frozenset(),
        outcome_evidence_counts={},
        uncovered_required_outcome_ids=frozenset({"o1"}),
    )
    scored = score_candidate(_q("q1", outcome="o1", pos=1, weight=5), ctx)
    assert scored.breakdown["uncovered_outcome"] > 0
    assert scored.breakdown["required_outcome"] > 0
    assert scored.breakdown["importance"] == 40.0  # 8.0 * 5
    assert scored.score > 0


# ── outcome backtracking: under-covered reward (Slice 18) ────────────────────


def test_undercovered_reward_only_when_backtrack_enabled() -> None:
    # An outcome with 1 point (touched but < COVERAGE_SUFFICIENT_POINTS=2) is
    # "under-covered". The reward term is present only when the flag is on.
    off = SelectionContext(
        asked_question_ids=frozenset(),
        skipped_question_ids=frozenset(),
        outcome_evidence_counts={"o1": 1},
        uncovered_required_outcome_ids=frozenset(),
    )
    on = SelectionContext(
        asked_question_ids=frozenset(),
        skipped_question_ids=frozenset(),
        outcome_evidence_counts={"o1": 1},
        uncovered_required_outcome_ids=frozenset(),
        backtrack_undercovered=True,
    )
    q = _q("q1", outcome="o1", pos=1)
    assert score_candidate(q, off).breakdown["undercovered_outcome"] == 0.0
    assert score_candidate(q, on).breakdown["undercovered_outcome"] > 0.0


def test_undercovered_reward_not_given_to_fresh_or_covered_outcomes() -> None:
    # The reward targets ONLY the 0<points<sufficient band: a fresh (0) outcome
    # keeps the big uncovered reward instead, and a covered (>=2) outcome gets
    # neither the under-covered nor uncovered reward.
    ctx_kwargs = {
        "asked_question_ids": frozenset(),
        "skipped_question_ids": frozenset(),
        "uncovered_required_outcome_ids": frozenset(),
        "backtrack_undercovered": True,
    }
    fresh = SelectionContext(outcome_evidence_counts={"o1": 0}, **ctx_kwargs)
    covered = SelectionContext(outcome_evidence_counts={"o1": 2}, **ctx_kwargs)
    q = _q("q1", outcome="o1", pos=1)
    assert score_candidate(q, fresh).breakdown["undercovered_outcome"] == 0.0
    assert score_candidate(q, fresh).breakdown["uncovered_outcome"] > 0.0
    assert score_candidate(q, covered).breakdown["undercovered_outcome"] == 0.0
    assert score_candidate(q, covered).breakdown["uncovered_outcome"] == 0.0


def test_backtrack_prefers_undercovered_over_fully_covered() -> None:
    # The realism win: an un-asked question on an under-covered outcome (1 pt)
    # is preferred over one on a fully-covered outcome, even when the covered
    # outcome has higher importance (which would otherwise win via importance).
    candidates = [
        _q("q_covered", outcome="o_covered", pos=1, weight=5),
        _q("q_partial", outcome="o_partial", pos=2, weight=1),
    ]
    ctx = SelectionContext(
        asked_question_ids=frozenset(),
        skipped_question_ids=frozenset(),
        outcome_evidence_counts={"o_covered": 2, "o_partial": 1},
        uncovered_required_outcome_ids=frozenset(),
        backtrack_undercovered=True,
    )
    picked = select_next_question(candidates, ctx)
    assert picked is not None
    assert picked.candidate.question_id == "q_partial"


def test_backtrack_still_prefers_fresh_uncovered_over_undercovered() -> None:
    # Breadth-first preserved: a fresh (0-point) outcome still outranks an
    # under-covered one, because the uncovered reward (100) > undercovered (40).
    candidates = [
        _q("q_fresh", outcome="o_fresh", pos=2, weight=1),
        _q("q_partial", outcome="o_partial", pos=1, weight=1),
    ]
    ctx = SelectionContext(
        asked_question_ids=frozenset(),
        skipped_question_ids=frozenset(),
        outcome_evidence_counts={"o_partial": 1},
        uncovered_required_outcome_ids=frozenset(),
        backtrack_undercovered=True,
    )
    picked = select_next_question(candidates, ctx)
    assert picked is not None
    assert picked.candidate.question_id == "q_fresh"


def test_backtrack_off_is_byte_for_byte_v1() -> None:
    # Parity: with the flag off, the under-covered outcome gets no special
    # treatment, so importance carries the covered outcome to the win (v1).
    candidates = [
        _q("q_covered", outcome="o_covered", pos=1, weight=5),
        _q("q_partial", outcome="o_partial", pos=2, weight=1),
    ]
    ctx = SelectionContext(
        asked_question_ids=frozenset(),
        skipped_question_ids=frozenset(),
        outcome_evidence_counts={"o_covered": 2, "o_partial": 1},
        uncovered_required_outcome_ids=frozenset(),
    )
    picked = select_next_question(candidates, ctx)
    assert picked is not None
    assert picked.candidate.question_id == "q_covered"
