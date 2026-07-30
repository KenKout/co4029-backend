"""Unit tests for the post-hoc interview quality metrics.

Covers the three pure layers — calibration (rule-based), transcript shaping
(follow-up detection), and judge-verdict scoring/aggregation. The LLM calls in
``quality.judges`` are I/O and are not exercised here.
"""

from __future__ import annotations

from abridgeai.features.interviews.quality.calibration import compute_calibration
from abridgeai.features.interviews.quality.scoring import (
    ContingencyReport,
    JudgedTurn,
    LeadingReport,
    parse_verdict,
)
from abridgeai.features.interviews.quality.transcript import (
    TranscriptTurn,
    build_qa_pairs,
    followup_pairs,
)

# ── calibration ──────────────────────────────────────────────────────────────


def test_over_confidence_is_flagged() -> None:
    """Runtime called it sufficient (2 points); the evaluator said not met."""
    report = compute_calibration(
        session_id="s1",
        runtime_coverage={"o1": {"coverage_points": 2, "evidence_count": 1}},
        verdicts={"o1": False},
    )
    assert report.scored == 1
    assert [o.outcome_id for o in report.over_confident] == ["o1"]
    assert report.under_confident == []
    assert report.agreement_rate == 0.0
    assert report.over_confidence_rate == 1.0


def test_under_confidence_is_flagged() -> None:
    report = compute_calibration(
        session_id="s1",
        runtime_coverage={"o1": {"coverage_points": 1}},
        verdicts={"o1": True},
    )
    assert [o.outcome_id for o in report.under_confident] == ["o1"]
    assert report.over_confident == []


def test_agreement_both_directions() -> None:
    report = compute_calibration(
        session_id="s1",
        runtime_coverage={"o1": {"coverage_points": 2}, "o2": {"coverage_points": 0}},
        verdicts={"o1": True, "o2": False},
    )
    assert report.agreement_rate == 1.0
    assert report.over_confidence_rate == 0.0


def test_outcomes_without_verdicts_are_skipped() -> None:
    """An unjudged outcome has nothing to calibrate against."""
    report = compute_calibration(
        session_id="s1",
        runtime_coverage={"o1": {"coverage_points": 2}, "o2": {"coverage_points": 2}},
        verdicts={"o1": True},
    )
    assert report.scored == 1


def test_verdicts_without_runtime_state_are_skipped() -> None:
    """Legacy/non-adaptive session: no runtime belief to compare."""
    report = compute_calibration(
        session_id="s1", runtime_coverage={}, verdicts={"o1": True}
    )
    assert report.scored == 0
    assert report.agreement_rate is None
    assert report.over_confidence_rate is None


def test_malformed_coverage_points_treated_as_zero() -> None:
    """JSONB is untyped; a junk value must not crash or look like coverage."""
    report = compute_calibration(
        session_id="s1",
        runtime_coverage={"o1": {"coverage_points": "lots"}},
        verdicts={"o1": True},
    )
    assert report.outcomes[0].coverage_points == 0
    assert report.outcomes[0].under_confident is True


def test_bool_coverage_points_not_counted_as_int() -> None:
    """``True`` is an int subclass in Python; it is not 1 coverage point."""
    report = compute_calibration(
        session_id="s1",
        runtime_coverage={"o1": {"coverage_points": True}},
        verdicts={"o1": False},
    )
    assert report.outcomes[0].coverage_points == 0


# ── transcript shaping / follow-up detection ─────────────────────────────────


def _t(mid: str, role: str, text: str, sqid: str | None) -> TranscriptTurn:
    return TranscriptTurn(message_id=mid, role=role, text=text, session_question_id=sqid)


def test_followup_detected_when_same_session_question() -> None:
    turns = [
        _t("m1", "ai", "What is an index?", "q1"),
        _t("m2", "user", "It speeds up lookups.", "q1"),
        _t("m3", "ai", "How does it speed them up?", "q1"),  # follow-up on q1
        _t("m4", "user", "A B-tree.", "q1"),
    ]
    pairs = followup_pairs(turns)
    assert [p.interviewer_message_id for p in pairs] == ["m3"]
    assert pairs[0].preceding_student_text == "It speeds up lookups."


def test_new_question_is_not_a_followup() -> None:
    turns = [
        _t("m1", "ai", "What is an index?", "q1"),
        _t("m2", "user", "It speeds up lookups.", "q1"),
        _t("m3", "ai", "Now, what is a transaction?", "q2"),  # new question
        _t("m4", "user", "Atomic unit of work.", "q2"),
    ]
    assert followup_pairs(turns) == []
    # ...but it is still judged for leading questions.
    assert [p.interviewer_message_id for p in build_qa_pairs(turns)] == ["m1", "m3"]


def test_scripted_interview_yields_no_followups() -> None:
    """The failure mode the metric exists to catch."""
    turns = [
        _t("m1", "ai", "Q1?", "q1"),
        _t("m2", "user", "A1.", "q1"),
        _t("m3", "ai", "Q2?", "q2"),
        _t("m4", "user", "A2.", "q2"),
        _t("m5", "ai", "Q3?", "q3"),
        _t("m6", "user", "A3.", "q3"),
    ]
    assert followup_pairs(turns) == []


def test_system_messages_and_blanks_ignored() -> None:
    turns = [
        _t("s0", "system", "session started", None),
        _t("m1", "ai", "   ", "q1"),  # blank AI turn
        _t("m2", "ai", "Real question?", "q1"),
        _t("m3", "user", "Answer.", "q1"),
    ]
    pairs = build_qa_pairs(turns)
    assert [p.interviewer_message_id for p in pairs] == ["m2"]


def test_trailing_interviewer_turn_without_answer_is_skipped() -> None:
    """A closing remark has no answer to judge contamination against."""
    turns = [
        _t("m1", "ai", "Question?", "q1"),
        _t("m2", "user", "Answer.", "q1"),
        _t("m3", "ai", "Thanks, that's all.", "q1"),
    ]
    assert [p.interviewer_message_id for p in build_qa_pairs(turns)] == ["m1"]


def test_missing_session_question_id_is_never_a_followup() -> None:
    """Without ids we cannot prove it is a follow-up, so we do not claim it."""
    turns = [
        _t("m1", "ai", "Question?", None),
        _t("m2", "user", "Answer.", None),
        _t("m3", "ai", "Tell me more.", None),
        _t("m4", "user", "More.", None),
    ]
    assert followup_pairs(turns) == []


# ── judge verdict parsing ────────────────────────────────────────────────────


def test_parse_valid_verdict() -> None:
    turn = parse_verdict(
        {
            "score": 4,
            "explanation": "picks up the B-tree claim",
            "grounded_in": "B-tree",
            "fabricated_premise": False,
        },
        message_id="m3",
    )
    assert turn is not None
    assert turn.score == 4
    assert turn.grounded_in == "B-tree"


def test_parse_accepts_stringified_score() -> None:
    """SparkMe's judges emit the score as a string; tolerate it."""
    turn = parse_verdict({"score": "2"}, message_id="m1")
    assert turn is not None
    assert turn.score == 2


def test_parse_rejects_out_of_range_and_junk() -> None:
    for payload in ({"score": 0}, {"score": 6}, {"score": "high"}, {}, None, {"score": True}):
        assert parse_verdict(payload, message_id="m1") is None


def test_unparseable_verdict_is_excluded_not_scored_one() -> None:
    """The bias SparkMe's eval_flow has and we deliberately avoid.

    A broken judge must look like missing data, never like a bad interviewer.
    """
    good = parse_verdict({"score": 5}, message_id="m1")
    assert good is not None
    report = ContingencyReport(
        session_id="s1", turns=[good], errors=["m2: unparseable judge verdict"]
    )
    assert report.mean_score == 5.0  # the failed turn did NOT drag it toward 1
    assert len(report.errors) == 1


# ── aggregation ──────────────────────────────────────────────────────────────


def test_contingency_flags_weak_and_fabricated() -> None:
    report = ContingencyReport(
        session_id="s1",
        turns=[
            JudgedTurn(message_id="m1", score=5),
            JudgedTurn(message_id="m2", score=2),
            JudgedTurn(message_id="m3", score=4, fabricated_premise=True),
        ],
    )
    assert report.mean_score == (5 + 2 + 4) / 3
    assert [t.message_id for t in report.weak_turns] == ["m2"]
    assert [t.message_id for t in report.fabricated_premises] == ["m3"]


def test_no_followups_is_reported_as_a_finding() -> None:
    report = ContingencyReport(session_id="s1", no_followups=True)
    assert report.no_followups is True
    assert report.mean_score is None
    assert report.to_dict()["no_followups"] is True


def test_leading_flags_contamination() -> None:
    report = LeadingReport(
        session_id="s1",
        turns=[
            JudgedTurn(message_id="m1", score=5),
            JudgedTurn(
                message_id="m2",
                score=1,
                leaked_content="indexes use B-trees",
                answer_echoes_question=True,
            ),
        ],
    )
    assert [t.message_id for t in report.leading_turns] == ["m2"]
    assert [t.message_id for t in report.contaminated_turns] == ["m2"]
    assert report.mean_score == 3.0


def test_empty_reports_are_none_not_zero() -> None:
    """No data must not read as a perfect (or terrible) score."""
    assert LeadingReport(session_id="s1").mean_score is None
    assert ContingencyReport(session_id="s1").mean_score is None
