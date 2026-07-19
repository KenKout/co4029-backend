"""Unit tests for adaptive-interviewer perception (Phase 2 + 3).

Pure, fast, no DB / no LLM — covers the deterministic intent rules, the
permissive intent parser, and the tolerant answer-analysis parser + fallback.
These are the safety nets that keep a session progressing when the model is
unavailable or returns garbage, so they must be exhaustively unit-tested.
"""

from __future__ import annotations

import pytest

from abridgeai.features.interviews.orchestrator.analysis import (
    AnswerAnalysis,
    Correctness,
    ProbeType,
    Relevance,
    fallback_analysis,
    parse_analysis_response,
)
from abridgeai.features.interviews.orchestrator.intent import (
    NON_ACADEMIC_INTENTS,
    StudentIntent,
    classify_by_rules,
    parse_intent_response,
)

# ── Intent: deterministic rules ──────────────────────────────────────────────


@pytest.mark.parametrize(
    ("utterance", "expected"),
    [
        ("Could you repeat the question?", StudentIntent.ASK_TO_REPEAT),
        ("Can you say that again", StudentIntent.ASK_TO_REPEAT),
        ("What do you mean by granularity?", StudentIntent.ASK_FOR_CLARIFICATION),
        ("Can we skip this question", StudentIntent.SKIP_QUESTION),
        ("I don't know", StudentIntent.CANNOT_ANSWER),
        ("I'm not sure", StudentIntent.CANNOT_ANSWER),
        ("My microphone is not working", StudentIntent.TECHNICAL_ISSUE),
        ("Can we end the interview", StudentIntent.END_INTERVIEW),
        ("Let me think", StudentIntent.ASK_FOR_MORE_TIME),
    ],
)
def test_rules_match_english(utterance: str, expected: StudentIntent) -> None:
    verdict = classify_by_rules(utterance)
    assert verdict is not None
    assert verdict.intent is expected
    assert verdict.source == "rules"
    assert verdict.confidence >= 0.9


@pytest.mark.parametrize(
    ("utterance", "expected"),
    [
        ("Bạn có thể nhắc lại câu hỏi không?", StudentIntent.ASK_TO_REPEAT),
        ("Ý bạn là gì?", StudentIntent.ASK_FOR_CLARIFICATION),
        ("Cho mình qua câu này", StudentIntent.SKIP_QUESTION),
        ("Tôi không biết", StudentIntent.CANNOT_ANSWER),
        ("Cho tôi thêm chút thời gian", StudentIntent.ASK_FOR_MORE_TIME),
    ],
)
def test_rules_match_vietnamese(utterance: str, expected: StudentIntent) -> None:
    verdict = classify_by_rules(utterance)
    assert verdict is not None
    assert verdict.intent is expected


def test_rules_return_none_for_genuine_answer() -> None:
    # A real content answer should NOT match any rule — defer to the LLM.
    assert classify_by_rules("A fact table stores measurable business events.") is None


def test_rules_empty_utterance_is_cannot_answer() -> None:
    verdict = classify_by_rules("   ")
    assert verdict is not None
    assert verdict.intent is StudentIntent.CANNOT_ANSWER


def test_non_academic_intents_membership() -> None:
    # Guard: repeat/clarify/skip/technical/more-time/end must be non-academic;
    # answer/partial must NOT be (they are the only scorable intents).
    assert StudentIntent.ASK_TO_REPEAT in NON_ACADEMIC_INTENTS
    assert StudentIntent.TECHNICAL_ISSUE in NON_ACADEMIC_INTENTS
    assert StudentIntent.ANSWER not in NON_ACADEMIC_INTENTS
    assert StudentIntent.PARTIAL_ANSWER not in NON_ACADEMIC_INTENTS


# ── Intent: parser ───────────────────────────────────────────────────────────


def test_parse_intent_valid() -> None:
    parsed = parse_intent_response(
        {"intent": "skip_question", "confidence": 0.8, "rationale": "asked to move on"}
    )
    assert parsed is not None
    assert parsed.intent is StudentIntent.SKIP_QUESTION
    assert parsed.confidence == 0.8
    assert parsed.source == "llm"


def test_parse_intent_unknown_value_returns_none() -> None:
    assert parse_intent_response({"intent": "banana"}) is None


def test_parse_intent_non_mapping_returns_none() -> None:
    assert parse_intent_response(None) is None
    assert parse_intent_response("nope") is None  # type: ignore[arg-type]


def test_parse_intent_clamps_confidence() -> None:
    parsed = parse_intent_response({"intent": "answer", "confidence": 5.0})
    assert parsed is not None
    assert parsed.confidence == 1.0


# ── Answer analysis: parser + fallback ───────────────────────────────────────


def test_parse_analysis_full() -> None:
    payload = {
        "relevance": "relevant",
        "completeness": "complete",
        "correctness": "correct",
        "specificity": "specific",
        "has_concrete_example": True,
        "identified_concepts": ["fact table", "grain"],
        "missing_concepts": [],
        "misconceptions": [],
        "contradictions": [],
        "recommended_probe_type": "explore_tradeoff",
        "provisional_quality_score": 0.9,
        "confidence": 0.85,
        "evidence": [
            {
                "outcome_id": "o1",
                "turn_id": "t1",
                "evidence_type": "supports",
                "summary": "correctly defined the grain",
                "confidence": 0.9,
            }
        ],
    }
    analysis = parse_analysis_response(payload, default_turn_id="t1")
    assert analysis is not None
    assert analysis.relevance is Relevance.RELEVANT
    assert analysis.correctness is Correctness.CORRECT
    assert analysis.recommended_probe_type is ProbeType.EXPLORE_TRADEOFF
    assert len(analysis.evidence) == 1
    assert analysis.evidence[0].outcome_id == "o1"
    assert analysis.evidence[0].turn_id == "t1"


def test_parse_analysis_unknown_enums_fall_back_safely() -> None:
    # Garbage enum values must degrade to the safe buckets, not raise.
    analysis = parse_analysis_response(
        {"relevance": "???", "correctness": "banana", "recommended_probe_type": "xyz"},
        default_turn_id="t9",
    )
    assert analysis is not None
    assert analysis.relevance is Relevance.OFF_TOPIC
    assert analysis.correctness is Correctness.NOT_ASSESSABLE
    assert analysis.recommended_probe_type is ProbeType.NONE


def test_parse_analysis_evidence_missing_turn_id_uses_default() -> None:
    analysis = parse_analysis_response(
        {"evidence": [{"outcome_id": "o2", "summary": "partial"}]},
        default_turn_id="default-turn",
    )
    assert analysis is not None
    assert analysis.evidence[0].turn_id == "default-turn"


def test_parse_analysis_non_mapping_returns_none() -> None:
    assert parse_analysis_response(None, default_turn_id="t") is None
    assert parse_analysis_response([], default_turn_id="t") is None  # type: ignore[arg-type]


def test_fallback_analysis_is_not_assessable() -> None:
    fb = fallback_analysis("t1")
    assert isinstance(fb, AnswerAnalysis)
    assert fb.correctness is Correctness.NOT_ASSESSABLE
    assert fb.recommended_probe_type is ProbeType.NONE
    assert fb.evidence == []
    assert fb.confidence == 0.0


def test_analysis_roundtrips_through_dict() -> None:
    # to_dict → from_dict must preserve the structured shape (state persistence).
    original = parse_analysis_response(
        {
            "relevance": "partially_relevant",
            "completeness": "partial",
            "correctness": "mixed",
            "specificity": "general",
            "misconceptions": ["confused grain with cardinality"],
            "evidence": [
                {"outcome_id": "o1", "turn_id": "t1", "evidence_type": "partially_supports"}
            ],
        },
        default_turn_id="t1",
    )
    assert original is not None
    restored = AnswerAnalysis.from_dict(original.to_dict(), default_turn_id="t1")
    assert restored.relevance is original.relevance
    assert restored.correctness is original.correctness
    assert restored.misconceptions == original.misconceptions
    assert restored.evidence[0].outcome_id == "o1"
