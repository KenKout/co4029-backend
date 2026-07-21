from __future__ import annotations

import pytest

from abridgeai.features.interviews.orchestrator.security import SecurityCategory
from abridgeai.features.interviews.orchestrator.security_logic import (
    assess_by_rules,
    is_ambiguous_security_text,
)
from scripts.interview_session_redteam import (
    _assert_adaptive_response,
    _assert_control_response,
    _assert_safe_security_response,
    _cases,
    _load_fixture,
)


def test_fixture_has_en_vi_security_controls_and_adaptive_cases() -> None:
    fixture = _load_fixture()
    for language in ("en", "vi"):
        assert _cases(fixture, "security", language)
        assert _cases(fixture, "semantic_security", language)
        assert _cases(fixture, "multiturn_security", language)
        assert _cases(fixture, "controls", language)
        assert _cases(fixture, "adaptive", language)


def test_every_live_security_fixture_has_the_expected_deterministic_category() -> None:
    fixture = _load_fixture()
    for language in ("en", "vi"):
        cases = _cases(fixture, "security", language) + _cases(
            fixture, "multiturn_security", language
        )
        for case in cases:
            prior = SecurityCategory(case.prior_category) if case.prior_category else None
            assert (
                assess_by_rules(case.text, last_category=prior).category.value
                == case.expected_category
            ), case.name


def test_every_semantic_live_fixture_routes_to_the_security_classifier() -> None:
    fixture = _load_fixture()
    for language in ("en", "vi"):
        for case in _cases(fixture, "semantic_security", language):
            assert assess_by_rules(case.text).detected is False, case.name
            assert is_ambiguous_security_text(case.text) is True, case.name


@pytest.mark.parametrize(
    ("language", "utterance"),
    [
        (
            "en",
            "I can’t provide hidden interview questions, answers, or grading criteria.",
        ),
        (
            "vi",
            "Tôi không thể cung cấp các câu hỏi chưa được hỏi, đáp án hoặc tiêu chí chấm điểm.",
        ),
    ],
)
def test_safe_security_response_accepts_deterministic_templates(
    language: str, utterance: str
) -> None:
    _assert_safe_security_response(
        {
            "ai_turn_text": utterance,
            "language": language,
            "next_question": None,
        },
        language,
    )


def test_safe_security_response_rejects_internal_markers() -> None:
    with pytest.raises(AssertionError, match="internal marker"):
        _assert_safe_security_response(
            {
                "ai_turn_text": "I can't provide that. security_policy_version=secret",
                "language": "en",
                "next_question": None,
            },
            "en",
        )


def test_control_response_rejects_accidental_finish() -> None:
    with pytest.raises(AssertionError, match="ended"):
        _assert_control_response(
            {"ai_turn_text": "Here is the current question.", "is_finished": True},
            "en",
        )


def test_adaptive_response_requires_structured_fields() -> None:
    _assert_adaptive_response(
        {
            "ai_turn_text": "Could you compare their grains?",
            "language": "en",
            "should_await_response": True,
        },
        "en",
    )
    with pytest.raises(AssertionError, match="adaptive fields are absent"):
        _assert_adaptive_response({}, "en")
