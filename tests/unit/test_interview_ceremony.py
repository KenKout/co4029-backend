from __future__ import annotations

from types import SimpleNamespace

import pytest

from abridgeai.features.interviews.services.ceremony import (
    audio_check_text,
    briefing_text,
    candidate_first_name,
    closing_text,
    language_check_text,
    normalize_language,
    opening_text,
    preparation_text,
)


def test_opening_is_transparent_and_asks_only_for_identity() -> None:
    result = opening_text(
        title="Backend Engineering",
        name="Mina",
        persona="supportive",
        language="en-US",
    )

    assert "Mina" in result
    assert "Backend Engineering" in result
    assert "virtual interview assistant" in result
    assert "hear me clearly" not in result
    assert "English" not in result
    assert result.count("?") == 1
    assert result.endswith("speaking with Mina?")


def test_setup_checks_are_short_single_question_turns() -> None:
    turns = [
        audio_check_text(language="en"),
        language_check_text(language="en"),
        preparation_text(language="en"),
    ]

    assert "hear me clearly" in turns[0]
    assert "English or Vietnamese" in turns[1]
    assert "moment to prepare" in turns[2]
    assert all(turn.count("?") == 1 for turn in turns)


def test_briefing_explains_truthful_rules_and_asks_readiness() -> None:
    result = briefing_text(
        title="Backend Engineering",
        time_limit_minutes=30,
        input_mode="hybrid",
        language="en",
    )

    assert "30 minutes" in result
    assert "current module criteria" in result
    assert "type or speak" in result
    assert "repeat or clarify" in result
    assert "no separate per-question limit" in result.lower()
    assert result.endswith("Are you ready to begin?")


@pytest.mark.parametrize("reason", ["natural", "ended_early", "timed_out"])
def test_closing_is_a_final_statement_for_every_reason(reason: str) -> None:
    result = closing_text(
        title="System Design",
        name="An",
        persona="neutral",
        language="vi-VN",
        reason=reason,  # type: ignore[arg-type]
    )

    assert "An" in result
    assert "System Design" in result
    assert result.endswith("Tạm biệt.")
    assert "?" not in result


def test_candidate_name_prefers_given_name_then_display_name() -> None:
    assert (
        candidate_first_name(
            SimpleNamespace(given_name="  Ada  ", display_name="Ada Lovelace")
        )
        == "Ada"
    )
    assert (
        candidate_first_name(
            SimpleNamespace(given_name=None, display_name="Grace Hopper")
        )
        == "Grace"
    )
    assert candidate_first_name(None) is None


def test_language_normalization_defaults_to_english() -> None:
    assert normalize_language("vi-VN") == "vi"
    assert normalize_language("en-GB") == "en"
    assert normalize_language(None) == "en"
