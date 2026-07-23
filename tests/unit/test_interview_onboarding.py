from __future__ import annotations

import pytest

from abridgeai.features.interviews.services.onboarding import (
    _guided_text,
    _language_choice,
    _natural_decision,
)


@pytest.mark.parametrize(
    "text",
    [
        "Yes, that's me and the audio is clear.",
        "I'm ready to begin.",
        "Vâng, tôi nghe rõ.",
        "Tôi đã sẵn sàng.",
    ],
)
def test_natural_confirmation_advances(text: str) -> None:
    assert _natural_decision(text) == "advance"


@pytest.mark.parametrize(
    "text",
    [
        "I am not ready yet.",
        "I can't hear you.",
        "Tôi chưa sẵn sàng.",
        "Tôi không nghe rõ.",
    ],
)
def test_natural_problem_response_holds(text: str) -> None:
    assert _natural_decision(text) == "hold"


def test_ambiguous_response_requests_clarification() -> None:
    assert _natural_decision("Maybe later this afternoon") == "unclear"


@pytest.mark.parametrize(
    ("text", "expected"),
    [("English", "en"), ("Tiếng Việt", "vi"), ("Vietnamese please", "vi")],
)
def test_language_choice_is_detected(text: str, expected: str) -> None:
    assert _language_choice(text) == expected
    assert _natural_decision(text, "language_check") == "advance"


@pytest.mark.parametrize(
    ("language", "expected"),
    [("en", "Skip the setup."), ("vi", "Bỏ qua phần thiết lập.")],
)
def test_skip_setup_has_guided_text(language: str, expected: str) -> None:
    # The skip action must carry a non-empty guided response in both languages
    # so the transcript records a coherent user turn when setup is fast-forwarded.
    assert _guided_text("skip_setup", language) == expected
