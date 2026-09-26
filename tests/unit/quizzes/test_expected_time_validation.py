"""Positive expected-time invariants at teacher write boundaries."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from abridgeai.core.exceptions import AppError
from abridgeai.features.quizzes.schemas.bank import (
    QuizQuestionBankItemCreate,
    QuizQuestionBankItemUpdate,
)
from abridgeai.features.quizzes.services.authoring import (
    _validated_expected_response_time_ms,
)


def test_authoring_expected_time_allows_null_or_positive_integer() -> None:
    assert _validated_expected_response_time_ms(None) is None
    assert _validated_expected_response_time_ms(1) == 1
    assert _validated_expected_response_time_ms(60_000) == 60_000


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "1000"])
def test_authoring_expected_time_rejects_non_positive_or_non_integer(value: object) -> None:
    with pytest.raises(AppError, match="positive integer"):
        _validated_expected_response_time_ms(value)


def test_curated_bank_rejects_zero_expected_time() -> None:
    with pytest.raises(ValidationError):
        QuizQuestionBankItemCreate(
            question_type="short_answer",
            prompt_text="Explain photosynthesis.",
            expected_response_time_ms=0,
        )

    with pytest.raises(ValidationError):
        QuizQuestionBankItemUpdate(expected_response_time_ms=0)
