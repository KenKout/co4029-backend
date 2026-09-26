"""Boundary validation for SR review scheduling."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from abridgeai.features.spaced_repetition.services import review


@pytest.mark.asyncio
async def test_non_positive_expected_time_fails_before_card_state_init(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invalid legacy T_exp data must not leave a dormant card state behind."""
    question_id = uuid4()
    student_id = uuid4()
    quiz_id = uuid4()
    load_or_init = AsyncMock()

    async def _t_exp(_db: object, _question_id: object) -> int:
        return 0

    async def _context(_db: object, _question_id: object) -> object:
        return SimpleNamespace(quiz_id=quiz_id, initial_ef=None)

    monkeypatch.setattr(review, "get_t_exp_for_question", _t_exp)
    monkeypatch.setattr(review, "get_question_with_quiz_context", _context)
    monkeypatch.setattr(review, "_load_or_init_state", load_or_init)

    with pytest.raises(ValueError, match="positive expected_response_time_ms"):
        await review.record_card_review(
            object(),
            student_id=student_id,
            question_id=question_id,
            quiz_attempt_id=None,
            t_actual_ms=1_000,
            correct=True,
            hint_used=False,
        )

    load_or_init.assert_not_awaited()
