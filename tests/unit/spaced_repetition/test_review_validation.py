"""Boundary validation for SR review scheduling."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("q", "passing"),
    [(4, True), (0, False)],
)
async def test_admin_interval_unit_controls_pass_and_failure_due_at(
    monkeypatch: pytest.MonkeyPatch,
    q: int,
    passing: bool,
) -> None:
    """The same runtime unit drives normal intervals and failure cooldown."""
    question_id = uuid4()
    student_id = uuid4()
    quiz_id = uuid4()
    state = SimpleNamespace(
        ef=Decimal("2.5"),
        interval_days=1,
        repetition_count=0,
        due_at=datetime.now(tz=UTC),
        total_reviews=0,
    )
    db = SimpleNamespace(add=lambda _row: None, flush=AsyncMock())

    monkeypatch.setattr(
        review,
        "_load_quiz_question_meta",
        AsyncMock(return_value=(30_000, quiz_id, None)),
    )
    monkeypatch.setattr(
        review,
        "_load_or_init_state",
        AsyncMock(return_value=(state, False)),
    )
    monkeypatch.setattr(review, "derive_q", lambda **_kwargs: q)
    monkeypatch.setattr(review, "get_guess_probability", AsyncMock(return_value=0.0))

    async def _setting(_db: object, key: str) -> int | bool:
        return {
            "spaced_repetition.interval_unit_seconds": 10,
            "spaced_repetition.jitter_percent": 0,
            "spaced_repetition.max_interval_days": 36500,
            "spaced_repetition.retire_beyond_max_interval": False,
        }[key]

    monkeypatch.setattr(review, "resolve_setting", _setting)

    before = datetime.now(tz=UTC)
    result = await review.record_card_review(
        db,
        student_id=student_id,
        question_id=question_id,
        quiz_attempt_id=None,
        t_actual_ms=30_000,
        correct=passing,
        hint_used=False,
    )
    after = datetime.now(tz=UTC)

    assert result.interval_after == 1
    assert result.due_at is not None
    assert before + timedelta(seconds=10) <= result.due_at
    assert result.due_at <= after + timedelta(seconds=10)
    assert (result.retry_available_at is not None) is (not passing)
