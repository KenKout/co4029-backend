"""Phase 2: unit tests for the review-visibility window resolver."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from abridgeai.features.quizzes.schemas.review_options import ReviewOptions
from abridgeai.features.quizzes.services.review_visibility import (
    IMMEDIATE_WINDOW_MINUTES,
    resolve_review_visibility,
    resolve_review_window,
)

UTC = timezone.utc


def _quiz(close_at=None, review_options=None):
    return SimpleNamespace(available_until=close_at, review_options=review_options)


def _attempt(submitted_at):
    return SimpleNamespace(submitted_at=submitted_at)


def test_window_after_close_when_past_close():
    now = datetime(2026, 1, 10, tzinfo=UTC)
    q = _quiz(close_at=datetime(2026, 1, 1, tzinfo=UTC))
    a = _attempt(datetime(2026, 1, 1, tzinfo=UTC))
    assert resolve_review_window(q, a, now) == "after_close"


def test_window_immediately_after_within_grace():
    now = datetime(2026, 1, 10, 12, 0, tzinfo=UTC)
    q = _quiz(close_at=None)
    a = _attempt(now - timedelta(minutes=IMMEDIATE_WINDOW_MINUTES - 1))
    assert resolve_review_window(q, a, now) == "immediately_after"


def test_window_later_while_open_after_grace():
    now = datetime(2026, 1, 10, 12, 0, tzinfo=UTC)
    q = _quiz(close_at=None)
    a = _attempt(now - timedelta(minutes=IMMEDIATE_WINDOW_MINUTES + 5))
    assert resolve_review_window(q, a, now) == "later_while_open"


def test_visibility_reads_flags_for_active_window():
    now = datetime(2026, 1, 10, 12, 0, tzinfo=UTC)
    opts = ReviewOptions.model_validate({"after_close": {"show_correct_answers": False}})
    q = _quiz(
        close_at=datetime(2026, 1, 1, tzinfo=UTC), review_options=opts.model_dump()
    )
    a = _attempt(datetime(2026, 1, 1, tzinfo=UTC))
    vis = resolve_review_visibility(q, a, now)
    assert vis.show_correct_answers is False
    assert vis.show_score is True


def test_visibility_defaults_all_true_when_options_missing():
    now = datetime(2026, 1, 10, 12, 0, tzinfo=UTC)
    q = _quiz(close_at=None, review_options=None)
    a = _attempt(now)
    vis = resolve_review_visibility(q, a, now)
    assert all(
        getattr(vis, f)
        for f in (
            "show_score",
            "show_correctness",
            "show_correct_answers",
            "show_explanation",
            "show_points",
        )
    )
