"""FR-4.13: quiz grade-band feedback — validation and band selection.

Covers TC-4.13.1. The feature had no automated test at all: a search of both
suites returned only implementation files, so every rule below — including the
half-open boundary and the special case at 100 — was resting on nothing.

Deliberately pure. ``_validate_no_overlap`` takes no session, and
``select_overall_feedback`` reaches the database only through ``list_bands``,
which is stubbed here. That keeps the boundary arithmetic — the part most
likely to be wrong and most expensive to notice — testable without Postgres,
where the integration suite cannot run at all.

NOT covered here: the second half of FR-4.13, that feedback is withheld
whenever the score itself is not disclosed. That gate lives in
``services/attempt_reading.py`` (``if vis.show_score:``) and needs a full
attempt/quiz/visibility fixture, so it belongs in the integration suite.
"""

from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace
from typing import Any

import pytest

from abridgeai.core.exceptions import AppError
from abridgeai.features.quizzes.schemas.feedback import FeedbackBandIn
from abridgeai.features.quizzes.services import feedback as fb
from abridgeai.features.quizzes.services.feedback import _validate_no_overlap


def band_in(lo: str, hi: str, text: str = "well done") -> FeedbackBandIn:
    return FeedbackBandIn(min_grade=Decimal(lo), max_grade=Decimal(hi), feedback_text=text)


def stub_band(lo: str, hi: str, text: str) -> SimpleNamespace:
    """Stands in for a ``QuizFeedback`` row — only these fields are read."""
    return SimpleNamespace(
        min_grade=Decimal(lo),
        max_grade=Decimal(hi),
        feedback_text=text,
        feedback_format="markdown",
    )


def use_bands(monkeypatch: pytest.MonkeyPatch, *bands: SimpleNamespace) -> None:
    """Serve ``bands`` from ``list_bands``, ordered as the real query returns."""
    ordered = sorted(bands, key=lambda b: b.min_grade)

    async def _fake_list_bands(_db: Any, _quiz_id: Any) -> list[SimpleNamespace]:
        return ordered

    monkeypatch.setattr(fb, "list_bands", _fake_list_bands)


# ---------------------------------------------------------------------------
# Validation — overlaps rejected, gaps allowed
# ---------------------------------------------------------------------------


def test_empty_band_set_is_valid():
    # Clearing every band is how a teacher turns the feature off.
    _validate_no_overlap([])


def test_adjacent_bands_are_valid():
    # The ranges are half-open, so prev.max == next.min is a touch, not an
    # overlap. Rejecting this would make a gapless scale impossible to express.
    _validate_no_overlap([band_in("0", "50"), band_in("50", "100")])


def test_gaps_between_bands_are_allowed():
    _validate_no_overlap([band_in("0", "40"), band_in("60", "100")])


def test_bands_are_validated_regardless_of_input_order():
    with pytest.raises(AppError, match="must not overlap"):
        _validate_no_overlap([band_in("40", "100"), band_in("0", "50")])


def test_overlapping_bands_are_rejected():
    with pytest.raises(AppError, match="must not overlap"):
        _validate_no_overlap([band_in("0", "60"), band_in("50", "100")])


def test_band_with_min_equal_to_max_is_rejected():
    with pytest.raises(AppError, match="min_grade < max_grade"):
        _validate_no_overlap([band_in("50", "50")])


def test_band_with_min_above_max_is_rejected():
    with pytest.raises(AppError, match="min_grade < max_grade"):
        _validate_no_overlap([band_in("80", "20")])


# ---------------------------------------------------------------------------
# Selection — which band a score lands in
# ---------------------------------------------------------------------------


async def test_score_inside_a_band_matches_it(monkeypatch: pytest.MonkeyPatch):
    use_bands(
        monkeypatch,
        stub_band("0", "50", "keep going"),
        stub_band("50", "100", "well done"),
    )
    band = await fb.select_overall_feedback(None, quiz_id=None, score_percent=Decimal("72.5"))
    assert band is not None
    assert band.feedback_text == "well done"


async def test_band_boundary_belongs_to_the_upper_band(
    monkeypatch: pytest.MonkeyPatch,
):
    # [min, max) — a score exactly on a boundary is the START of the next band,
    # never the end of the previous one.
    use_bands(
        monkeypatch,
        stub_band("0", "50", "keep going"),
        stub_band("50", "100", "well done"),
    )
    band = await fb.select_overall_feedback(None, quiz_id=None, score_percent=Decimal("50"))
    assert band is not None
    assert band.feedback_text == "well done"


async def test_perfect_score_matches_the_top_band(monkeypatch: pytest.MonkeyPatch):
    # The one exception to half-open: a band reaching 100 includes 100, or a
    # perfect score would fall through every band and show no feedback at all.
    use_bands(
        monkeypatch,
        stub_band("0", "50", "keep going"),
        stub_band("50", "100", "well done"),
    )
    band = await fb.select_overall_feedback(None, quiz_id=None, score_percent=Decimal("100"))
    assert band is not None
    assert band.feedback_text == "well done"


async def test_score_at_the_bottom_of_the_lowest_band_matches(
    monkeypatch: pytest.MonkeyPatch,
):
    use_bands(monkeypatch, stub_band("0", "50", "keep going"))
    band = await fb.select_overall_feedback(None, quiz_id=None, score_percent=Decimal("0"))
    assert band is not None
    assert band.feedback_text == "keep going"


async def test_score_in_a_gap_matches_nothing(monkeypatch: pytest.MonkeyPatch):
    # Gaps are legal, so a score inside one must yield no feedback rather than
    # snapping to a neighbouring band.
    use_bands(
        monkeypatch,
        stub_band("0", "40", "keep going"),
        stub_band("60", "100", "well done"),
    )
    assert await fb.select_overall_feedback(None, quiz_id=None, score_percent=Decimal("50")) is None


async def test_no_bands_configured_yields_no_feedback(
    monkeypatch: pytest.MonkeyPatch,
):
    use_bands(monkeypatch)
    assert await fb.select_overall_feedback(None, quiz_id=None, score_percent=Decimal("80")) is None


async def test_absent_score_yields_no_feedback(monkeypatch: pytest.MonkeyPatch):
    # An ungraded attempt has no score. Feedback keyed to a score the student
    # does not have would be a disclosure about work not yet marked — the same
    # reason the review path withholds it when the score is hidden.
    use_bands(monkeypatch, stub_band("0", "100", "well done"))
    assert await fb.select_overall_feedback(None, quiz_id=None, score_percent=None) is None


async def test_top_band_below_100_does_not_capture_a_perfect_score(
    monkeypatch: pytest.MonkeyPatch,
):
    # The inclusive rule keys off the band reaching 100, not off being last.
    # A scale that stops at 90 leaves 100 unmatched, which is the honest read
    # of what the teacher configured.
    use_bands(monkeypatch, stub_band("0", "90", "keep going"))
    assert (
        await fb.select_overall_feedback(None, quiz_id=None, score_percent=Decimal("100")) is None
    )
