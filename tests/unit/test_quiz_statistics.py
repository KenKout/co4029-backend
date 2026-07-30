"""Unit tests for the pure point-biserial discrimination index (Phase 10)."""

from __future__ import annotations

from abridgeai.features.quizzes.services.statistics import point_biserial


def test_point_biserial_perfect_positive():
    # high scorers right, low scorers wrong -> strongly positive
    flags = [True, True, False, False]
    totals = [90.0, 80.0, 40.0, 30.0]
    r, note = point_biserial(flags, totals)
    assert note is None
    assert r is not None and r > 0.9


def test_point_biserial_perfect_negative():
    flags = [False, False, True, True]
    totals = [90.0, 80.0, 40.0, 30.0]
    r, note = point_biserial(flags, totals)
    assert r is not None and r < -0.9


def test_point_biserial_all_correct_returns_none():
    r, note = point_biserial([True, True, True], [50.0, 60.0, 70.0])
    assert r is None and note is not None


def test_point_biserial_single_student_returns_none():
    r, note = point_biserial([True], [50.0])
    assert r is None and note is not None


def test_point_biserial_zero_total_variance_returns_none():
    r, note = point_biserial([True, False], [50.0, 50.0])
    assert r is None and note is not None
