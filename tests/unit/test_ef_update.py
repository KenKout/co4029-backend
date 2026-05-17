from __future__ import annotations

import pytest

from abridgeai.features.spaced_repetition.sm2.ef_update import EF_MIN, update_ef


def test_update_ef_q4_neutral_unchanged() -> None:
    assert update_ef(2.5, q=4, n=0) == 2.5
    assert update_ef(2.5, q=4, n=10) == 2.5


def test_update_ef_q5_calibrated_during_early_reps() -> None:
    assert update_ef(2.5, q=5, n=0, alpha=0.6) == 2.56
    assert update_ef(2.5, q=5, n=1, alpha=0.6) == 2.56
    assert update_ef(2.5, q=5, n=3, alpha=0.6) == 2.56


def test_update_ef_q5_full_after_calibration_window() -> None:
    assert update_ef(2.5, q=5, n=4, alpha=0.6) == 2.6
    assert update_ef(2.5, q=5, n=10, alpha=0.6) == 2.6


def test_update_ef_q0_negative_delta_not_calibrated() -> None:
    assert update_ef(2.5, q=0, n=0) == pytest.approx(1.7)
    assert update_ef(2.5, q=0, n=1) == pytest.approx(1.7)
    assert update_ef(2.5, q=0, n=10) == pytest.approx(1.7)


def test_update_ef_q2_negative_delta_not_calibrated() -> None:
    # Q=2 -> delta = 0.1 - 3*(0.08 + 3*0.02) = -0.32
    assert update_ef(2.5, q=2, n=0) == pytest.approx(2.18)
    assert update_ef(2.5, q=2, n=10) == pytest.approx(2.18)


def test_update_ef_floor_enforced() -> None:
    assert update_ef(1.5, q=0, n=0) == EF_MIN
    assert update_ef(1.3, q=0, n=10) == EF_MIN


def test_update_ef_returns_rounded_4dp() -> None:
    result = update_ef(2.0, q=5, n=10, alpha=0.6)
    assert isinstance(result, float)
    assert result == round(result, 4)


@pytest.mark.parametrize("alpha", [0, -0.1, 1.5, 2.0])
def test_update_ef_invalid_alpha_raises(alpha: float) -> None:
    with pytest.raises(ValueError, match="alpha must be in"):
        update_ef(2.5, q=5, n=0, alpha=alpha)


@pytest.mark.parametrize("q", [-1, 6, 100])
def test_update_ef_invalid_q_raises(q: int) -> None:
    with pytest.raises(ValueError, match="q must be 0-5"):
        update_ef(2.5, q=q, n=0)


def test_update_ef_alpha_one_is_no_calibration() -> None:
    assert update_ef(2.5, q=5, n=0, alpha=1.0) == 2.6
