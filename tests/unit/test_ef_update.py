from __future__ import annotations

import pytest

from abridgeai.features.spaced_repetition.sm2.ef_update import EF_MAX, EF_MIN, update_ef


def test_update_ef_q4_neutral_unchanged() -> None:
    assert update_ef(2.5, q=4, n=0) == 2.5
    assert update_ef(2.5, q=4, n=10) == 2.5


def test_update_ef_q5_calibrated_during_early_reps() -> None:
    # From 2.4 (below the 2.5 ceiling) a perfect review grows EF by
    # 0.1 * alpha; from 2.5 the ceiling pins it at 2.5.
    assert update_ef(2.4, q=5, n=0, alpha=0.6) == 2.46
    assert update_ef(2.4, q=5, n=1, alpha=0.6) == 2.46
    assert update_ef(2.4, q=5, n=3, alpha=0.6) == 2.46
    assert update_ef(2.5, q=5, n=0, alpha=0.6) == 2.5


def test_update_ef_q5_full_after_calibration_window() -> None:
    # A perfect review adds +0.1, which would take 2.5 → 2.6; the ceiling
    # caps it at 2.5 so EF stays within the KR-normalisation range.
    assert update_ef(2.5, q=5, n=4, alpha=0.6) == EF_MAX
    assert update_ef(2.5, q=5, n=10, alpha=0.6) == EF_MAX


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
    assert update_ef(2.5, q=5, n=0, alpha=1.0) == EF_MAX


def test_update_ef_ceiling_enforced() -> None:
    # Any input above the ceiling must never come back above 2.5 — this is
    # the invariant the KR estimate (kr_estimate.sql / class_kr_distribution.sql)
    # relies on to stay in [0, 1]. Previously uncapped, a run of perfect
    # reviews drifted EF to 2.6+ and KR silently exceeded 100%.
    assert EF_MAX == 2.5
    assert update_ef(2.5, q=5, n=10, positive_delta_scale=1.0) == EF_MAX
    assert update_ef(2.6, q=5, n=10) == EF_MAX
    assert update_ef(3.0, q=5, n=10) == EF_MAX
    assert update_ef(2.5, q=5, n=0, alpha=1.0) == EF_MAX
    # Dampened positive deltas below the ceiling still apply.
    assert update_ef(2.4, q=5, n=10, positive_delta_scale=0.75) == 2.475
    # The floor is unaffected: negative deltas still drop EF toward 1.3.
    assert update_ef(1.5, q=0, n=10) == EF_MIN


# -- guess-channel dampening (positive_delta_scale) --------------------------


def test_update_ef_guess_scale_dampens_positive_delta() -> None:
    # n=10 (calibration off) so we isolate the guess scale. Q=5 delta = 0.1.
    # Start at 2.4 so the 2.5 ceiling doesn't swallow the dampened delta.
    # 4-option MCQ → guess prob 0.25 → scale 0.75 → delta 0.075 → 2.475.
    assert update_ef(2.4, q=5, n=10, positive_delta_scale=0.75) == 2.475
    # true/false → guess prob 0.5 → scale 0.5 → delta 0.05 → 2.45.
    assert update_ef(2.4, q=5, n=10, positive_delta_scale=0.5) == 2.45


def test_update_ef_guess_scale_default_is_no_effect() -> None:
    # Default 1.0 must reproduce the pre-feature value exactly (capped at 2.5).
    assert update_ef(2.5, q=5, n=10) == update_ef(
        2.5, q=5, n=10, positive_delta_scale=1.0
    )
    assert update_ef(2.5, q=5, n=10, positive_delta_scale=1.0) == EF_MAX


def test_update_ef_guess_scale_never_touches_negative_delta() -> None:
    # A wrong answer's EF drop must be identical regardless of format — the
    # forgetting signal is never dampened by the guess channel.
    assert update_ef(2.5, q=0, n=10, positive_delta_scale=0.5) == update_ef(
        2.5, q=0, n=10, positive_delta_scale=1.0
    )
    assert update_ef(2.5, q=2, n=10, positive_delta_scale=0.25) == pytest.approx(
        2.18
    )


def test_update_ef_guess_scale_stacks_with_calibration() -> None:
    # Early rep (n=0) MCQ: both alpha (0.6) and guess scale (0.75) apply to the
    # positive delta → 0.1 * 0.6 * 0.75 = 0.045 → 2.445 (base 2.4, under cap).
    assert update_ef(2.4, q=5, n=0, alpha=0.6, positive_delta_scale=0.75) == 2.445


@pytest.mark.parametrize("scale", [0, -0.1, 1.5, 2.0])
def test_update_ef_invalid_guess_scale_raises(scale: float) -> None:
    with pytest.raises(ValueError, match="positive_delta_scale must be in"):
        update_ef(2.5, q=5, n=0, positive_delta_scale=scale)
