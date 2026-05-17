from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta, timezone

import pytest

from abridgeai.features.spaced_repetition.sm2.scheduler import (
    apply_jitter,
    next_due_at,
    next_interval_days,
)


def test_next_interval_failure_resets_to_one_day() -> None:
    assert next_interval_days(ef=2.5, n=5, q=0, prev_interval=100) == 1
    assert next_interval_days(ef=2.5, n=5, q=2, prev_interval=100) == 1


def test_next_interval_first_review_is_one_day() -> None:
    assert next_interval_days(ef=2.5, n=0, q=5, prev_interval=0) == 1
    assert next_interval_days(ef=1.3, n=0, q=3, prev_interval=0) == 1


def test_next_interval_second_review_is_six_days() -> None:
    assert next_interval_days(ef=2.5, n=1, q=5, prev_interval=1) == 6
    assert next_interval_days(ef=1.3, n=1, q=3, prev_interval=1) == 6


def test_next_interval_third_review_uses_ef_recurrence() -> None:
    assert next_interval_days(ef=2.5, n=2, q=5, prev_interval=6) == 15
    assert next_interval_days(ef=2.5, n=2, q=5, prev_interval=10) == 25


def test_next_interval_subsequent_reviews_use_ef_recurrence() -> None:
    assert next_interval_days(ef=2.5, n=3, q=5, prev_interval=15) == 38


def test_next_interval_min_one_day_floor() -> None:
    assert next_interval_days(ef=2.5, n=2, q=5, prev_interval=0) == 1


def test_apply_jitter_zero_fraction_returns_unchanged() -> None:
    assert apply_jitter(10, fraction=0) == 10
    assert apply_jitter(100, fraction=0) == 100


def test_apply_jitter_within_bounds() -> None:
    rng = random.Random(42)  # noqa: S311 - non-crypto jitter test
    interval = 100
    fraction = 0.1
    tolerance = round(interval * fraction) + 1
    for _ in range(100):
        result = apply_jitter(interval, fraction=fraction, rng=rng)
        assert abs(result - interval) <= tolerance


def test_apply_jitter_min_one_day_floor() -> None:
    rng = random.Random(0)  # noqa: S311 - non-crypto jitter test
    for _ in range(50):
        result = apply_jitter(2, fraction=0.5, rng=rng)
        assert result >= 1


def test_apply_jitter_deterministic_with_seed() -> None:
    rng_a = random.Random(1234)  # noqa: S311 - non-crypto jitter test
    rng_b = random.Random(1234)  # noqa: S311 - non-crypto jitter test
    interval = 30
    seq_a = [apply_jitter(interval, fraction=0.1, rng=rng_a) for _ in range(20)]
    seq_b = [apply_jitter(interval, fraction=0.1, rng=rng_b) for _ in range(20)]
    assert seq_a == seq_b


def test_apply_jitter_invalid_fraction_raises() -> None:
    with pytest.raises(ValueError, match="fraction must be in"):
        apply_jitter(10, fraction=-0.1)
    with pytest.raises(ValueError, match="fraction must be in"):
        apply_jitter(10, fraction=1.0)
    with pytest.raises(ValueError, match="fraction must be in"):
        apply_jitter(10, fraction=2.0)


def test_apply_jitter_short_intervals_not_dead_zone() -> None:
    """BUG-1 regression guard: integer-truncation jitter was dead for intervals 1-9 days.

    Float-math jitter (round(interval * (1 + epsilon))) is active across all
    short intervals once fraction is large enough to push round() past 0.5.
    For intervals 6-9 the default fraction=0.1 is sufficient (this covers the
    critical n=1 SM-2 cohort at 6 days). For intervals 1-5 the result domain
    is too narrow at 0.1, so we test with fraction=0.9 to prove the value is
    not hardcoded.
    """
    for interval in range(6, 10):
        rng = random.Random(interval * 17 + 3)  # noqa: S311 - non-crypto jitter test
        results = [apply_jitter(interval, fraction=0.1, rng=rng) for _ in range(50)]
        assert any(r != interval for r in results), (
            f"jitter dead zone re-introduced at interval={interval} "
            f"(all 50 trials returned {interval})"
        )
    for interval in range(1, 6):
        rng = random.Random(interval * 17 + 3)  # noqa: S311 - non-crypto jitter test
        results = [apply_jitter(interval, fraction=0.9, rng=rng) for _ in range(50)]
        assert any(r != interval for r in results), (
            f"jitter dead at interval={interval} even with fraction=0.9 "
            f"(all 50 trials returned {interval})"
        )


def test_apply_jitter_active_for_n1_cohort_6_days() -> None:
    """The n=1 SM-2 cohort fixes interval at 6 days; jitter MUST apply there."""
    rng = random.Random(42)  # noqa: S311 - non-crypto jitter test
    results = [apply_jitter(6, fraction=0.1, rng=rng) for _ in range(100)]
    assert any(r != 6 for r in results)


def test_apply_jitter_floor_at_one_day() -> None:
    rng = random.Random(0)  # noqa: S311 - non-crypto jitter test
    for _ in range(100):
        result = apply_jitter(1, fraction=0.5, rng=rng)
        assert result >= 1


def test_next_due_at_requires_tz_aware_now() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        next_due_at(now=datetime(2025, 1, 1), interval_days=10)


def test_next_due_at_returns_tz_aware_future() -> None:
    now = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
    result = next_due_at(now=now, interval_days=10, jitter_fraction=0)
    assert result.tzinfo is not None
    assert result == now + timedelta(days=10)


def test_next_due_at_deterministic_with_seeded_rng() -> None:
    now = datetime(2025, 6, 15, 9, 30, 0, tzinfo=UTC)
    rng_a = random.Random(2024)  # noqa: S311 - non-crypto jitter test
    rng_b = random.Random(2024)  # noqa: S311 - non-crypto jitter test
    result_a = next_due_at(now=now, interval_days=30, jitter_fraction=0.1, rng=rng_a)
    result_b = next_due_at(now=now, interval_days=30, jitter_fraction=0.1, rng=rng_b)
    assert result_a == result_b


def test_next_due_at_jitter_within_bounds() -> None:
    now = datetime(2025, 1, 1, tzinfo=UTC)
    rng = random.Random(7)  # noqa: S311 - non-crypto jitter test
    result = next_due_at(now=now, interval_days=100, jitter_fraction=0.1, rng=rng)
    delta_days = (result - now).days
    assert 90 <= delta_days <= 110


def test_next_due_at_accepts_non_utc_tz() -> None:
    tz_jst = timezone(timedelta(hours=9))
    now = datetime(2025, 1, 1, tzinfo=tz_jst)
    result = next_due_at(now=now, interval_days=5, jitter_fraction=0)
    assert result == now + timedelta(days=5)
