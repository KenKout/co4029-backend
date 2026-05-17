"""SM-2 interval scheduling with jitter (thesis section 5)."""

from __future__ import annotations

import random
from datetime import datetime, timedelta


def next_interval_days(*, ef: float, n: int, q: int, prev_interval: int) -> int:
    """SM-2 interval recurrence.

    Args:
        ef: current EF (after update).
        n: repetition count BEFORE this review (0 = first ever).
        q: 0-5 grade for THIS review.
        prev_interval: previous interval in days.

    Returns:
        Number of days until next review (>= 1).

    Rules:
        - q < 3 (failure): reset -> 1 day
        - q >= 3, n == 0 (first review): 1 day
        - q >= 3, n == 1 (second review): 6 days
        - q >= 3, n >= 2: round(prev_interval * ef), floored at 1
    """
    if q < 3:
        return 1
    if n == 0:
        return 1
    if n == 1:
        return 6
    return max(1, round(prev_interval * ef))


def apply_jitter(
    interval_days: int,
    *,
    fraction: float = 0.1,
    rng: random.Random | None = None,
) -> int:
    """Add +/- ``fraction`` jitter to ``interval_days`` to prevent review pile-up.

    Uses float math (``round(interval * (1 + epsilon))``) so jitter applies
    even for short intervals (e.g. n=1 cohort at 6 days gets +/- 0.6 -> +/- 1
    day). The integer-truncation form previously used here had a dead zone
    for ``interval_days in [1, 9]`` with ``fraction=0.1`` (BUG-1).

    ``fraction=0.0`` disables jitter (deterministic). ``fraction=0.1`` yields
    a result in roughly ``[interval*0.9, interval*1.1]``. The result is
    floored at 1 day.

    Args:
        interval_days: base interval in days.
        fraction: jitter magnitude in [0, 1). 0 disables jitter.
        rng: optional ``random.Random`` for determinism in tests.

    Raises:
        ValueError: if ``fraction`` is outside [0, 1).
    """
    if not (0 <= fraction < 1):
        raise ValueError(f"fraction must be in [0, 1); got {fraction}")
    if fraction == 0.0:
        return interval_days
    actual_rng = rng if rng is not None else random.Random()  # noqa: S311 - non-crypto jitter  # nosec B311
    epsilon = actual_rng.uniform(-fraction, fraction)
    jittered = round(interval_days * (1 + epsilon))
    return max(1, jittered)


def next_due_at(
    *,
    now: datetime,
    interval_days: int,
    jitter_fraction: float = 0.1,
    rng: random.Random | None = None,
) -> datetime:
    """Compose interval + jitter into an absolute UTC-aware due timestamp.

    Args:
        now: timezone-aware reference datetime (must have tzinfo).
        interval_days: pre-computed base interval in days.
        jitter_fraction: jitter magnitude passed to :func:`apply_jitter`.
        rng: optional RNG for determinism in tests.

    Raises:
        ValueError: if ``now`` is naive (no tzinfo).
    """
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware (use UTC)")
    jittered = apply_jitter(interval_days, fraction=jitter_fraction, rng=rng)
    return now + timedelta(days=jittered)


__all__ = ["apply_jitter", "next_due_at", "next_interval_days"]
