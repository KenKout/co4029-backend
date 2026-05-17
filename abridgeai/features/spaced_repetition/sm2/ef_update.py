"""SM-2 EF (Easiness Factor) update with calibration dampening (thesis section 5)."""

from __future__ import annotations

EF_MIN = 1.3


def update_ef(ef_old: float, q: int, n: int, alpha: float = 0.6) -> float:
    """SM-2 EF update with calibration dampening.

    During the first 3 repetitions (n <= 3), positive deltas are multiplied
    by ``alpha`` (default 0.6) to slow EF growth and mitigate early over-
    confidence. Negative deltas are NOT calibrated so the forgetting signal
    is preserved.

    Args:
        ef_old: previous EF.
        q: 0-5 grade.
        n: repetition count BEFORE this review (0 = first ever).
        alpha: calibration weight; must be in (0, 1].

    Returns:
        New EF, floored at 1.3, rounded to 4 decimals.

    Raises:
        ValueError: if alpha is outside (0, 1] or q is outside [0, 5].
    """
    if not (0 < alpha <= 1):
        raise ValueError(f"alpha must be in (0, 1]; got {alpha}")
    if not (0 <= q <= 5):
        raise ValueError(f"q must be 0-5; got {q}")
    # Standard SM-2 EF delta formula (Wozniak): 0.1 - (5-q)*(0.08 + (5-q)*0.02)
    delta = 0.1 - (5 - q) * (0.08 + (5 - q) * 0.02)
    # Calibration dampens growth during early reps; preserves forgetting signal
    if n <= 3 and delta > 0:
        delta *= alpha
    ef_new = max(EF_MIN, ef_old + delta)
    return round(ef_new, 4)


__all__ = ["EF_MIN", "update_ef"]
