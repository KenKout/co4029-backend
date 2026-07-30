"""Quiz item-statistics (Phase 10): pure discrimination-index math.

The point-biserial correlation is the discrimination index: how well a single
question separates high-scoring from low-scoring students. Pure stdlib (math /
statistics) — no numpy, no DB — so it is trivially unit-testable.
"""

from __future__ import annotations

import math
import statistics


def point_biserial(
    correct_flags: list[bool], totals: list[float]
) -> tuple[float | None, str | None]:
    """Point-biserial correlation between per-question correctness and total score.

    ``r_pb = ((M1 - M0) / s_y) * sqrt(p * q)`` where M1/M0 are the mean totals of
    students who got this question right/wrong, s_y is the population stdev of all
    totals, p is the fraction correct, q = 1 - p.

    Returns ``(value, note)``. Degenerate inputs return ``(None, note)`` rather
    than raising or producing NaN:
      * fewer than 2 students        -> insufficient attempts
      * everyone right / everyone wrong -> no variance in correctness
      * all totals identical         -> no variance in totals
    """
    n = len(correct_flags)
    if n != len(totals):
        raise ValueError("correct_flags and totals must be the same length")
    if n < 2:
        return None, "insufficient attempts"

    right = [t for flag, t in zip(correct_flags, totals, strict=True) if flag]
    wrong = [t for flag, t in zip(correct_flags, totals, strict=True) if not flag]
    p = len(right) / n
    q = 1.0 - p
    if p == 0.0 or p == 1.0:
        return None, "no variance in correctness"

    s_y = statistics.pstdev(totals)
    if s_y == 0.0:
        return None, "no variance in totals"

    m1 = statistics.mean(right)
    m0 = statistics.mean(wrong)
    r_pb = ((m1 - m0) / s_y) * math.sqrt(p * q)
    return r_pb, None


__all__ = ["point_biserial"]
