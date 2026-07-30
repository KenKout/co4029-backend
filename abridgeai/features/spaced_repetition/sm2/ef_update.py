"""SM-2 EF (Easiness Factor) update with calibration dampening (thesis section 5)."""

from __future__ import annotations

EF_MIN = 1.3


def update_ef(
    ef_old: float,
    q: int,
    n: int,
    alpha: float = 0.6,
    *,
    positive_delta_scale: float = 1.0,
) -> float:
    """SM-2 EF update with calibration dampening.

    During the first 3 repetitions (n <= 3), positive deltas are multiplied
    by ``alpha`` (default 0.6) to slow EF growth and mitigate early over-
    confidence. Negative deltas are NOT calibrated so the forgetting signal
    is preserved.

    ``positive_delta_scale`` (default 1.0 = no effect) applies a SECOND,
    independent multiplier to positive deltas only. It exists for guess-channel
    dampening: a correct multiple-choice answer carries a 1/N chance of being a
    lucky guess, so its EF growth is scaled by the non-guess probability
    ``1 - 1/N``. This deliberately does NOT touch ``q`` (the thesis Q-derivation
    is unchanged and still recorded verbatim) — it only softens how fast a
    guessable format inflates the easiness factor, so a fast lucky MCQ can't
    balloon the interval the way genuine free recall does. Free-recall formats
    (short_answer / fill_blank) pass 1.0 and are unaffected. Like ``alpha`` it
    leaves the forgetting signal (negative deltas) untouched.

    Args:
        ef_old: previous EF.
        q: 0-5 grade.
        n: repetition count BEFORE this review (0 = first ever).
        alpha: calibration weight; must be in (0, 1].
        positive_delta_scale: guess-channel weight for positive deltas;
            must be in (0, 1].

    Returns:
        New EF, floored at 1.3, rounded to 4 decimals.

    Raises:
        ValueError: if alpha/positive_delta_scale is outside (0, 1] or q is
            outside [0, 5].
    """
    if not (0 < alpha <= 1):
        raise ValueError(f"alpha must be in (0, 1]; got {alpha}")
    if not (0 < positive_delta_scale <= 1):
        raise ValueError(
            f"positive_delta_scale must be in (0, 1]; got {positive_delta_scale}"
        )
    if not (0 <= q <= 5):
        raise ValueError(f"q must be 0-5; got {q}")
    # Standard SM-2 EF delta formula (Wozniak): 0.1 - (5-q)*(0.08 + (5-q)*0.02)
    delta = 0.1 - (5 - q) * (0.08 + (5 - q) * 0.02)
    # Calibration dampens growth during early reps; preserves forgetting signal.
    if n <= 3 and delta > 0:
        delta *= alpha
    # Guess-channel dampening: only ever shrinks a POSITIVE delta, never a
    # negative one, so a wrong answer's EF drop is unaffected.
    if delta > 0:
        delta *= positive_delta_scale
    ef_new = max(EF_MIN, ef_old + delta)
    return round(ef_new, 4)


__all__ = ["EF_MIN", "update_ef"]
