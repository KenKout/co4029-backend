"""SM-2 Q grade derivation per thesis section 5 Table 1."""

from __future__ import annotations


def derive_q(
    *,
    correct: bool,
    hint_used: bool,
    t_actual_ms: int,
    t_exp_ms: int,
) -> int:
    """Derive SM-2 Q grade (0-5) per thesis section 5 Table 1.

    Q semantics:
      0     = incorrect
      1-2   = correct WITH hint (assisted recall)
      3-5   = correct WITHOUT hint (independent recall, by speed)

    Args:
        correct: whether the answer was correct.
        hint_used: whether the student consumed a hint.
        t_actual_ms: observed answer time in milliseconds.
        t_exp_ms: teacher-set expected answer time in ms (must be > 0).

    Raises:
        ValueError: if ``t_exp_ms`` is 0; caller must validate before calling.
    """
    if t_exp_ms == 0:
        raise ValueError(
            "t_exp_ms must be > 0; caller must validate before deriving Q",
        )
    if not correct:
        return 0
    rho = t_actual_ms / t_exp_ms
    if hint_used:
        # Q in {1, 2}: rho < 2 -> 2; rho >= 2 -> 1 (thesis Table 1)
        return max(1, 2 - int(rho / 2))
    # Q in {3, 4, 5}: rho < 0.5 -> 5; 0.5 <= rho < 1 -> 4; rho >= 1 -> 3
    return max(3, 5 - int(2 * rho))


__all__ = ["derive_q"]
