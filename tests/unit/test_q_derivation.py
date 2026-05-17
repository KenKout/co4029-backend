from __future__ import annotations

import pytest

from abridgeai.features.spaced_repetition.sm2.q_derivation import derive_q


@pytest.mark.parametrize(
    ("t_actual_ms", "t_exp_ms", "expected_q"),
    [
        (250, 1000, 5),
        (500, 1000, 4),
        (999, 1000, 4),
        (1000, 1000, 3),
        (5000, 1000, 3),
    ],
)
def test_derive_q_no_hint_correct_buckets(t_actual_ms: int, t_exp_ms: int, expected_q: int) -> None:
    assert (
        derive_q(
            correct=True,
            hint_used=False,
            t_actual_ms=t_actual_ms,
            t_exp_ms=t_exp_ms,
        )
        == expected_q
    )


@pytest.mark.parametrize(
    ("t_actual_ms", "t_exp_ms", "expected_q"),
    [
        (1000, 1000, 2),
        (1999, 1000, 2),
        (2000, 1000, 1),
        (5000, 1000, 1),
    ],
)
def test_derive_q_hint_correct_buckets(t_actual_ms: int, t_exp_ms: int, expected_q: int) -> None:
    assert (
        derive_q(
            correct=True,
            hint_used=True,
            t_actual_ms=t_actual_ms,
            t_exp_ms=t_exp_ms,
        )
        == expected_q
    )


@pytest.mark.parametrize("hint_used", [True, False])
@pytest.mark.parametrize("t_actual_ms", [0, 100, 1000, 99_999])
def test_derive_q_incorrect_always_zero(hint_used: bool, t_actual_ms: int) -> None:
    assert (
        derive_q(
            correct=False,
            hint_used=hint_used,
            t_actual_ms=t_actual_ms,
            t_exp_ms=1000,
        )
        == 0
    )


def test_derive_q_t_exp_zero_raises() -> None:
    with pytest.raises(ValueError, match="t_exp_ms must be > 0"):
        derive_q(correct=True, hint_used=False, t_actual_ms=100, t_exp_ms=0)
