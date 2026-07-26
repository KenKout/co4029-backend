"""Clamping of client-reported answer time (``t_actual_ms``).

The frontend measures per-question ATTENTION time so the signal keeps working
when several questions share one screen (pagination 1/5/10/All). It cannot cap
the value itself: the student-facing payload deliberately omits
``expected_response_time_ms``, so only the server knows the ceiling.

These tests pin two properties:

1. Clamping never changes the derived Q — everything above rho=1 already floors
   to Q=3 for a correct unhinted answer, so the cap is free.
2. Pathological inputs (walked away from the desk, clock skew, crafted payload)
   can't write absurd values into ``t_actual_ms`` for analytics to read later.
"""

from __future__ import annotations

import pytest

from abridgeai.features.spaced_repetition.services.review import (
    T_ACTUAL_CAP_MULTIPLIER,
    _clamp_t_actual,
)
from abridgeai.features.spaced_repetition.sm2.q_derivation import derive_q

T_EXP = 60_000  # 60s


class TestClampBounds:
    def test_leaves_normal_times_untouched(self) -> None:
        for ms in (0, 1_000, 30_000, 59_999, T_EXP, 120_000, 180_000):
            assert _clamp_t_actual(ms, T_EXP) == ms

    def test_caps_at_the_multiplier(self) -> None:
        ceiling = T_EXP * T_ACTUAL_CAP_MULTIPLIER
        assert _clamp_t_actual(ceiling, T_EXP) == ceiling
        assert _clamp_t_actual(ceiling + 1, T_EXP) == ceiling

    def test_caps_pathological_values(self) -> None:
        one_hour = 3_600_000
        one_week = 604_800_000
        assert _clamp_t_actual(one_hour, T_EXP) == T_EXP * T_ACTUAL_CAP_MULTIPLIER
        assert _clamp_t_actual(one_week, T_EXP) == T_EXP * T_ACTUAL_CAP_MULTIPLIER

    def test_floors_negative_values(self) -> None:
        # Schema enforces ge=0; defence in depth for non-HTTP callers.
        assert _clamp_t_actual(-1, T_EXP) == 0
        assert _clamp_t_actual(-999_999, T_EXP) == 0

    def test_passes_through_when_expected_is_unusable(self) -> None:
        # Guard against div-by-zero style surprises; derive_q raises on 0 anyway.
        assert _clamp_t_actual(5_000, 0) == 5_000
        assert _clamp_t_actual(5_000, -10) == 5_000


class TestClampDoesNotChangeQ:
    """The whole justification for the cap: it costs zero model fidelity."""

    @pytest.mark.parametrize(
        "t_actual",
        [
            0,
            1,
            29_999,
            30_000,  # rho = 0.5 boundary
            59_999,
            60_000,  # rho = 1.0 boundary
            60_001,
            180_000,  # exactly the cap
            180_001,  # just over
            3_600_000,  # an hour
            604_800_000,  # a week
        ],
    )
    @pytest.mark.parametrize("hint_used", [False, True])
    def test_q_identical_before_and_after_clamping(
        self, t_actual: int, hint_used: bool
    ) -> None:
        raw_q = derive_q(
            correct=True, hint_used=hint_used, t_actual_ms=t_actual, t_exp_ms=T_EXP
        )
        clamped_q = derive_q(
            correct=True,
            hint_used=hint_used,
            t_actual_ms=_clamp_t_actual(t_actual, T_EXP),
            t_exp_ms=T_EXP,
        )
        assert raw_q == clamped_q

    def test_incorrect_stays_zero_regardless(self) -> None:
        for t_actual in (0, 60_000, 604_800_000):
            assert (
                derive_q(
                    correct=False,
                    hint_used=False,
                    t_actual_ms=_clamp_t_actual(t_actual, T_EXP),
                    t_exp_ms=T_EXP,
                )
                == 0
            )


class TestQBucketsAreCoarse:
    """Documents WHY per-question millisecond precision isn't required."""

    def test_unhinted_correct_has_exactly_three_outcomes(self) -> None:
        seen = {
            derive_q(
                correct=True, hint_used=False, t_actual_ms=ms, t_exp_ms=T_EXP
            )
            for ms in range(0, 400_000, 500)
        }
        assert seen == {3, 4, 5}

    def test_boundaries_are_half_and_full_expected_time(self) -> None:
        q = lambda ms: derive_q(  # noqa: E731
            correct=True, hint_used=False, t_actual_ms=ms, t_exp_ms=T_EXP
        )
        assert q(29_999) == 5
        assert q(30_000) == 4
        assert q(59_999) == 4
        assert q(60_000) == 3
