"""The bound on how far a review may recede, and the option to stop there.

SM-2's recurrence is ``I(n) = I(n-1) * EF`` with nothing above it. With EF
capped at 2.5 the intervals run 1, 6, 15, 38, 94, 234, 586, 1465 days -- four
years after eight clean reviews, which is well past the end of any course the
question belongs to.

``apply_interval_ceiling`` answers two different wishes about that, and they
pull opposite ways:

* a **ceiling** is Anki's ``Maximum Interval``: the card still comes due, just
  never later than the bound. Lowering it means MORE review, not less.
* a **finish line** stops scheduling at the bound instead.

The second is not an SM-2 idea and not an Anki one -- Anki's only automatic
removal is leech suspension, which fires on repeated failure -- so it is off
by default and the tests below pin that.
"""

from __future__ import annotations

import pytest

from abridgeai.features.spaced_repetition.sm2 import apply_interval_ceiling


class TestTheCeiling:
    def test_an_interval_under_the_bound_is_untouched(self) -> None:
        assert apply_interval_ceiling(30, max_interval_days=365, retire_beyond=False) == (
            30,
            False,
        )

    def test_an_interval_over_the_bound_is_clamped_to_it(self) -> None:
        """Anki's semantics: the card keeps coming, at the bound's cadence."""
        assert apply_interval_ceiling(5000, max_interval_days=365, retire_beyond=False) == (
            365,
            False,
        )

    def test_the_bound_itself_is_allowed(self) -> None:
        assert apply_interval_ceiling(365, max_interval_days=365, retire_beyond=False) == (
            365,
            False,
        )

    def test_the_default_bound_clamps_nothing_reachable(self) -> None:
        """36500 days is Anki's own default and this one matches it.

        Eight clean reviews at the EF ceiling reach ~1465 days, so nothing a
        student can produce comes near it. The setting exists to be lowered,
        not to bite out of the box.
        """
        assert apply_interval_ceiling(1465, max_interval_days=36500, retire_beyond=True) == (
            1465,
            False,
        )


class TestTheFinishLine:
    def test_exceeding_the_bound_retires_the_card(self) -> None:
        interval, retired = apply_interval_ceiling(
            5000, max_interval_days=365, retire_beyond=True
        )

        assert retired is True
        assert interval == 365, "the last interval is still recorded, not the raw 5000"

    def test_reaching_the_bound_exactly_does_not_retire(self) -> None:
        """Retirement needs the interval to EXCEED the bound.

        An installation that sets the bound to the interval it wants cards to
        settle at should get that interval. Retiring on equality would make
        the bound unreachable as a resting cadence.
        """
        assert apply_interval_ceiling(365, max_interval_days=365, retire_beyond=True) == (
            365,
            False,
        )

    def test_retirement_is_off_unless_asked_for(self) -> None:
        """The same interval, the same bound, and the only difference is the
        flag -- which is the whole point of keeping them separate settings."""
        _, retired = apply_interval_ceiling(
            5000, max_interval_days=365, retire_beyond=False
        )

        assert retired is False

    @pytest.mark.parametrize("interval", [366, 500, 10_000])
    def test_anything_past_the_bound_retires_equally(self, interval: int) -> None:
        """One day over and ten thousand days over are the same decision.

        How far past the bound a card landed says nothing extra: it is a
        product of the previous interval and EF, not of how well the student
        knows it.
        """
        _, retired = apply_interval_ceiling(
            interval, max_interval_days=365, retire_beyond=True
        )

        assert retired is True
