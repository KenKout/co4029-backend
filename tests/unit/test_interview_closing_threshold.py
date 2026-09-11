"""Live closing-threshold derivation (plan §6).

``below_closing_threshold`` was snapshotted ONCE at join, so a session that
crossed the closing window mid-interview never saw the agent's urgency change:
the state reminder kept saying there was time, the agent tool kept reporting
"not closing", and auto-advance never locked — all until the hard stop. The fix
is a live derivation, ``below_closing_threshold_now()``, computed from the
monotonic clock reading and the injected TOTAL duration, against the SHARED
orchestrator fraction (``DecisionInputs.closing_time_fraction``) — no duplicate
0.1 literal for the paths to disagree on.

Fake monotonic clocks: these tests inject ``clock_read_monotonic`` and patch
``time.monotonic`` so the crossing happens WITHOUT reloading userdata — exactly
the production sequence (clock passes the threshold after join).
"""

from __future__ import annotations

import time
from typing import Any
from uuid import uuid4

from abridgeai.features.interviews.realtime.agent_userdata import InterviewUserdata

FRACTION: float = 0.1  # mirrors DecisionInputs.closing_time_fraction


def _userdata(
    *,
    remaining: int | None = 600,
    total: int | None = 1800,
    elapsed: float = 0.0,
) -> InterviewUserdata:
    """``elapsed`` = monotonic seconds since the countdown was read."""
    return InterviewUserdata(
        interview_session_id=uuid4(),
        student_id=uuid4(),
        time_remaining_seconds=remaining,
        clock_read_monotonic=time.monotonic() - elapsed,
        total_duration_seconds=total,
    )


class TestThresholdSemantics:
    def test_untimed_session_is_never_closing(self) -> None:
        assert _userdata(remaining=None).below_closing_threshold_now() is False

    def test_invalid_total_is_never_closing(self) -> None:
        assert _userdata(total=0).below_closing_threshold_now() is False

    def test_zero_total_is_never_closing(self) -> None:
        assert _userdata(total=None).below_closing_threshold_now() is False

    def test_above_threshold_is_not_closing(self) -> None:
        # 600s left of 1800 = 33% > 10%.
        assert _userdata().below_closing_threshold_now() is False

    def test_at_exact_boundary_is_closing(self) -> None:
        # 180s left of 1800 = exactly 10% — the boundary closes.
        u = _userdata(remaining=180, total=1800)
        assert u.below_closing_threshold_now() is True

    def test_below_threshold_is_closing(self) -> None:
        # 60s left of 1800 = 3.3% < 10%.
        assert _userdata(remaining=60).below_closing_threshold_now() is True

    def test_zero_remaining_is_closing(self) -> None:
        assert _userdata(remaining=0).below_closing_threshold_now() is True


class TestClockCrossesAfterJoin:
    def test_clock_passing_the_threshold_flips_without_reload(self, monkeypatch: Any) -> None:
        """Join above the threshold; time passes; NOW it closes — no reload."""
        u = _userdata(remaining=200, total=1800, elapsed=0.0)
        assert u.below_closing_threshold_now() is False

        # 100s of monotonic time elapse (no reload of the userdata): the
        # derivation reads the clock again → 100s remaining → 5.6% ≤ 10%.
        real_monotonic = time.monotonic
        monkeypatch.setattr(
            "abridgeai.features.interviews.realtime.agent_userdata.time.monotonic",
            lambda: real_monotonic() + 100.0,
        )
        assert u.below_closing_threshold_now() is True

    def test_the_old_frozen_bool_stays_false_across_the_same_crossing(self) -> None:
        """Documents WHY the frozen field is deprecated: it cannot cross."""
        u = _userdata(remaining=200, total=1800)
        assert u.below_closing_threshold is False
        # ... and the frozen value never changes no matter how much time passes,
        # which is exactly the bug. The live call is the one readers must use.

    def test_shares_the_orchestrator_fraction(self, monkeypatch: Any) -> None:
        """The boundary is the SHARED orchestrator fraction, not a local 0.1."""
        from abridgeai.features.interviews.orchestrator.decision import DecisionInputs

        # Freeze the clock: int() truncation at the boundary is sensitive to
        # the microseconds that pass between construction and the call.
        frozen = 1000.0
        monkeypatch.setattr(
            "abridgeai.features.interviews.realtime.agent_userdata.time.monotonic",
            lambda: frozen,
        )
        boundary = 1800 * DecisionInputs.closing_time_fraction

        u = _userdata(remaining=int(boundary), total=1800)
        assert u.remaining_seconds_now() == int(boundary)
        assert u.below_closing_threshold_now() is True

        u2 = _userdata(remaining=int(boundary) + 1, total=1800)
        assert u2.below_closing_threshold_now() is False
