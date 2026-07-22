"""Unit tests for the pure phase-progression policy (Slice 7).

No DB, no LLM — plain dataclass inputs → next phase / difficulty bias.
"""

from __future__ import annotations

from abridgeai.features.interviews.orchestrator.phases import (
    PhaseInputs,
    next_phase,
    phase_difficulty_bias,
)
from abridgeai.features.interviews.orchestrator.state import InterviewPhase


def _inp(**kw: object) -> PhaseInputs:
    base: dict[str, object] = {
        "current_phase": InterviewPhase.OPENING,
        "turns_in_phase": 0,
        "all_required_covered": False,
        "time_fraction_remaining": 1.0,
        "depth_budget_remaining": True,
        "warmup_turns_target": 1,
    }
    base.update(kw)
    return PhaseInputs(**base)  # type: ignore[arg-type]


def test_opening_advances_to_warmup_after_first_turn() -> None:
    got = next_phase(_inp(current_phase=InterviewPhase.OPENING, turns_in_phase=1))
    assert got is InterviewPhase.WARMUP


def test_opening_stays_until_first_turn_completes() -> None:
    got = next_phase(_inp(current_phase=InterviewPhase.OPENING, turns_in_phase=0))
    assert got is InterviewPhase.OPENING


def test_warmup_advances_to_core_after_target() -> None:
    got = next_phase(_inp(current_phase=InterviewPhase.WARMUP, turns_in_phase=1))
    assert got is InterviewPhase.CORE


def test_warmup_stays_below_target() -> None:
    got = next_phase(
        _inp(current_phase=InterviewPhase.WARMUP, turns_in_phase=0, warmup_turns_target=2)
    )
    assert got is InterviewPhase.WARMUP


def test_core_enters_deep_probe_when_covered_and_time_and_budget() -> None:
    got = next_phase(
        _inp(
            current_phase=InterviewPhase.CORE,
            all_required_covered=True,
            time_fraction_remaining=0.6,
            depth_budget_remaining=True,
        )
    )
    assert got is InterviewPhase.DEEP_PROBE


def test_core_closes_when_covered_but_low_time() -> None:
    got = next_phase(
        _inp(
            current_phase=InterviewPhase.CORE,
            all_required_covered=True,
            time_fraction_remaining=0.05,
        )
    )
    assert got is InterviewPhase.CLOSING


def test_core_closes_when_covered_but_no_depth_budget() -> None:
    got = next_phase(
        _inp(
            current_phase=InterviewPhase.CORE,
            all_required_covered=True,
            time_fraction_remaining=0.6,
            depth_budget_remaining=False,
        )
    )
    assert got is InterviewPhase.CLOSING


def test_core_stays_when_not_covered_and_time_ok() -> None:
    got = next_phase(
        _inp(
            current_phase=InterviewPhase.CORE,
            all_required_covered=False,
            time_fraction_remaining=0.6,
        )
    )
    assert got is InterviewPhase.CORE


def test_core_closes_when_not_covered_but_out_of_time() -> None:
    got = next_phase(
        _inp(
            current_phase=InterviewPhase.CORE,
            all_required_covered=False,
            time_fraction_remaining=0.05,
        )
    )
    assert got is InterviewPhase.CLOSING


def test_deep_probe_closes_when_budget_gone() -> None:
    got = next_phase(_inp(current_phase=InterviewPhase.DEEP_PROBE, depth_budget_remaining=False))
    assert got is InterviewPhase.CLOSING


def test_deep_probe_closes_when_low_time() -> None:
    got = next_phase(_inp(current_phase=InterviewPhase.DEEP_PROBE, time_fraction_remaining=0.05))
    assert got is InterviewPhase.CLOSING


def test_deep_probe_stays_with_budget_and_time() -> None:
    got = next_phase(
        _inp(
            current_phase=InterviewPhase.DEEP_PROBE,
            depth_budget_remaining=True,
            time_fraction_remaining=0.6,
        )
    )
    assert got is InterviewPhase.DEEP_PROBE


def test_closing_is_terminal() -> None:
    got = next_phase(_inp(current_phase=InterviewPhase.CLOSING))
    assert got is InterviewPhase.CLOSING


def test_untimed_session_treated_as_full_time() -> None:
    # time_fraction_remaining=None (untimed) must not force closing.
    got = next_phase(
        _inp(
            current_phase=InterviewPhase.CORE,
            all_required_covered=True,
            time_fraction_remaining=None,
            depth_budget_remaining=True,
        )
    )
    assert got is InterviewPhase.DEEP_PROBE


def test_phase_difficulty_bias_monotonic() -> None:
    assert phase_difficulty_bias(InterviewPhase.WARMUP) < phase_difficulty_bias(InterviewPhase.CORE)
    assert phase_difficulty_bias(InterviewPhase.CORE) < phase_difficulty_bias(
        InterviewPhase.DEEP_PROBE
    )


def test_phase_difficulty_bias_warmup_and_closing_ease_off() -> None:
    assert phase_difficulty_bias(InterviewPhase.WARMUP) == -1
    assert phase_difficulty_bias(InterviewPhase.CLOSING) == -1
    assert phase_difficulty_bias(InterviewPhase.DEEP_PROBE) == 1
