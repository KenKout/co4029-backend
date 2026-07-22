"""Deterministic phase progression (Slice 7).

Pure policy — NO DB, NO LLM. Maps runtime signals to the NEXT interview phase
and a per-phase difficulty bias the selector applies on top of the streak
target. OPENING -> WARMUP -> CORE -> DEEP_PROBE -> CLOSING. DEEP_PROBE is
entered only when all required outcomes are provisionally covered AND time +
follow-up budget remain — i.e. the candidate has cleared the bar and we push
for their ceiling. Mirrors the conventions of decision.py / selection.py so the
phase rules can be unit-tested with plain enums, independent of the DB path.
"""

from __future__ import annotations

from dataclasses import dataclass

from abridgeai.features.interviews.orchestrator.state import InterviewPhase

# Below this fraction of time remaining, head to closing regardless of phase.
_LOW_TIME = 0.1
# Need at least this much time remaining to open (or stay in) a depth pass.
_DEEP_PROBE_MIN_TIME = 0.15


@dataclass(frozen=True)
class PhaseInputs:
    """Everything the phase policy needs, decoupled from the DB.

    ``time_fraction_remaining`` is None when the interview is untimed (treated
    as fully remaining). ``depth_budget_remaining`` is False once the global
    follow-up budget is spent, which forces DEEP_PROBE to exit to CLOSING.
    """

    current_phase: InterviewPhase
    turns_in_phase: int
    all_required_covered: bool
    time_fraction_remaining: float | None
    depth_budget_remaining: bool
    warmup_turns_target: int = 1


def _time(x: float | None) -> float:
    return 1.0 if x is None else x


def next_phase(inp: PhaseInputs) -> InterviewPhase:
    """Return the phase the NEXT turn should be in given the current signals.

    Terminal phases (CLOSING / COMPLETED) are returned unchanged — once the
    interview is winding down, the phase policy never re-opens it.
    """
    t = _time(inp.time_fraction_remaining)
    if inp.current_phase is InterviewPhase.OPENING:
        return InterviewPhase.WARMUP if inp.turns_in_phase >= 1 else InterviewPhase.OPENING
    if inp.current_phase is InterviewPhase.WARMUP:
        if inp.turns_in_phase >= inp.warmup_turns_target:
            return InterviewPhase.CORE
        return InterviewPhase.WARMUP
    if inp.current_phase is InterviewPhase.CORE:
        if inp.all_required_covered:
            if t >= _DEEP_PROBE_MIN_TIME and inp.depth_budget_remaining:
                return InterviewPhase.DEEP_PROBE
            return InterviewPhase.CLOSING
        if t <= _LOW_TIME:
            return InterviewPhase.CLOSING
        return InterviewPhase.CORE
    if inp.current_phase is InterviewPhase.DEEP_PROBE:
        if not inp.depth_budget_remaining or t <= _DEEP_PROBE_MIN_TIME:
            return InterviewPhase.CLOSING
        return InterviewPhase.DEEP_PROBE
    return inp.current_phase  # CLOSING / COMPLETED are terminal here


_DIFFICULTY_BIAS: dict[InterviewPhase, int] = {
    InterviewPhase.OPENING: -1,
    InterviewPhase.WARMUP: -1,
    InterviewPhase.CORE: 0,
    InterviewPhase.DEEP_PROBE: 1,
    InterviewPhase.CLOSING: -1,
    InterviewPhase.COMPLETED: 0,
}


def phase_difficulty_bias(phase: InterviewPhase) -> int:
    """Signed nudge (-1..+1) added to the streak difficulty target."""
    return _DIFFICULTY_BIAS.get(phase, 0)


__all__ = ["PhaseInputs", "next_phase", "phase_difficulty_bias"]
