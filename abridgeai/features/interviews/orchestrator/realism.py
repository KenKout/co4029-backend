"""Realism-cluster tone signals for the adaptive interviewer.

Extracted from ``adaptive.py`` when that file hit its LOC budget: the helpers
are pure (state-free, LLM-free), which is exactly the sibling-module criterion
the locsize guard documents. Behaviour is unchanged — the same deterministic
``detect_affect`` backs the rambling signal so the decision-time signal and the
tone lead-in never disagree.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from abridgeai.features.interviews.orchestrator.affect import Affect, detect_affect

if TYPE_CHECKING:
    from abridgeai.features.interviews.orchestrator.analysis import AnswerAnalysis


def is_rambling(
    *,
    answer_text: str,
    analysis: AnswerAnalysis | None,
    enabled: bool,
) -> bool:
    """Whether the candidate is rambling (Slice 17, v2), gated on the flag.

    Uses the same deterministic ``detect_affect`` as the tone layer (single
    source of truth) so the decision-time signal and the tone lead-in never
    disagree. Off → False → the decision is byte-for-byte v1.
    """
    if not enabled:
        return False
    return detect_affect(answer_text=answer_text, analysis=analysis) is Affect.RAMBLING


# Communication-polish thresholds (Slice 20, v2).
COMMS_TIME_PRESSURE_FRACTION = 0.2  # signal to prioritise below this time fraction
COMMS_RECOVERY_WEAK_STREAK = 2  # rebuild after this many consecutive weak answers


def comms_polish_signals(
    *,
    time_fraction_remaining: float | None,
    consecutive_weak_answers: int,
    enabled: bool,
) -> tuple[bool, bool]:
    """Return ``(time_pressure, recovery)`` tone signals (Slice 20, v2).

    ``time_pressure`` fires when little time remains so the interviewer can tell
    the candidate to prioritise; ``recovery`` fires after a weak streak so a
    rattled candidate gets an encouraging, scoped lead-in. TONE ONLY — these
    feed the utterance lead-in, never the decision. Off → (False, False) → v1.
    """
    if not enabled:
        return False, False
    time_pressure = (
        time_fraction_remaining is not None
        and time_fraction_remaining <= COMMS_TIME_PRESSURE_FRACTION
    )
    recovery = consecutive_weak_answers >= COMMS_RECOVERY_WEAK_STREAK
    return time_pressure, recovery
