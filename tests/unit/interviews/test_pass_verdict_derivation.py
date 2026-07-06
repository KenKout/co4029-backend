"""Unit tests for the pass-verdict derivation (thesis §4.3 threshold rule).

``_derive_pass_verdict`` is the single place that turns per-outcome verdicts +
the teacher-configured ``min_outcomes_to_pass`` into the session pass/fail. Pure
function — no DB, no LLM.
"""

from __future__ import annotations

from uuid import uuid4

from abridgeai.features.interviews.ai.stages.evaluation.outcome_verdicts import (
    OutcomeVerdict,
    build_outcome_verdicts,
)
from abridgeai.features.interviews.services.evaluation import _derive_pass_verdict


def _verdicts(met_flags: list[bool]):
    return build_outcome_verdicts(
        [
            OutcomeVerdict(outcome_id=uuid4(), met=flag, reasoning="r", evidence=None)
            for flag in met_flags
        ]
    )


def test_pass_when_met_count_meets_threshold() -> None:
    verdicts = _verdicts([True, True, False])  # 2 met
    assert _derive_pass_verdict(verdicts, min_outcomes_to_pass=2) is True


def test_fail_when_below_threshold() -> None:
    verdicts = _verdicts([True, False, False])  # 1 met
    assert _derive_pass_verdict(verdicts, min_outcomes_to_pass=2) is False


def test_pass_exactly_at_threshold() -> None:
    verdicts = _verdicts([True, True])
    assert _derive_pass_verdict(verdicts, min_outcomes_to_pass=2) is True


def test_null_threshold_requires_all_met() -> None:
    all_met = _verdicts([True, True, True])
    assert _derive_pass_verdict(all_met, min_outcomes_to_pass=None) is True

    one_missing = _verdicts([True, True, False])
    assert _derive_pass_verdict(one_missing, min_outcomes_to_pass=None) is False


def test_no_outcomes_never_passes() -> None:
    empty = build_outcome_verdicts([])
    assert _derive_pass_verdict(empty, min_outcomes_to_pass=None) is False
    assert _derive_pass_verdict(empty, min_outcomes_to_pass=0) is False
