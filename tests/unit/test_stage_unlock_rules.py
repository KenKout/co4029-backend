"""Stage unlock: a gate opens only for a student who reached it.

``satisfied(course)`` has no path in it — it is a completion award on
(student, course). So a stage of path A can evaluate complete from work done
entirely on path B, whose courses happen to mirror it, the moment A is
joined. That is deliberate and correct for progress: the work was really
done, it counts toward A, and it latches.

What it must not do is open the rest of A. ``_apply_unlock`` used to read
only ``evals[idx - 1].complete``, so a stage the student was never allowed to
enter — locked, and complete only by transfer — served as the prerequisite
pass for the stage after it. A curriculum whose stage 1 is a hard
requirement could be entered at stage 3 without stage 1 ever being touched.

These pin the chained gate, and pin that it changed nothing for a student
progressing normally.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from abridgeai.features.career_paths.services.stages import StageEval, _apply_unlock


def _stage(
    position: int,
    policy: str,
    *,
    required: int = 2,
    satisfied: int = 0,
    optional: int = 0,
    satisfied_optional: int = 0,
    min_optional: int = 0,
) -> StageEval:
    """One stage at ``position``, gated by ``policy``."""
    courses: list[dict[str, object]] = [
        {"is_required": True, "satisfied": i < satisfied} for i in range(required)
    ]
    courses += [
        {"is_required": False, "satisfied": i < satisfied_optional} for i in range(optional)
    ]
    return StageEval(
        stage=SimpleNamespace(
            position=position,
            unlock_policy=policy,
            enforcement="hard",
            min_optional_to_complete=min_optional,
        ),
        courses=courses,
    )


def _unlocked(evals: list[StageEval]) -> list[bool]:
    _apply_unlock(evals)
    return [ev.unlocked for ev in evals]


class TestTheCrossPathBypass:
    """The defect: completion earned elsewhere must not open a later stage."""

    def test_a_stage_completed_on_another_path_does_not_unlock_the_next(self) -> None:
        # Path A: stage 1 untouched; stage 2 fully satisfied because its
        # courses were completed on path B; stage 3 gated on stage 2.
        evals = [
            _stage(1, "always", satisfied=0),
            _stage(2, "after_previous", satisfied=2),
            _stage(3, "after_previous", satisfied=0),
        ]
        assert _unlocked(evals) == [True, False, False]
        # Stage 2 really is complete — the work counts and will latch. It is
        # simply not a route into stage 3.
        assert evals[1].live_complete is True

    def test_the_same_holds_for_the_required_only_gate(self) -> None:
        evals = [
            _stage(1, "always", satisfied=0),
            _stage(2, "after_previous_required", satisfied=2),
            _stage(3, "after_previous_required", satisfied=0),
        ]
        assert _unlocked(evals) == [True, False, False]

    def test_a_whole_run_of_transferred_stages_stays_shut(self) -> None:
        """Two mirrored stages in a row must not compound into an opening."""
        evals = [
            _stage(1, "always", satisfied=0),
            _stage(2, "after_previous", satisfied=2),
            _stage(3, "after_previous", satisfied=2),
            _stage(4, "after_previous", satisfied=0),
        ]
        assert _unlocked(evals) == [True, False, False, False]


class TestNormalProgressionIsUnchanged:
    """The chain must cost nothing to a student walking the path in order."""

    def test_first_stage_is_always_open(self) -> None:
        evals = [_stage(1, "after_previous", satisfied=0)]
        assert _unlocked(evals) == [True]

    def test_clearing_stage_one_opens_stage_two_only(self) -> None:
        evals = [
            _stage(1, "always", satisfied=2),
            _stage(2, "after_previous", satisfied=0),
            _stage(3, "after_previous", satisfied=0),
        ]
        assert _unlocked(evals) == [True, True, False]

    def test_clearing_two_stages_opens_the_third(self) -> None:
        evals = [
            _stage(1, "always", satisfied=2),
            _stage(2, "after_previous", satisfied=2),
            _stage(3, "after_previous", satisfied=0),
        ]
        assert _unlocked(evals) == [True, True, True]

    def test_required_only_gate_opens_with_electives_outstanding(self) -> None:
        """``after_previous_required`` ignores the previous stage's quota."""
        evals = [
            _stage(1, "always", satisfied=2, optional=3, satisfied_optional=0, min_optional=2),
            _stage(2, "after_previous_required", satisfied=0),
        ]
        assert _unlocked(evals) == [True, True]
        # The previous stage is NOT complete — only its required work is done.
        assert evals[0].live_complete is False

    def test_a_latched_stage_still_opens_the_next(self) -> None:
        """The latch is how completion survives a demotion; it must still gate."""
        first = _stage(1, "always", satisfied=0)
        first.latched = True
        evals = [first, _stage(2, "after_previous", satisfied=0)]
        assert _unlocked(evals) == [True, True]


class TestPoliciesThatDoNotChain:
    def test_always_is_open_regardless_of_what_precedes_it(self) -> None:
        evals = [
            _stage(1, "always", satisfied=0),
            _stage(2, "always", satisfied=0),
            _stage(3, "after_previous", satisfied=0),
        ]
        assert _unlocked(evals) == [True, True, False]

    def test_an_always_stage_propagates_rather_than_blocking(self) -> None:
        """An unfinished ``always`` stage must not shut the rest of the path.

        It is unlocked, so the chain passes through it: stage 3 is decided by
        whether stage 2 is complete, exactly as before.
        """
        evals = [
            _stage(1, "always", satisfied=2),
            _stage(2, "always", satisfied=2),
            _stage(3, "after_previous", satisfied=0),
        ]
        assert _unlocked(evals) == [True, True, True]

    @pytest.mark.parametrize("policy", ["", "sometimes", "AFTER_PREVIOUS", None])
    def test_an_unknown_policy_fails_closed(self, policy: str | None) -> None:
        evals = [
            _stage(1, "always", satisfied=2),
            _stage(2, policy or "unknown", satisfied=0),
        ]
        assert _unlocked(evals) == [True, False]
