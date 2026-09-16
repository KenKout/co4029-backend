"""The stage-aware completion predicate, and its agreement with progress.

Formula 2 credits at most ``min_optional_to_complete`` electives per stage, so
a stage with a surplus of optional courses can be finished without touching
them. Anything that decides completion has to run on that same rule.

These pin the rule itself and, more importantly, pin the answers TOGETHER —
a path is complete if and only if it reads 100% under the stage-aware formula,
and completion never consults a separate progress setting.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from abridgeai.features.career_paths.services.stages import (
    StageEval,
    path_complete,
    path_progress_percent,
)


def _stage(
    *,
    required: int,
    optional: int,
    min_optional: int,
    satisfied_required: int,
    satisfied_optional: int,
) -> StageEval:
    """One stage with the given course counts and completion state."""
    courses: list[dict[str, object]] = [
        {"is_required": True, "satisfied": i < satisfied_required} for i in range(required)
    ]
    courses += [
        {"is_required": False, "satisfied": i < satisfied_optional} for i in range(optional)
    ]
    return StageEval(
        stage=SimpleNamespace(min_optional_to_complete=min_optional),
        courses=courses,
    )


class TestQuotaIsTheRule:
    def test_quota_met_completes_with_electives_outstanding(self) -> None:
        """The case the old ``all(satisfied)`` rule got wrong.

        Two required, five optional, quota of one. The student finishes both
        required courses and one elective: four electives are untouched and
        the stage is done.
        """
        evals = [
            _stage(
                required=2,
                optional=5,
                min_optional=1,
                satisfied_required=2,
                satisfied_optional=1,
            )
        ]
        assert path_complete(evals) is True
        assert path_progress_percent(evals) == 100.0

    def test_quota_short_by_one_is_not_complete(self) -> None:
        evals = [
            _stage(
                required=2,
                optional=5,
                min_optional=2,
                satisfied_required=2,
                satisfied_optional=1,
            )
        ]
        assert path_complete(evals) is False

    def test_a_missing_required_course_is_never_excused(self) -> None:
        """Surplus electives do not substitute for required work."""
        evals = [
            _stage(
                required=2,
                optional=5,
                min_optional=1,
                satisfied_required=1,
                satisfied_optional=5,
            )
        ]
        assert path_complete(evals) is False

    def test_stage_with_no_measurable_work_is_complete(self) -> None:
        """Pure electives, no quota: nothing is owed, so nothing is missing."""
        evals = [
            _stage(
                required=0,
                optional=3,
                min_optional=0,
                satisfied_required=0,
                satisfied_optional=0,
            )
        ]
        assert path_complete(evals) is True

    def test_every_stage_must_be_complete(self) -> None:
        evals = [
            _stage(
                required=1,
                optional=0,
                min_optional=0,
                satisfied_required=1,
                satisfied_optional=0,
            ),
            _stage(
                required=2,
                optional=1,
                min_optional=1,
                satisfied_required=2,
                satisfied_optional=0,
            ),
        ]
        assert path_complete(evals) is False


class TestCompletionAgreesWithProgress:
    """The invariant that stops the gap coming back.

    ``stage_done`` can never exceed ``stage_total``, so the sum reaches the
    total only when every stage does. Any future rule that breaks this
    equivalence reintroduces a path that reads 100% while something built on
    completion stays open.
    """

    @pytest.mark.parametrize(
        ("required", "optional", "min_optional", "sat_required", "sat_optional"),
        [
            (2, 5, 1, 2, 1),  # quota met, electives outstanding
            (2, 5, 1, 2, 5),  # everything done
            (2, 5, 2, 2, 1),  # quota short
            (2, 5, 1, 1, 5),  # required outstanding
            (0, 3, 0, 0, 0),  # nothing measurable
            (1, 0, 0, 0, 0),  # single required, untouched
            (3, 2, 2, 3, 2),  # quota exactly consumed
        ],
    )
    def test_complete_iff_one_hundred_percent(
        self,
        required: int,
        optional: int,
        min_optional: int,
        sat_required: int,
        sat_optional: int,
    ) -> None:
        evals = [
            _stage(
                required=required,
                optional=optional,
                min_optional=min_optional,
                satisfied_required=sat_required,
                satisfied_optional=sat_optional,
            )
        ]
        assert path_complete(evals) is (path_progress_percent(evals) == 100.0)

    def test_agreement_holds_across_several_stages(self) -> None:
        done = _stage(
            required=2,
            optional=4,
            min_optional=1,
            satisfied_required=2,
            satisfied_optional=1,
        )
        pending = _stage(
            required=1,
            optional=0,
            min_optional=0,
            satisfied_required=0,
            satisfied_optional=0,
        )
        assert path_complete([done, pending]) is False
        assert path_progress_percent([done, pending]) < 100.0
        assert path_complete([done, done]) is True
        assert path_progress_percent([done, done]) == 100.0
