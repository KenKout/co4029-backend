"""The default Bloom distribution for a generated quiz.

When a teacher has not set a distribution themselves, this heuristic decides
the cognitive mix of the whole quiz: how many questions ask the learner to
recall, to apply, to analyse. It is rendered verbatim into the ideation prompt
(``prompts/user.j2``), so its output is an instruction to the model rather
than an internal tally.

The invariant that matters is that the parts sum to the number of questions
asked for. A distribution that sums low silently shortens the quiz; one that
sums high asks for questions the teacher did not request. The tiered
if-chain makes that easy to break in one branch while the others stay right,
which is why the sum is asserted across the whole range rather than at a few
sample points.

Pure integer arithmetic — no gateway, no database.
"""

from __future__ import annotations

import pytest

from abridgeai.features.quizzes.ai.stages.ideation.logic import (
    _default_bloom_distribution,
)


class TestTheDistributionSumsToWhatWasAsked:
    @pytest.mark.parametrize("count", list(range(1, 121)))
    def test_every_count_is_fully_allocated(self, count: int) -> None:
        assert sum(_default_bloom_distribution(count).values()) == count

    @pytest.mark.parametrize("count", [0, -1, -50])
    def test_a_nonsense_count_is_clamped_to_one_question(self, count: int) -> None:
        """Zero or negative cannot be honoured, and an empty quiz is not useful."""
        distribution = _default_bloom_distribution(count)
        assert sum(distribution.values()) == 1

    def test_large_counts_reach_the_top_of_the_taxonomy(self) -> None:
        """A long quiz should not stop at ``analyze``."""
        distribution = _default_bloom_distribution(40)
        assert "create" in distribution
        assert sum(distribution.values()) == 40

    @pytest.mark.parametrize("count", [15, 30, 60, 120])
    def test_the_trimming_loop_lands_exactly(self, count: int) -> None:
        """The largest tier over-allocates first, then trims down to the target.

        A loop that removed one too many or stopped one short would leave the
        quiz a question adrift, so the arithmetic is asserted at sizes that
        exercise it.
        """
        assert sum(_default_bloom_distribution(count).values()) == count


class TestTheShapeIsSensible:
    def test_a_single_question_is_not_asked_at_the_lowest_level(self) -> None:
        """One question should probe understanding, not bare recall."""
        assert _default_bloom_distribution(1) == {"understand": 1}

    def test_the_mix_widens_as_the_quiz_grows(self) -> None:
        """More questions should mean more levels, not more of the same one."""
        assert len(_default_bloom_distribution(3)) >= len(_default_bloom_distribution(1))
        assert len(_default_bloom_distribution(12)) >= len(_default_bloom_distribution(3))

    @pytest.mark.parametrize("count", list(range(1, 121)))
    def test_no_level_is_budgeted_zero_questions(self, count: int) -> None:
        """A level allocated nothing must not appear at all.

        The dictionary is rendered verbatim into the ideation prompt, so a
        level left at zero tells the model to target a band with no questions
        in it. Each tier computes its top level as ``count - floor``, which is
        zero at the first count the tier covers — 6 and 8 are where that bites.
        """
        assert 0 not in _default_bloom_distribution(count).values()

    @pytest.mark.parametrize("count", [6, 8])
    def test_dropping_a_zero_level_does_not_change_the_total(self, count: int) -> None:
        """The filter is safe precisely because the level contributed nothing."""
        distribution = _default_bloom_distribution(count)
        assert sum(distribution.values()) == count
        assert "analyze" not in distribution


class TestKnownShapes:
    @pytest.mark.parametrize(
        ("count", "expected"),
        [
            (1, {"understand": 1}),
            (2, {"understand": 1, "apply": 1}),
            (3, {"remember": 1, "understand": 1, "apply": 1}),
            (5, {"remember": 1, "understand": 2, "apply": 2}),
            (7, {"remember": 1, "understand": 2, "apply": 3, "analyze": 1}),
            (10, {"remember": 2, "understand": 3, "apply": 3, "analyze": 2}),
        ],
    )
    def test_small_quizzes_use_the_hand_written_table(
        self, count: int, expected: dict[str, int]
    ) -> None:
        """The low counts are enumerated rather than computed.

        Pinned exactly because a teacher asking for three questions gets this
        precise mix, and a change to it changes what every short quiz probes.
        """
        assert _default_bloom_distribution(count) == expected
