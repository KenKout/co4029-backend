"""Module prerequisites, and the gate that had been ignoring them.

The product offers exactly one way to say "finish module A before module B":
``PUT /teacher/modules/{id}/prerequisites``, which writes
``module_prerequisites``. Until this wiring, nothing read that table. The
unlock gate consulted only ``lesson_prerequisites`` — a table with no write
route at all, whose sole writer is the course-clone copier — so for any course
authored through the app the prerequisite half of the gate was vacuously
satisfied, and the only thing locking anything was a lesson's own same-module
EF threshold. A teacher could set the control, watch it save, and change
nothing.

These tests pin the composition: both edge kinds resolve to the same rule (a
prerequisite is met when that lesson is itself eligible, which folds in its own
EF gate), the two are unioned rather than one shadowing the other, and the walk
terminates when the graph loops.

No database: the two fetches and the per-lesson EF read are stubbed, because
what is under test is which lessons the gate decides to consult.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch
from uuid import UUID, uuid4

import pytest

from abridgeai.features.spaced_repetition.sm2 import lesson_unlock

STUDENT = uuid4()
# Lesson under test, in the gated module.
TARGET = uuid4()
# Lessons of the prerequisite module.
PREREQ_A = uuid4()
PREREQ_B = uuid4()
# An explicitly named lesson prerequisite, the older edge kind.
NAMED_PREREQ = uuid4()


class _Graph:
    """Stubs the per-lesson reads `_check_unlock_recursive` makes.

    `failing` is the set of lessons whose EF gate does not pass; everything
    else has cards and passes. Records which lessons were consulted so a test
    can assert the gate stopped early instead of walking the whole course.
    """

    def __init__(
        self,
        *,
        lesson_edges: dict[UUID, list[UUID]] | None = None,
        module_edges: dict[UUID, list[UUID]] | None = None,
        failing: set[UUID] | None = None,
    ) -> None:
        self.lesson_edges = lesson_edges or {}
        self.module_edges = module_edges or {}
        self.failing = failing or set()
        self.aggregated: list[UUID] = []

    async def fetch_lesson_prereqs(self, db: Any, *, lesson_id: UUID) -> list[UUID]:
        return self.lesson_edges.get(lesson_id, [])

    async def fetch_module_prereqs(self, db: Any, *, lesson_id: UUID) -> list[UUID]:
        return self.module_edges.get(lesson_id, [])

    async def fetch_config(self, db: Any, *, lesson_id: UUID) -> tuple[float, float, bool]:
        # ef_min 2.0, tau 1.0 (every card must pass), no interview.
        return (2.0, 1.0, False)

    async def aggregate(
        self, db: Any, *, student_id: UUID, lesson_id: UUID, ef_min: float, blocking_limit: int
    ) -> tuple[int, int, list[dict[str, Any]]]:
        self.aggregated.append(lesson_id)
        if lesson_id in self.failing:
            return 0, 1, [{"question_id": str(uuid4()), "quiz_id": str(uuid4())}]
        return 1, 1, []


def _patched(graph: _Graph) -> Any:
    return patch.multiple(
        lesson_unlock,
        fetch_prerequisite_lesson_ids=AsyncMock(side_effect=graph.fetch_lesson_prereqs),
        fetch_prerequisite_module_lesson_ids=AsyncMock(side_effect=graph.fetch_module_prereqs),
        fetch_lesson_unlock_config=AsyncMock(side_effect=graph.fetch_config),
        aggregate_lesson_card_ef=AsyncMock(side_effect=graph.aggregate),
    )


async def _check(graph: _Graph, lesson_id: UUID = TARGET) -> Any:
    with _patched(graph):
        return await lesson_unlock._check_unlock_recursive(
            AsyncMock(), student_id=STUDENT, lesson_id=lesson_id, visited=set()
        )


class TestModuleEdges:
    async def test_a_failing_lesson_in_the_prerequisite_module_locks_the_target(
        self,
    ) -> None:
        # The whole point of the control: module B stays shut until the
        # student has actually retained module A.
        graph = _Graph(module_edges={TARGET: [PREREQ_A, PREREQ_B]}, failing={PREREQ_B})

        status = await _check(graph)

        assert status.eligible is False
        assert status.prereq_lesson_ids_unlocked is False

    async def test_the_target_opens_once_the_prerequisite_module_is_retained(
        self,
    ) -> None:
        graph = _Graph(module_edges={TARGET: [PREREQ_A, PREREQ_B]})

        status = await _check(graph)

        assert status.eligible is True
        assert status.prereq_lesson_ids_unlocked is True

    async def test_a_module_with_no_prerequisites_is_unaffected(self) -> None:
        graph = _Graph()

        status = await _check(graph)

        assert status.eligible is True
        # Only the target's own cards were read — no phantom walk.
        assert graph.aggregated == [TARGET]


class TestBothEdgeKinds:
    async def test_lesson_edges_and_module_edges_are_both_honoured(self) -> None:
        # The module edge must not shadow the explicit lesson edge, nor vice
        # versa: they are a union, and either one can block.
        graph = _Graph(
            lesson_edges={TARGET: [NAMED_PREREQ]},
            module_edges={TARGET: [PREREQ_A]},
            failing={NAMED_PREREQ},
        )

        assert (await _check(graph)).eligible is False

        graph = _Graph(
            lesson_edges={TARGET: [NAMED_PREREQ]},
            module_edges={TARGET: [PREREQ_A]},
            failing={PREREQ_A},
        )

        assert (await _check(graph)).eligible is False

    async def test_a_lesson_reached_through_both_edges_is_checked_once(self) -> None:
        graph = _Graph(lesson_edges={TARGET: [PREREQ_A]}, module_edges={TARGET: [PREREQ_A]})

        await _check(graph)

        assert graph.aggregated.count(PREREQ_A) == 1


class TestCycles:
    async def test_a_module_cycle_terminates_rather_than_bricking_the_student(
        self,
    ) -> None:
        # Mutual module prerequisites are an authoring mistake, but the learner
        # must not pay for it with an infinite walk or a permanent lock — the
        # same "treat the cycle as satisfied" rule the lesson graph already has.
        graph = _Graph(
            module_edges={TARGET: [PREREQ_A], PREREQ_A: [TARGET]},
        )

        status = await _check(graph)

        assert status.eligible is True


@pytest.mark.parametrize("edge_kind", ["lesson_edges", "module_edges"])
async def test_a_deep_chain_still_blocks_from_the_far_end(edge_kind: str) -> None:
    # target -> A -> B, with B failing: the block has to propagate back up.
    graph = _Graph(
        **{edge_kind: {TARGET: [PREREQ_A], PREREQ_A: [PREREQ_B]}},
        failing={PREREQ_B},
    )

    assert (await _check(graph)).eligible is False
