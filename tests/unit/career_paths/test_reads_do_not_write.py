"""Progress reads are reads; the latch rides on the write that causes it.

Two GET endpoints used to write. ``GET /me/career-enrollments`` and
``GET /me/career-enrollments/{id}/progress`` each latched newly complete
stages, flipped finished career enrollments to ``completed``, and committed.

The flip is not a small side effect: it calls ``complete_program_attempts``,
so a GET could complete a Learning Program enrolment -- graduate a student
from a degree programme. Anything that issues a GET could fire it: a client
retry, a prefetch, a link crawler. It is also what closed the enrolment in the
case where a programme's default path was one the student had already
finished, after which ``select_path`` refused every further path with
``paths_can_only_be_added_to_an_open_program``.

The fix is not new semantics. This codebase already states the pattern, in
``enrollments/services/completion.py``: fire synchronously from every write
point that can change a unit's state, and keep a nightly sweeper for drift.
Course completion already worked that way; the stage latch did not. It does
now, so the reads have nothing left to commit.

The latch stays PROMOTION-only. ``student_stage_progress`` is append-only
precisely so that un-marking a lesson cannot un-complete a student, and the
demotion branch must never reach it.

Everything below the service is mocked.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from abridgeai.features.career_paths.services import enrollment as service


@pytest.fixture
def world(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """One student, one active enrolment, every collaborator observable."""
    student_id = uuid4()
    career_path_id = uuid4()
    enrollment = SimpleNamespace(id=uuid4(), version_id=uuid4())
    latch = AsyncMock(return_value=1)
    evals: list[Any] = []

    monkeypatch.setattr(service.stage_service, "latch_completed_stages", latch)
    monkeypatch.setattr(service.stage_service, "evaluate_stages", AsyncMock(return_value=evals))
    monkeypatch.setattr(
        service.student_queries,
        "get_my_career_enrollment",
        AsyncMock(return_value=enrollment),
    )
    monkeypatch.setattr(
        service.student_queries, "get_path_course_progress", AsyncMock(return_value=[])
    )
    # The read reaches two more collaborators after the latch decision. They
    # are irrelevant to what these tests assert, but leaving them live would
    # put a DB call in a unit test rather than failing it.
    monkeypatch.setattr(
        service.authoring_queries,
        "get_career_path_for_authoring",
        # ``max_concurrent`` is the only attribute the read takes off this
        # row; a stand-in without it fails inside the service rather than
        # at the assertion, which reads as a code defect.
        AsyncMock(return_value=SimpleNamespace(max_concurrent=None)),
    )
    monkeypatch.setattr(
        service.enrollments_api,
        "count_active_enrollments_in_courses",
        AsyncMock(return_value=0),
    )
    return {
        "db": SimpleNamespace(),
        "student_id": student_id,
        "career_path_id": career_path_id,
        "enrollment": enrollment,
        "latch": latch,
        "monkeypatch": monkeypatch,
    }


class TestTheProgressReadIsARead:
    async def test_reading_progress_writes_no_latch(self, world: dict[str, Any]) -> None:
        """The default. A GET must not be able to change a student's record,
        however many times a client happens to issue it."""
        await service.get_my_path_progress(
            world["db"],
            career_path_id=world["career_path_id"],
            student_id=world["student_id"],
        )

        world["latch"].assert_not_awaited()

    async def test_a_caller_that_already_writes_can_ask_for_the_latch(
        self, world: dict[str, Any]
    ) -> None:
        """``latch=True`` is for callers on a write path -- the completion
        writer and the nightly readiness snapshot -- not for routers."""
        await service.get_my_path_progress(
            world["db"],
            career_path_id=world["career_path_id"],
            student_id=world["student_id"],
            latch=True,
        )

        world["latch"].assert_awaited_once()
        assert world["latch"].await_args.kwargs["enrollment_id"] == world["enrollment"].id

    async def test_a_preview_without_an_enrolment_never_latches(
        self, world: dict[str, Any]
    ) -> None:
        """There is no enrolment to latch against; asking for it changes
        nothing rather than raising."""
        world["monkeypatch"].setattr(
            service.student_queries, "get_my_career_enrollment", AsyncMock(return_value=None)
        )
        world["monkeypatch"].setattr(
            service.authoring_queries,
            "get_published_version",
            AsyncMock(return_value=SimpleNamespace(id=uuid4())),
        )

        await service.get_my_path_progress(
            world["db"],
            career_path_id=world["career_path_id"],
            student_id=world["student_id"],
            latch=True,
        )

        world["latch"].assert_not_awaited()


class TestSyncingAfterACourseCompletes:
    """The write path that replaced the lazy read."""

    @pytest.fixture
    def synced(self, world: dict[str, Any]) -> dict[str, Any]:
        progress = AsyncMock(return_value="progress")
        flip = AsyncMock(return_value=True)
        world["monkeypatch"].setattr(service, "get_my_path_progress", progress)
        world["monkeypatch"].setattr(service, "sync_enrollment_completion", flip)
        world.update({"progress": progress, "flip": flip})
        return world

    def _rows(self, world: dict[str, Any], *statuses: str) -> None:
        world["monkeypatch"].setattr(
            service.student_queries,
            "list_my_career_enrollments",
            AsyncMock(
                return_value=[
                    {"career_path_id": uuid4(), "status": status} for status in statuses
                ]
            ),
        )

    async def test_each_active_pathway_is_evaluated_with_the_latch_on(
        self, synced: dict[str, Any]
    ) -> None:
        self._rows(synced, "active", "active")

        flipped = await service.sync_paths_after_course_completion(
            synced["db"], student_id=synced["student_id"]
        )

        assert synced["progress"].await_count == 2
        assert all(
            call.kwargs["latch"] is True for call in synced["progress"].await_args_list
        )
        assert flipped == 2

    async def test_a_finished_pathway_is_skipped(self, synced: dict[str, Any]) -> None:
        """Its stages are already latched and there is no flip left to make,
        so re-evaluating it is work with no possible result."""
        self._rows(synced, "completed")

        assert (
            await service.sync_paths_after_course_completion(
                synced["db"], student_id=synced["student_id"]
            )
            == 0
        )
        synced["progress"].assert_not_awaited()

    async def test_only_the_active_ones_are_touched_in_a_mixed_set(
        self, synced: dict[str, Any]
    ) -> None:
        self._rows(synced, "completed", "active", "withdrawn")

        await service.sync_paths_after_course_completion(
            synced["db"], student_id=synced["student_id"]
        )

        assert synced["progress"].await_count == 1

    async def test_a_student_with_no_pathways_is_a_no_op(
        self, synced: dict[str, Any]
    ) -> None:
        """Most course completions belong to no career path at all, so this is
        the common case and must cost nothing."""
        self._rows(synced)

        assert (
            await service.sync_paths_after_course_completion(
                synced["db"], student_id=synced["student_id"]
            )
            == 0
        )
        synced["flip"].assert_not_awaited()

    async def test_the_count_reports_only_pathways_that_actually_flipped(
        self, synced: dict[str, Any]
    ) -> None:
        """The caller uses it to decide whether anything changed; counting
        evaluations instead would report a change on every course completion."""
        self._rows(synced, "active", "active")
        synced["flip"].side_effect = [True, False]

        assert (
            await service.sync_paths_after_course_completion(
                synced["db"], student_id=synced["student_id"]
            )
            == 1
        )
