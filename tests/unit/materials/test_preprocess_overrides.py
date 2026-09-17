"""The teacher's window into the noise filter, and the lever to overturn it.

Between extraction and chunking a cascade removes headers, footers, page
numbers and other boilerplate, recording every decision in a quarantine
table. This service is how a teacher sees those decisions and reverses the
wrong ones.

One line in it is worth the rest of the file. ``apply_teacher_action``
re-checks that the quarantine row belongs to the material in the URL,
because the router's permission dependency guards the *material* path
parameter and nothing else. A bare quarantine id would otherwise let a
caller restore or confirm another course's rows through their own
material's URL -- a write across a tenant boundary, reachable by anyone who
owns any material at all.

The rest is shaping: a report the teacher can read, and a mode switch that
takes effect on the next reprocess rather than rewriting embedded chunks in
place.

The query layer is mocked; its SQL is covered by the integration suite.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest

from abridgeai.core.exceptions import NotFoundError
from abridgeai.features.materials.services.authoring import _preprocess


def _unit(**over: Any) -> dict[str, Any]:
    """One quarantine row as the query layer returns it."""
    base: dict[str, Any] = {
        "id": uuid4(),
        "unit_kind": "line",
        "page_number": 3,
        "ordinal": 7,
        "content": "Faculty of Computer Science and Engineering",
        "occurrences": 42,
        "rule_name": "repeated_header",
        "reason_code": "repeated_across_pages",
        "action": "drop",
        "rule_score": 0.94,
        "detector_stage": "rule",
        "teacher_action": None,
        "teacher_action_at": None,
        "created_at": datetime(2026, 9, 1, tzinfo=UTC),
    }
    base.update(over)
    return base


class TestTheReport:
    async def test_a_material_with_no_current_version_is_not_found(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Nothing has been uploaded, so there is nothing the filter can
        have done to it."""
        monkeypatch.setattr(
            _preprocess, "get_current_version_and_mode", AsyncMock(return_value=None)
        )

        with pytest.raises(NotFoundError, match="no current version"):
            await _preprocess.get_preprocess_report(object(), uuid4())

    async def test_the_report_carries_the_removed_text_itself(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A teacher cannot sensibly override what they cannot see.

        The counts alone would say a rule fired 42 times without saying on
        what, which is not a basis for deciding whether the rule was right.
        """
        version_id = uuid4()
        monkeypatch.setattr(
            _preprocess,
            "get_current_version_and_mode",
            AsyncMock(return_value=(version_id, "full", {})),
        )
        monkeypatch.setattr(
            _preprocess, "list_quarantine_for_version", AsyncMock(return_value=[_unit()])
        )

        report = await _preprocess.get_preprocess_report(object(), uuid4())

        assert report.material_version_id == version_id
        assert report.preprocess_mode == "full"
        assert len(report.units) == 1
        assert report.units[0].content == "Faculty of Computer Science and Engineering"
        assert report.units[0].occurrences == 42
        assert report.units[0].page_number == 3

    async def test_the_summary_is_read_from_the_versions_metadata(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        summary = {"lines_removed": 118, "pages_scanned": 24}
        monkeypatch.setattr(
            _preprocess,
            "get_current_version_and_mode",
            AsyncMock(return_value=(uuid4(), "full", {"preprocess": summary})),
        )
        monkeypatch.setattr(
            _preprocess, "list_quarantine_for_version", AsyncMock(return_value=[])
        )

        report = await _preprocess.get_preprocess_report(object(), uuid4())

        assert report.summary == summary

    @pytest.mark.parametrize("stored", ["a string", 42, ["a", "list"], None])
    async def test_a_summary_of_the_wrong_shape_reads_as_absent(
        self, monkeypatch: pytest.MonkeyPatch, stored: Any
    ) -> None:
        """``extracted_metadata`` is a JSONB blob written by the pipeline.

        A shape the response model cannot hold would fail the whole report,
        and the report is the only way to reach the override -- so a
        malformed summary costs the summary, not the page.
        """
        monkeypatch.setattr(
            _preprocess,
            "get_current_version_and_mode",
            AsyncMock(return_value=(uuid4(), "full", {"preprocess": stored})),
        )
        monkeypatch.setattr(
            _preprocess, "list_quarantine_for_version", AsyncMock(return_value=[])
        )

        report = await _preprocess.get_preprocess_report(object(), uuid4())

        assert report.summary is None

    async def test_a_version_the_filter_never_touched_reports_nothing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Mode ``off``, or a clean document: an empty list, not an error."""
        monkeypatch.setattr(
            _preprocess,
            "get_current_version_and_mode",
            AsyncMock(return_value=(uuid4(), "off", {})),
        )
        monkeypatch.setattr(
            _preprocess, "list_quarantine_for_version", AsyncMock(return_value=[])
        )

        report = await _preprocess.get_preprocess_report(object(), uuid4())

        assert report.units == []
        assert report.preprocess_mode == "off"


class TestTheOwnershipCheckOnAnOverride:
    """The guard the module docstring singles out.

    The router's dependency authorises the material in the path. The
    quarantine id is just a number in the URL beside it, so this is the only
    thing standing between a caller and another course's rows.
    """

    @pytest.fixture
    def world(self, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
        stamp = AsyncMock(return_value=True)
        monkeypatch.setattr(_preprocess, "set_teacher_action", stamp)
        return {"stamp": stamp, "material_id": uuid4(), "user_id": uuid4()}

    async def _apply(
        self, world: dict[str, Any], quarantine_id: UUID, verdict: str = "restore"
    ):
        return await _preprocess.apply_teacher_action(
            object(),
            world["material_id"],
            quarantine_id,
            action=verdict,
            user_id=world["user_id"],
        )

    async def test_a_row_from_another_material_is_refused(
        self, monkeypatch: pytest.MonkeyPatch, world: dict[str, Any]
    ) -> None:
        """The row exists and the caller may touch their own material, so
        every other check in the chain passes. Only this one does not.
        """
        monkeypatch.setattr(
            _preprocess,
            "get_quarantine_row",
            AsyncMock(return_value={"material_id": uuid4()}),
        )

        with pytest.raises(NotFoundError, match="not found for this material"):
            await self._apply(world, uuid4())

        world["stamp"].assert_not_awaited()

    async def test_an_unknown_row_is_refused_the_same_way(
        self, monkeypatch: pytest.MonkeyPatch, world: dict[str, Any]
    ) -> None:
        """Identical message and error for "does not exist" and "is not
        yours": distinguishing them would let a caller enumerate which
        quarantine ids are real.
        """
        monkeypatch.setattr(_preprocess, "get_quarantine_row", AsyncMock(return_value=None))

        with pytest.raises(NotFoundError, match="not found for this material"):
            await self._apply(world, uuid4())

    async def test_a_row_on_the_callers_own_material_is_stamped(
        self, monkeypatch: pytest.MonkeyPatch, world: dict[str, Any]
    ) -> None:
        monkeypatch.setattr(
            _preprocess,
            "get_quarantine_row",
            AsyncMock(return_value={"material_id": world["material_id"]}),
        )
        quarantine_id = uuid4()

        assert await self._apply(world, quarantine_id) is True

        assert world["stamp"].await_args.args[1] == quarantine_id
        assert world["stamp"].await_args.kwargs["action"] == "restore"
        assert world["stamp"].await_args.kwargs["user_id"] == world["user_id"]

    @pytest.mark.parametrize("verdict", ["restore", "confirm"])
    async def test_both_verdicts_are_recorded_against_the_teacher(
        self, monkeypatch: pytest.MonkeyPatch, world: dict[str, Any], verdict: str
    ) -> None:
        """``confirm`` matters as much as ``restore``: it is what feeds the
        precision audit, so a rule nobody ever confirms looks the same as
        one nobody has reviewed.
        """
        monkeypatch.setattr(
            _preprocess,
            "get_quarantine_row",
            AsyncMock(return_value={"material_id": world["material_id"]}),
        )

        await self._apply(world, uuid4(), verdict)

        assert world["stamp"].await_args.kwargs["action"] == verdict

    async def test_a_row_that_vanished_between_read_and_write_reports_false(
        self, monkeypatch: pytest.MonkeyPatch, world: dict[str, Any]
    ) -> None:
        """The stamp is a separate statement, so the row can go in between.

        ``False`` rather than an exception: the caller's intent was to mark
        a row that no longer needs marking.
        """
        monkeypatch.setattr(
            _preprocess,
            "get_quarantine_row",
            AsyncMock(return_value={"material_id": world["material_id"]}),
        )
        world["stamp"].return_value = False

        assert await self._apply(world, uuid4()) is False


class TestTheModeSwitch:
    async def test_an_unknown_material_is_not_found(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The update touched no rows, which is how a missing material
        presents on a bare UPDATE."""
        monkeypatch.setattr(
            _preprocess, "update_preprocess_mode", AsyncMock(return_value=False)
        )

        with pytest.raises(NotFoundError, match="not found"):
            await _preprocess.set_preprocess_mode(object(), uuid4(), mode="off")

    @pytest.mark.parametrize("mode", ["full", "normalize_only", "off"])
    async def test_each_mode_is_accepted_and_echoed(
        self, monkeypatch: pytest.MonkeyPatch, mode: str
    ) -> None:
        """The echo is what the client renders back into the control, so a
        mode that saved but reported something else would show the teacher
        a setting they did not choose.
        """
        update = AsyncMock(return_value=True)
        monkeypatch.setattr(_preprocess, "update_preprocess_mode", update)
        material_id = uuid4()

        assert await _preprocess.set_preprocess_mode(object(), material_id, mode=mode) == mode
        assert update.await_args.kwargs["mode"] == mode
        assert update.await_args.args[1] == material_id


class TestTheCourseWideAudit:
    async def test_per_reason_counts_are_projected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A reason code with many restores is a rule eating real content.

        That comparison only works if restores and confirms travel
        together -- a restore count alone cannot say whether ten is a lot.
        """
        monkeypatch.setattr(
            _preprocess,
            "course_filter_summary",
            AsyncMock(
                return_value=[
                    {
                        "reason_code": "repeated_across_pages",
                        "unit_count": 120,
                        "occurrence_count": 480,
                        "restored": 2,
                        "confirmed": 90,
                    }
                ]
            ),
        )

        rows = await _preprocess.get_course_filter_summary(object(), uuid4())

        assert len(rows) == 1
        assert rows[0].reason_code == "repeated_across_pages"
        assert rows[0].unit_count == 120
        assert rows[0].occurrence_count == 480
        assert rows[0].restored == 2

    async def test_a_course_the_filter_never_ran_on_is_empty(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            _preprocess, "course_filter_summary", AsyncMock(return_value=[])
        )

        assert await _preprocess.get_course_filter_summary(object(), uuid4()) == []
