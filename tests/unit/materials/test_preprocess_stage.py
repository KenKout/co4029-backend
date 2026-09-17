"""The wiring between the pure preprocessing cascade and the outside world.

``ai/preprocessing`` holds the algorithms and knows nothing about sessions
or files. This module supplies the two side-effecting collaborators and
makes one promise on top of them, stated in its own docstring: **on ANY
failure the original content is returned unchanged.** Preprocessing is a
quality improvement, never a correctness dependency, so it must not be able
to fail an ingest.

That promise is what most of this file is about. Each test names a way it
could quietly stop holding -- a gateway outage, a rule raising, the
teacher-restore lookup failing -- because none of them have a visible
failure mode. A teacher whose document silently lost half its pages sees a
successful upload.

The other half is ``_persist_quarantine``, the audit trail that makes a
drop reversible. Its ordinals are the upsert key, so a change in how they
are assigned would silently re-key every row on the next reprocess and
orphan the teacher actions attached to the old ones.

No database, no gateway, no PDF: ``fitz`` is injected into ``sys.modules``
and the query layer is mocked.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from abridgeai.ai.extraction import ExtractedContent
from abridgeai.ai.preprocessing import PreprocessReport
from abridgeai.ai.preprocessing.base import Action, Decision, ReasonCode
from abridgeai.features.materials.ingestion import preprocess as stage


# One more than the cap, so the break is exercised rather than the loop
# simply running out.
_CAP_PROBE = 520


def _settings(**over: Any) -> SimpleNamespace:
    base: dict[str, Any] = {
        "preprocess_enabled": True,
        "preprocess_dehyphenation": True,
        "preprocess_running_marks": True,
        "preprocess_page_roles": True,
        "preprocess_deck_detection": True,
        "preprocess_ocr_enabled": True,
        "preprocess_ocr_dpi": 150,
        "preprocess_ocr_max_pages": 30,
        "preprocess_llm_adjudication": True,
        "preprocess_llm_min_confidence": 0.8,
    }
    base.update(over)
    return SimpleNamespace(**base)


def _extracted(source_type: str = "pdf") -> ExtractedContent:
    return ExtractedContent(
        text="[Page 1]\nphotosynthesis\n", metadata={}, source_type=source_type
    )


def _decision(action: Action, **over: Any) -> Decision:
    kwargs: dict[str, Any] = {
        "action": action,
        "reason": ReasonCode.RUNNING_HEADER,
        "rule_name": "repeated_header",
        "page_number": 3,
        "content": "Faculty of Computer Science and Engineering",
        "score": 0.94,
        "stage": "deterministic",
        "occurrences": 42,
    }
    kwargs.update(over)
    return Decision(**kwargs)


class TestTheEscapeHatchForADocumentTheFiltersMisread:
    """``preprocess_mode`` on the material row.

    A teacher whose lecture notes were mangled needs a way out that does not
    cost them the repairs they do want, which is why there are three modes
    and not a checkbox.
    """

    def test_off_disables_the_cascade_outright(self) -> None:
        config = stage.config_from_settings(_settings(), mode="off")

        assert config.enabled is False

    def test_off_ignores_the_global_switch_entirely(self) -> None:
        """A per-material ``off`` cannot be re-enabled by env, and it does
        not need env's permission to take effect either."""
        config = stage.config_from_settings(_settings(preprocess_enabled=True), mode="off")

        assert config.enabled is False

    def test_normalize_only_keeps_the_repairs_and_drops_the_judgement(self) -> None:
        """The whole point of the middle setting.

        Unicode folds and de-hyphenation are never destructive -- they
        repair ligatures and line-break hyphens. Everything that can remove
        or de-prioritize a page is off, so the teacher keeps every page and
        still gets the repairs.
        """
        config = stage.config_from_settings(_settings(), mode="normalize_only")

        assert config.normalize is True
        assert config.dehyphenation is True
        assert config.blankness is False
        assert config.running_marks is False
        assert config.page_roles is False
        assert config.deck_detection is False
        assert config.ocr_enabled is False
        assert config.llm_adjudication is False

    def test_normalize_only_still_honours_the_global_switch(self) -> None:
        """Unlike ``off``, this one is a narrowing of the cascade rather
        than a replacement for it, so a deployment-wide disable still
        wins."""
        config = stage.config_from_settings(
            _settings(preprocess_enabled=False), mode="normalize_only"
        )

        assert config.enabled is False

    def test_normalize_only_respects_the_dehyphenation_setting(self) -> None:
        config = stage.config_from_settings(
            _settings(preprocess_dehyphenation=False), mode="normalize_only"
        )

        assert config.dehyphenation is False

    def test_full_turns_on_the_whole_cascade(self) -> None:
        config = stage.config_from_settings(_settings(), mode="full")

        assert config.enabled is True
        assert config.running_marks is True
        assert config.page_roles is True
        assert config.deck_detection is True
        assert config.llm_adjudication is True

    def test_an_unrecognised_mode_falls_through_to_full(self) -> None:
        """The three values are enforced by a CHECK constraint on the
        column, not by this function -- ``mode`` is a plain ``str`` here.

        So the fallback direction is the thing to pin, because the two ways
        of getting it wrong are not symmetrical: falling through to ``off``
        would silently disable preprocessing for every material a future
        mode name touched, and nothing would report it.
        """
        config = stage.config_from_settings(_settings(), mode="something_new")

        assert config.enabled is True
        assert config.running_marks is True


class TestTheAdminTunableKnobs:
    """``runtime`` is the resolved registry chain (org -> global -> env).

    It outranks the process-wide ``Settings`` precisely so an admin can turn
    OCR off during an incident without a redeploy. If env won instead, that
    control would appear to work and change nothing.
    """

    def test_runtime_outranks_the_environment_for_ocr(self) -> None:
        config = stage.config_from_settings(
            _settings(preprocess_ocr_enabled=True),
            mode="full",
            runtime={"preprocess.ocr_enabled": False},
        )

        assert config.ocr_enabled is False

    def test_runtime_outranks_the_environment_for_the_page_budget(self) -> None:
        config = stage.config_from_settings(
            _settings(preprocess_ocr_max_pages=30),
            mode="full",
            runtime={"preprocess.ocr_advisory_max_pages": 5},
        )

        assert config.ocr_max_pages == 5

    def test_the_environment_is_used_when_the_registry_says_nothing(self) -> None:
        config = stage.config_from_settings(
            _settings(preprocess_ocr_enabled=False, preprocess_ocr_max_pages=7),
            mode="full",
            runtime={},
        )

        assert config.ocr_enabled is False
        assert config.ocr_max_pages == 7

    def test_no_runtime_at_all_is_the_same_as_an_empty_one(self) -> None:
        assert stage.config_from_settings(_settings(), mode="full") == (
            stage.config_from_settings(_settings(), mode="full", runtime={})
        )

    def test_the_figure_thresholds_have_no_environment_entry(self) -> None:
        """These two exist only in the registry, so their fallback is the
        literal default here rather than a ``Settings`` field -- reading
        them off ``settings`` would raise on a real config object.
        """
        config = stage.config_from_settings(_settings(), mode="full", runtime={})

        assert config.figure_page_max_words == 20
        assert config.figure_area_min == 0.12

    def test_the_figure_thresholds_are_tunable(self) -> None:
        config = stage.config_from_settings(
            _settings(),
            mode="full",
            runtime={"preprocess.figure_page_max_words": 45, "preprocess.figure_area_min": 0.4},
        )

        assert config.figure_page_max_words == 45
        assert config.figure_area_min == 0.4

    def test_registry_values_are_coerced_to_their_declared_types(self) -> None:
        """Registry values arrive out of a JSONB column, so a number that
        was stored as a string would otherwise reach ``ocr_max_pages`` as
        text and fail a comparison deep inside the cascade.
        """
        config = stage.config_from_settings(
            _settings(),
            mode="full",
            runtime={
                "preprocess.ocr_advisory_max_pages": "12",
                "preprocess.figure_page_max_words": "30",
                "preprocess.figure_area_min": "0.25",
            },
        )

        assert config.ocr_max_pages == 12
        assert config.figure_page_max_words == 30
        assert config.figure_area_min == 0.25


class _FakeOcr:
    """Stands in for ``_PdfPageOcr`` so construction and close are visible."""

    instances: list[_FakeOcr] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.closed = 0
        _FakeOcr.instances.append(self)

    def close(self) -> None:
        self.closed += 1


@pytest.fixture
def world(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """A stage whose every collaborator is inert and observable."""
    _FakeOcr.instances = []
    report = PreprocessReport()
    cleaned = ExtractedContent(text="cleaned", metadata={}, source_type="pdf")
    run = AsyncMock(return_value=(cleaned, report))
    persist = AsyncMock()
    monkeypatch.setattr(stage, "_PdfPageOcr", _FakeOcr)
    monkeypatch.setattr(stage, "run_preprocessing", run)
    monkeypatch.setattr(stage, "_persist_quarantine", persist)
    monkeypatch.setattr(stage, "list_restored_pages", AsyncMock(return_value=set()))
    return {"run": run, "persist": persist, "cleaned": cleaned, "report": report}


class TestPreprocessingCannotFailAnIngest:
    """The promise in the module docstring, one failure mode per test.

    None of these surface to the teacher: the upload succeeds either way.
    The only difference is whether the material they can search is the one
    they uploaded.
    """

    async def test_a_disabled_cascade_returns_the_extraction_untouched(
        self, world: dict[str, Any]
    ) -> None:
        extracted = _extracted()

        content, report = await stage.run_preprocess_stage(
            extracted, db=object(), settings=_settings(), mode="off"
        )

        assert content is extracted
        assert report is None
        world["run"].assert_not_awaited()

    async def test_the_global_switch_returns_the_extraction_untouched(
        self, world: dict[str, Any]
    ) -> None:
        extracted = _extracted()

        content, report = await stage.run_preprocess_stage(
            extracted, db=object(), settings=_settings(preprocess_enabled=False)
        )

        assert content is extracted
        assert report is None
        world["run"].assert_not_awaited()

    async def test_a_raising_cascade_returns_the_raw_extraction(
        self, world: dict[str, Any]
    ) -> None:
        """A bug in any rule costs retrieval quality, not the upload."""
        world["run"].side_effect = RuntimeError("a rule divided by zero")
        extracted = _extracted()

        content, report = await stage.run_preprocess_stage(
            extracted, db=object(), settings=_settings()
        )

        assert content is extracted
        assert report is None

    async def test_a_raising_cascade_does_not_write_an_audit_trail(
        self, world: dict[str, Any]
    ) -> None:
        """There is no report to persist, and persisting a partial one
        would describe decisions that were never applied."""
        world["run"].side_effect = RuntimeError("boom")

        await stage.run_preprocess_stage(
            _extracted(),
            db=object(),
            settings=_settings(),
            material_version_id=uuid4(),
            course_id=uuid4(),
        )

        world["persist"].assert_not_awaited()

    async def test_a_failed_restore_lookup_does_not_stop_the_run(
        self, monkeypatch: pytest.MonkeyPatch, world: dict[str, Any]
    ) -> None:
        """Losing the teacher's restores costs those pages their protection.

        Refusing to run at all would cost the whole document its
        preprocessing, which is strictly worse -- so this one degrades
        rather than aborting.
        """
        monkeypatch.setattr(
            stage, "list_restored_pages", AsyncMock(side_effect=RuntimeError("db gone"))
        )

        content, report = await stage.run_preprocess_stage(
            _extracted(),
            db=object(),
            settings=_settings(),
            material_version_id=uuid4(),
        )

        assert content is world["cleaned"]
        assert world["run"].await_args.kwargs["protected_pages"] == set()

    async def test_the_pdf_document_is_closed_even_when_the_cascade_raises(
        self, world: dict[str, Any]
    ) -> None:
        """``close`` sits in a ``finally`` for this case.

        The renderer holds an open file handle per ingest; leaking one per
        failed document is a slow resource exhaustion in a long-lived
        worker rather than an error anyone would notice.
        """
        world["run"].side_effect = RuntimeError("boom")

        await stage.run_preprocess_stage(
            _extracted(),
            db=object(),
            settings=_settings(),
            source_path=Path("/tmp/doc.pdf"),
            llm_gateway=object(),
        )

        assert len(_FakeOcr.instances) == 1
        assert _FakeOcr.instances[0].closed == 1

    async def test_the_pdf_document_is_closed_on_the_happy_path_too(
        self, world: dict[str, Any]
    ) -> None:
        await stage.run_preprocess_stage(
            _extracted(),
            db=object(),
            settings=_settings(),
            source_path=Path("/tmp/doc.pdf"),
            llm_gateway=object(),
        )

        assert _FakeOcr.instances[0].closed == 1


class TestWhenTheExpensiveCollaboratorsAreBuilt:
    """Both cost money per page, so every guard here is a bill.

    Each is also optional: the cascade takes ``None`` and skips that tier.
    """

    async def test_ocr_needs_the_switch_the_gateway_the_file_and_a_pdf(
        self, world: dict[str, Any]
    ) -> None:
        await stage.run_preprocess_stage(
            _extracted("pdf"),
            db=object(),
            settings=_settings(),
            source_path=Path("/tmp/doc.pdf"),
            llm_gateway=object(),
        )

        assert world["run"].await_args.kwargs["ocr"] is _FakeOcr.instances[0]

    @pytest.mark.parametrize(
        ("label", "kwargs"),
        [
            ("no gateway", {"source_path": Path("/tmp/d.pdf"), "llm_gateway": None}),
            ("no file on disk", {"source_path": None, "llm_gateway": object()}),
        ],
    )
    async def test_ocr_is_skipped_when_a_prerequisite_is_missing(
        self, world: dict[str, Any], label: str, kwargs: dict[str, Any]
    ) -> None:
        await stage.run_preprocess_stage(
            _extracted("pdf"), db=object(), settings=_settings(), **kwargs
        )

        assert world["run"].await_args.kwargs["ocr"] is None, label

    async def test_ocr_is_skipped_for_a_source_that_is_not_a_pdf(
        self, world: dict[str, Any]
    ) -> None:
        """The renderer opens the file with a PDF library. A docx reaching
        it would raise inside the stage rather than being skipped.
        """
        await stage.run_preprocess_stage(
            _extracted("docx"),
            db=object(),
            settings=_settings(),
            source_path=Path("/tmp/doc.docx"),
            llm_gateway=object(),
        )

        assert world["run"].await_args.kwargs["ocr"] is None

    async def test_ocr_is_skipped_when_an_admin_turned_it_off(
        self, world: dict[str, Any]
    ) -> None:
        await stage.run_preprocess_stage(
            _extracted("pdf"),
            db=object(),
            settings=_settings(),
            source_path=Path("/tmp/doc.pdf"),
            llm_gateway=object(),
            runtime={"preprocess.ocr_enabled": False},
        )

        assert world["run"].await_args.kwargs["ocr"] is None

    async def test_the_render_resolution_is_tunable_without_a_redeploy(
        self, world: dict[str, Any]
    ) -> None:
        """DPI is the cost/legibility dial: it changes the image size sent
        to a vision model, so it is the first thing to turn down."""
        await stage.run_preprocess_stage(
            _extracted("pdf"),
            db=object(),
            settings=_settings(preprocess_ocr_dpi=150),
            source_path=Path("/tmp/doc.pdf"),
            llm_gateway=object(),
            runtime={"preprocess.ocr_dpi": 96},
        )

        assert _FakeOcr.instances[0].kwargs["dpi"] == 96

    async def test_the_adjudicator_needs_the_switch_and_a_gateway(
        self, world: dict[str, Any]
    ) -> None:
        await stage.run_preprocess_stage(
            _extracted(), db=object(), settings=_settings(), llm_gateway=object()
        )

        assert isinstance(world["run"].await_args.kwargs["adjudicator"], stage._LlmPageAdjudicator)

    @pytest.mark.parametrize(
        ("label", "settings_over", "gateway"),
        [
            ("switched off", {"preprocess_llm_adjudication": False}, object()),
            ("no gateway", {}, None),
        ],
    )
    async def test_the_adjudicator_is_skipped_without_its_prerequisites(
        self,
        world: dict[str, Any],
        label: str,
        settings_over: dict[str, Any],
        gateway: object | None,
    ) -> None:
        await stage.run_preprocess_stage(
            _extracted(),
            db=object(),
            settings=_settings(**settings_over),
            llm_gateway=gateway,
        )

        assert world["run"].await_args.kwargs["adjudicator"] is None, label


class TestTheTeacherRestores:
    async def test_restored_pages_are_handed_to_the_cascade(
        self, monkeypatch: pytest.MonkeyPatch, world: dict[str, Any]
    ) -> None:
        monkeypatch.setattr(stage, "list_restored_pages", AsyncMock(return_value={3, 9}))

        await stage.run_preprocess_stage(
            _extracted(), db=object(), settings=_settings(), material_version_id=uuid4()
        )

        assert world["run"].await_args.kwargs["protected_pages"] == {3, 9}

    async def test_a_first_run_has_no_version_to_look_restores_up_by(
        self, monkeypatch: pytest.MonkeyPatch, world: dict[str, Any]
    ) -> None:
        """The version row may not exist yet. Querying with ``None`` would
        be a wasted round trip at best."""
        lookup = AsyncMock(return_value=set())
        monkeypatch.setattr(stage, "list_restored_pages", lookup)

        await stage.run_preprocess_stage(_extracted(), db=object(), settings=_settings())

        lookup.assert_not_awaited()
        assert world["run"].await_args.kwargs["protected_pages"] == set()


class TestWritingTheAuditTrail:
    async def test_the_trail_is_written_when_both_owners_are_known(
        self, world: dict[str, Any]
    ) -> None:
        version_id, course_id = uuid4(), uuid4()

        await stage.run_preprocess_stage(
            _extracted(),
            db=object(),
            settings=_settings(),
            material_version_id=version_id,
            course_id=course_id,
        )

        assert world["persist"].await_args.kwargs["material_version_id"] == version_id
        assert world["persist"].await_args.kwargs["course_id"] == course_id

    @pytest.mark.parametrize(
        ("label", "ids"),
        [
            ("no course", {"material_version_id": uuid4(), "course_id": None}),
            ("no version", {"material_version_id": None, "course_id": uuid4()}),
            ("neither", {"material_version_id": None, "course_id": None}),
        ],
    )
    async def test_an_unowned_run_writes_no_trail(
        self, world: dict[str, Any], label: str, ids: dict[str, Any]
    ) -> None:
        """``course_id`` is a NOT NULL column on the quarantine table, so
        this is a guard against an integrity error as much as a policy."""
        await stage.run_preprocess_stage(
            _extracted(), db=object(), settings=_settings(), **ids
        )

        assert not world["persist"].await_count, label

    async def test_the_cleaned_content_is_what_comes_back(
        self, world: dict[str, Any]
    ) -> None:
        content, report = await stage.run_preprocess_stage(
            _extracted(), db=object(), settings=_settings()
        )

        assert content is world["cleaned"]
        assert report is world["report"]


class TestWhichDecisionsEarnAnAuditRow:
    """Only decisions a teacher might want to overturn.

    ``route_ocr`` re-reads a page rather than removing anything, so a row
    for it would be noise in a list whose whole purpose is "here is what was
    taken away from you".
    """

    @pytest.fixture
    def db(self) -> SimpleNamespace:
        return SimpleNamespace(execute=AsyncMock())

    async def _persist(self, db: SimpleNamespace, *decisions: Decision) -> list[dict[str, Any]]:
        report = PreprocessReport(decisions=list(decisions))
        await stage._persist_quarantine(
            db, report, material_version_id=uuid4(), course_id=uuid4()
        )
        if not db.execute.await_args:
            return []
        return db.execute.await_args.args[1]

    @pytest.mark.parametrize(
        "action",
        [Action.DROP_PAGE, Action.STRIP_LINES, Action.EXCLUDE_RETRIEVAL, Action.TAG_ROLE],
    )
    async def test_a_removal_or_a_demotion_is_recorded(
        self, db: SimpleNamespace, action: Action
    ) -> None:
        rows = await self._persist(db, _decision(action))

        assert len(rows) == 1
        assert rows[0]["action"] == action.value

    async def test_tagging_a_role_is_recorded_though_no_text_was_removed(
        self, db: SimpleNamespace
    ) -> None:
        """De-prioritizing a page to front_matter is retrieval-visible: the
        page still exists but stops being retrieved, which looks to the
        teacher exactly like it was deleted.
        """
        rows = await self._persist(db, _decision(Action.TAG_ROLE))

        assert rows[0]["action"] == "tag_role"
        assert rows[0]["unit_kind"] == "page"

    @pytest.mark.parametrize("action", [Action.ROUTE_OCR, Action.LINK_CANONICAL])
    async def test_a_decision_that_removed_nothing_earns_no_row(
        self, db: SimpleNamespace, action: Action
    ) -> None:
        await self._persist(db, _decision(action))

        db.execute.assert_not_awaited()

    async def test_a_run_that_changed_nothing_touches_the_database_at_all(
        self, db: SimpleNamespace
    ) -> None:
        """A clean document is the common case; an empty INSERT would be a
        round trip per ingest for nothing."""
        await self._persist(db)

        db.execute.assert_not_awaited()

    async def test_only_line_removals_are_line_units(self, db: SimpleNamespace) -> None:
        """``unit_kind`` is part of the upsert key, so a page and a line at
        the same ordinal are separate rows rather than one overwriting the
        other."""
        rows = await self._persist(
            db, _decision(Action.STRIP_LINES), _decision(Action.DROP_PAGE)
        )

        assert rows[0]["unit_kind"] == "line"
        assert rows[1]["unit_kind"] == "page"


class TestTheOrdinalsThatKeyTheUpsert:
    @pytest.fixture
    def db(self) -> SimpleNamespace:
        return SimpleNamespace(execute=AsyncMock())

    async def _rows(self, db: SimpleNamespace, *decisions: Decision) -> list[dict[str, Any]]:
        await stage._persist_quarantine(
            db,
            PreprocessReport(decisions=list(decisions)),
            material_version_id=uuid4(),
            course_id=uuid4(),
        )
        return db.execute.await_args.args[1]

    async def test_an_ordinal_is_a_position_in_the_full_decision_log(
        self, db: SimpleNamespace
    ) -> None:
        """Not a position among the recorded rows.

        Skipped decisions still consume their number, so the ordinals are
        gappy on purpose: a decision keeps its identity across a reprocess
        even when an earlier unrecorded rule stops firing.
        """
        rows = await self._rows(
            db,
            _decision(Action.ROUTE_OCR),
            _decision(Action.DROP_PAGE),
            _decision(Action.LINK_CANONICAL),
            _decision(Action.STRIP_LINES),
        )

        assert [r["ordinal"] for r in rows] == [1, 3]

    async def test_the_owning_version_is_stamped_on_every_row(
        self, db: SimpleNamespace
    ) -> None:
        version_id, course_id = uuid4(), uuid4()
        await stage._persist_quarantine(
            db,
            PreprocessReport(decisions=[_decision(Action.DROP_PAGE) for _ in range(3)]),
            material_version_id=version_id,
            course_id=course_id,
        )

        rows = db.execute.await_args.args[1]
        assert {r["material_version_id"] for r in rows} == {version_id}
        assert {r["course_id"] for r in rows} == {course_id}


class TestTheBackstopsOnOneRow:
    @pytest.fixture
    def db(self) -> SimpleNamespace:
        return SimpleNamespace(execute=AsyncMock())

    async def _rows(self, db: SimpleNamespace, *decisions: Decision) -> list[dict[str, Any]]:
        await stage._persist_quarantine(
            db,
            PreprocessReport(decisions=list(decisions)),
            material_version_id=uuid4(),
            course_id=uuid4(),
        )
        return db.execute.await_args.args[1]

    async def test_a_pathological_document_cannot_write_unbounded_rows(
        self, db: SimpleNamespace
    ) -> None:
        """A 500-page scan where every line trips a rule would otherwise
        put tens of thousands of rows behind one upload."""
        rows = await self._rows(
            db, *[_decision(Action.STRIP_LINES) for _ in range(_CAP_PROBE)]
        )

        assert len(rows) == stage._MAX_QUARANTINE_ROWS

    async def test_the_cap_keeps_the_earliest_decisions(
        self, db: SimpleNamespace
    ) -> None:
        """Truncating the tail rather than sampling keeps the report
        readable: the teacher sees a contiguous prefix of the document."""
        rows = await self._rows(
            db,
            *[
                _decision(Action.STRIP_LINES, page_number=n)
                for n in range(_CAP_PROBE)
            ],
        )

        assert rows[0]["page_number"] == 0
        assert rows[-1]["page_number"] == stage._MAX_QUARANTINE_ROWS - 1

    async def test_removed_text_is_truncated_to_the_column_width(
        self, db: SimpleNamespace
    ) -> None:
        """``content`` is what makes the drop reviewable, so it is stored
        rather than summarized -- but a single page of dense text can
        exceed the column."""
        rows = await self._rows(
            db, _decision(Action.DROP_PAGE, content="x" * 9000)
        )

        assert len(rows[0]["content"]) == 4000

    async def test_a_long_rule_name_is_truncated(self, db: SimpleNamespace) -> None:
        rows = await self._rows(
            db, _decision(Action.DROP_PAGE, rule_name="r" * 200)
        )

        assert len(rows[0]["rule_name"]) == 64

    async def test_a_long_stage_name_is_truncated(self, db: SimpleNamespace) -> None:
        rows = await self._rows(db, _decision(Action.DROP_PAGE, stage="s" * 40))

        assert len(rows[0]["detector_stage"]) == 16

    async def test_the_evidence_a_teacher_judges_by_is_carried_whole(
        self, db: SimpleNamespace
    ) -> None:
        """Occurrence count and score are the difference between "a rule
        fired once at 0.5" and "a rule fired on 42 pages at 0.94"."""
        rows = await self._rows(
            db,
            _decision(
                Action.STRIP_LINES,
                occurrences=42,
                score=0.94,
                page_number=3,
                content="Faculty of Computer Science and Engineering",
            ),
        )

        assert rows[0]["occurrences"] == 42
        assert rows[0]["rule_score"] == 0.94
        assert rows[0]["page_number"] == 3
        assert rows[0]["content"] == "Faculty of Computer Science and Engineering"
        assert rows[0]["reason_code"] == "running_header"

    async def test_a_failed_write_does_not_reach_the_pipeline(
        self, db: SimpleNamespace
    ) -> None:
        """The audit trail is worth less than the ingest.

        Raising here would fail an upload whose content was preprocessed
        perfectly well.
        """
        db.execute.side_effect = RuntimeError("deadlock detected")

        await stage._persist_quarantine(
            db,
            PreprocessReport(decisions=[_decision(Action.DROP_PAGE)]),
            material_version_id=uuid4(),
            course_id=uuid4(),
        )


class _Pixmap:
    def __init__(self, dpi: int) -> None:
        self.dpi = dpi

    def tobytes(self, fmt: str) -> bytes:
        assert fmt == "png"
        return b"\x89PNG-bytes"


class _Page:
    def __init__(self, recorder: list[int]) -> None:
        self._recorder = recorder

    def get_pixmap(self, dpi: int) -> _Pixmap:
        self._recorder.append(dpi)
        return _Pixmap(dpi)


class _Doc:
    def __init__(self, page_count: int, recorder: list[int]) -> None:
        self.page_count = page_count
        self.closed = 0
        self._recorder = recorder
        self.loaded: list[int] = []

    def load_page(self, index: int) -> _Page:
        self.loaded.append(index)
        return _Page(self._recorder)

    def close(self) -> None:
        self.closed += 1


@pytest.fixture
def fitz(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """A fake PDF library, so none of this needs a real document."""
    recorder: list[int] = []
    doc = _Doc(page_count=3, recorder=recorder)
    opened: list[str] = []

    def _open(path: str) -> _Doc:
        opened.append(path)
        return doc

    monkeypatch.setitem(sys.modules, "fitz", SimpleNamespace(open=_open))
    return {"doc": doc, "opened": opened, "dpis": recorder}


def _gateway(content: Any) -> SimpleNamespace:
    return SimpleNamespace(
        generate_json=AsyncMock(return_value=SimpleNamespace(content_json=content))
    )


def _ocr(gateway: Any, *, dpi: int = 150) -> Any:
    return stage._PdfPageOcr(
        source_path=Path("/tmp/doc.pdf"),
        db=object(),
        gateway=gateway,
        dpi=dpi,
        pipeline_run_id=uuid4(),
        parent_job_id=uuid4(),
    )


class TestReadingAnImageOnlyPage:
    """The OCR collaborator. Every ``None`` here means "leave the page as
    the deterministic tiers left it", which is the fail-open contract."""

    async def test_a_page_is_rendered_and_sent_as_a_data_url(
        self, fitz: dict[str, Any]
    ) -> None:
        gateway = _gateway({"text": "Photosynthesis converts light"})

        assert await _ocr(gateway).ocr_page(2) == "Photosynthesis converts light"

        assert fitz["doc"].loaded == [1], "page numbers are 1-based, indices are not"
        assert gateway.generate_json.await_args.kwargs["image_data_url"].startswith(
            "data:image/png;base64,"
        )

    async def test_the_configured_resolution_reaches_the_renderer(
        self, fitz: dict[str, Any]
    ) -> None:
        await _ocr(_gateway({"text": "x"}), dpi=96).ocr_page(1)

        assert fitz["dpis"] == [96]

    async def test_the_document_is_opened_once_across_pages(
        self, fitz: dict[str, Any]
    ) -> None:
        """Reopening per page would re-parse the whole file for every image
        page in a scan."""
        reader = _ocr(_gateway({"text": "x"}))

        await reader.ocr_page(1)
        await reader.ocr_page(2)

        assert len(fitz["opened"]) == 1

    @pytest.mark.parametrize("page_number", [0, -1, 4, 99])
    async def test_a_page_outside_the_document_is_refused_not_raised(
        self, fitz: dict[str, Any], page_number: int
    ) -> None:
        """Page numbers come from the extractor's own markers, so a
        mismatch is a bug -- but raising here would take the ingest with
        it, and the page in question does not exist to be lost.
        """
        gateway = _gateway({"text": "x"})

        assert await _ocr(gateway).ocr_page(page_number) is None

        gateway.generate_json.assert_not_awaited()

    async def test_no_pdf_library_installed_is_a_skip_not_a_crash(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """OCR is an optional extra. A deployment without the renderer
        keeps every other tier."""
        monkeypatch.setitem(sys.modules, "fitz", None)
        gateway = _gateway({"text": "x"})

        assert await _ocr(gateway).ocr_page(1) is None

        gateway.generate_json.assert_not_awaited()

    @pytest.mark.parametrize(
        "content",
        ["a bare string", ["a", "list"], None, 42, {"no_text_key": "here"}],
    )
    async def test_a_response_of_the_wrong_shape_leaves_the_page_alone(
        self, fitz: dict[str, Any], content: Any
    ) -> None:
        assert await _ocr(_gateway(content)).ocr_page(1) is None

    async def test_a_non_string_transcription_is_rejected(
        self, fitz: dict[str, Any]
    ) -> None:
        """The value is substituted for the page body. A list reaching the
        chunker would fail much further downstream, where nothing points
        back to the model that produced it.
        """
        assert await _ocr(_gateway({"text": ["line", "line"]})).ocr_page(1) is None

    async def test_a_blank_page_transcribes_to_an_empty_string(
        self, fitz: dict[str, Any]
    ) -> None:
        """Distinct from ``None``: the model read the page and found
        nothing, which is an answer."""
        assert await _ocr(_gateway({"text": ""})).ocr_page(1) == ""

    def test_closing_before_anything_was_opened_is_harmless(self) -> None:
        """``run_preprocess_stage`` closes in a ``finally``, which runs even
        when the cascade never routed a page to OCR."""
        _ocr(_gateway({"text": "x"})).close()

    async def test_closing_twice_only_closes_the_document_once(
        self, fitz: dict[str, Any]
    ) -> None:
        reader = _ocr(_gateway({"text": "x"}))
        await reader.ocr_page(1)

        reader.close()
        reader.close()

        assert fitz["doc"].closed == 1

    async def test_a_document_that_fails_to_close_is_still_released(
        self, fitz: dict[str, Any]
    ) -> None:
        """The handle is dropped in a ``finally`` so a raising ``close``
        cannot leave the reader holding a document it will never close
        again."""
        reader = _ocr(_gateway({"text": "x"}))
        await reader.ocr_page(1)

        def _boom() -> None:
            raise OSError("file already gone")

        fitz["doc"].close = _boom

        with pytest.raises(OSError, match="already gone"):
            reader.close()

        assert reader._doc is None


class TestLabellingTheAmbiguousPages:
    """The adjudicator. It only sees pages the deterministic rules could not
    settle, and a malformed row must cost that page its label rather than
    the batch."""

    def _adjudicator(self, gateway: Any) -> Any:
        return stage._LlmPageAdjudicator(
            db=object(), gateway=gateway, pipeline_run_id=uuid4(), parent_job_id=uuid4()
        )

    async def test_labels_and_confidences_come_back_keyed_by_page(self) -> None:
        gateway = _gateway(
            {
                "pages": [
                    {"page": 3, "label": "front_matter", "confidence": 0.91},
                    {"page": 7, "label": "body", "confidence": 0.4},
                ]
            }
        )

        out = await self._adjudicator(gateway).classify([(3, "cover"), (7, "content")])

        assert out == {3: ("front_matter", 0.91), 7: ("body", 0.4)}

    async def test_only_a_prefix_of_each_page_is_sent(self) -> None:
        """The label depends on how a page opens, not on all of it. Sending
        whole pages would multiply the token cost of a tier that exists to
        be cheap.
        """
        gateway = _gateway({"pages": []})

        await self._adjudicator(gateway).classify([(1, "w" * 5000)])

        sent = gateway.generate_json.await_args.kwargs["user_prompt"]
        assert '"' + "w" * 400 + '"' in sent
        assert "w" * 401 not in sent

    async def test_the_word_count_is_measured_on_the_whole_page(self) -> None:
        """A near-empty divider and a dense page can open identically, so
        the count is the signal that separates them -- measuring it on the
        truncated prefix would erase the difference.
        """
        gateway = _gateway({"pages": []})

        await self._adjudicator(gateway).classify([(1, "word " * 900)])

        assert '"word_count": 900' in gateway.generate_json.await_args.kwargs["user_prompt"]

    async def test_non_ascii_text_is_sent_unescaped(self) -> None:
        """Escaping it would spend several tokens per accented character in
        a payload that is mostly prose."""
        gateway = _gateway({"pages": []})

        await self._adjudicator(gateway).classify([(1, "Trường Đại học")])

        assert "Trường Đại học" in gateway.generate_json.await_args.kwargs["user_prompt"]

    @pytest.mark.parametrize("content", ["a string", ["a", "list"], None, 42])
    async def test_a_response_of_the_wrong_shape_labels_nothing(self, content: Any) -> None:
        assert await self._adjudicator(_gateway(content)).classify([(1, "x")]) == {}

    async def test_a_response_with_no_pages_key_labels_nothing(self) -> None:
        assert await self._adjudicator(_gateway({"other": 1})).classify([(1, "x")]) == {}

    async def test_a_null_pages_value_labels_nothing(self) -> None:
        assert await self._adjudicator(_gateway({"pages": None})).classify([(1, "x")]) == {}

    @pytest.mark.parametrize(
        ("label", "row"),
        [
            ("not an object", "front_matter"),
            ("no page number", {"label": "body", "confidence": 0.9}),
            ("no label", {"page": 1, "confidence": 0.9}),
            ("page is not a number", {"page": "three", "label": "body"}),
            ("page is null", {"page": None, "label": "body"}),
            ("confidence is not a number", {"page": 1, "label": "body", "confidence": "high"}),
        ],
    )
    async def test_one_malformed_row_costs_that_page_only(
        self, label: str, row: Any
    ) -> None:
        """The batch is ten pages. Discarding all ten because the model
        fumbled one would make this tier useless exactly when it is least
        reliable.
        """
        gateway = _gateway(
            {"pages": [row, {"page": 9, "label": "summary", "confidence": 0.8}]}
        )

        out = await self._adjudicator(gateway).classify([(1, "x"), (9, "y")])

        assert out == {9: ("summary", 0.8)}, label

    async def test_a_missing_confidence_reads_as_no_confidence(self) -> None:
        """Rather than defaulting high. The caller compares against a
        minimum, so an absent value must fall below it and leave the
        deterministic verdict standing.
        """
        gateway = _gateway({"pages": [{"page": 1, "label": "body"}]})

        assert await self._adjudicator(gateway).classify([(1, "x")]) == {1: ("body", 0.0)}

    async def test_a_null_confidence_reads_as_no_confidence(self) -> None:
        gateway = _gateway({"pages": [{"page": 1, "label": "body", "confidence": None}]})

        assert await self._adjudicator(gateway).classify([(1, "x")]) == {1: ("body", 0.0)}

    async def test_a_page_the_batch_never_contained_is_still_returned(self) -> None:
        """Deliberately not filtered here.

        The caller matches labels back to its own units, so an invented page
        number finds no unit and is dropped there -- one place doing the
        matching rather than two disagreeing about it.
        """
        gateway = _gateway({"pages": [{"page": 404, "label": "body", "confidence": 0.9}]})

        assert await self._adjudicator(gateway).classify([(1, "x")]) == {404: ("body", 0.9)}

    async def test_a_numeric_string_page_is_accepted(self) -> None:
        """JSON from a model routinely quotes numbers. Refusing them would
        discard a well-formed label over its type."""
        gateway = _gateway({"pages": [{"page": "5", "label": "divider", "confidence": "0.7"}]})

        assert await self._adjudicator(gateway).classify([(5, "x")]) == {5: ("divider", 0.7)}
