"""Authoring read-side projections for the material hub.

Three of these functions carry a decision the caller cannot see, and each
is the reason its own test class exists.

``get_processing_progress`` has to choose between two sources that
routinely disagree. The ingest pipeline runs inside one long database
transaction that commits only at the end, so under MVCC the
``processing_jobs`` row and the version's status stay frozen at their
pre-run values for the entire run -- on a *reprocess* that frozen snapshot
is the previous run's "ready, 100%". Redis is written outside the
transaction and is the only thing that moves while the work is happening.
So a live key wins outright rather than being reconciled with the database
row, and getting that precedence backwards shows a teacher "embedding,
100%" for several minutes.

``get_lesson_knowledge_graph`` talks to Neo4j, which is optional
infrastructure. Every failure mode has to come back as a well-formed empty
graph, because the alternative is a 500 on a panel that is a sidebar
nicety rather than the point of the page.

``get_authoring_stream_url`` interpolates a teacher-supplied title into an
HTTP header.

The database, Redis and Neo4j are all mocked.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest

from abridgeai.features.materials.services.authoring import _reads


def _material(material_id: UUID, version_id: UUID | None) -> SimpleNamespace:
    return SimpleNamespace(id=material_id, current_version_id=version_id)


def _version(version_id: UUID, *, status: str = "processing", error: str | None = None):
    return SimpleNamespace(id=version_id, processing_status=status, processing_error=error)


@pytest.fixture
def progress_world(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """A material with a current version and a job sitting at 40%."""
    material_id, version_id = uuid4(), uuid4()
    version = _version(version_id)

    monkeypatch.setattr(
        _reads,
        "get_material_for_authoring",
        AsyncMock(return_value=_material(material_id, version_id)),
    )
    monkeypatch.setattr(
        _reads,
        "get_latest_processing_job",
        AsyncMock(return_value=SimpleNamespace(progress_percent=40)),
    )
    db = SimpleNamespace(get=AsyncMock(return_value=version))
    return {"db": db, "material_id": material_id, "version_id": version_id, "version": version}


def _live(monkeypatch: pytest.MonkeyPatch, snapshot: dict[str, Any] | None) -> None:
    """Stub the Redis progress read the service imports at call time."""
    import abridgeai.features.materials.ingestion.progress as progress_module

    monkeypatch.setattr(progress_module, "read_progress", AsyncMock(return_value=snapshot))


class TestWhichProgressSourceWins:
    async def test_with_no_live_key_the_database_row_is_reported(
        self, monkeypatch: pytest.MonkeyPatch, progress_world: dict[str, Any]
    ) -> None:
        """The run has finished or never started, and the row is then the
        authoritative answer."""
        _live(monkeypatch, None)

        progress = await _reads.get_processing_progress(
            progress_world["db"], progress_world["material_id"]
        )

        assert progress is not None
        assert progress.progress_percent == 40
        assert progress.processing_status == "processing"
        assert progress.latest_log_line is None

    async def test_a_live_snapshot_overrides_a_higher_database_percent(
        self, monkeypatch: pytest.MonkeyPatch, progress_world: dict[str, Any]
    ) -> None:
        """The case that makes this a precedence rule rather than a maximum.

        On a reprocess the frozen row still holds the *previous* run's 100%,
        so taking the larger of the two would peg the bar at 100 for the
        whole of the new run. The live value is lower and correct.
        """
        progress_world["version"].processing_status = "ready"
        monkeypatch.setattr(
            _reads,
            "get_latest_processing_job",
            AsyncMock(return_value=SimpleNamespace(progress_percent=100)),
        )
        _live(monkeypatch, {"status": "processing", "percent": 30, "stage_label": "chunking"})

        progress = await _reads.get_processing_progress(
            progress_world["db"], progress_world["material_id"]
        )

        assert progress is not None
        assert progress.progress_percent == 30
        assert progress.processing_status == "processing", "not the stale 'ready'"

    async def test_a_live_snapshot_missing_a_status_falls_back_to_the_row(
        self, monkeypatch: pytest.MonkeyPatch, progress_world: dict[str, Any]
    ) -> None:
        _live(monkeypatch, {"percent": 55})

        progress = await _reads.get_processing_progress(
            progress_world["db"], progress_world["material_id"]
        )

        assert progress is not None
        assert progress.progress_percent == 55
        assert progress.processing_status == "processing"

    async def test_a_live_snapshot_missing_a_percent_falls_back_to_the_row(
        self, monkeypatch: pytest.MonkeyPatch, progress_world: dict[str, Any]
    ) -> None:
        _live(monkeypatch, {"status": "processing"})

        progress = await _reads.get_processing_progress(
            progress_world["db"], progress_world["material_id"]
        )
        assert progress is not None
        assert progress.progress_percent == 40

    @pytest.mark.parametrize(("given", "shown"), [(-10, 0), (0, 0), (150, 100)])
    async def test_a_nonsense_live_percent_is_clamped(
        self,
        monkeypatch: pytest.MonkeyPatch,
        progress_world: dict[str, Any],
        given: int,
        shown: int,
    ) -> None:
        """Clamped a second time here, having already been clamped on write.

        The two clamps guard different things: the writer protects the key,
        this protects the response schema from a key written by an older
        build of the worker.
        """
        _live(monkeypatch, {"status": "processing", "percent": given})

        progress = await _reads.get_processing_progress(
            progress_world["db"], progress_world["material_id"]
        )
        assert progress is not None
        assert progress.progress_percent == shown

    async def test_the_error_message_always_comes_from_the_row(
        self, monkeypatch: pytest.MonkeyPatch, progress_world: dict[str, Any]
    ) -> None:
        """The live snapshot has no error field; a failure is committed to
        the database before anyone reads it."""
        progress_world["version"].processing_error = "extraction failed on page 3"
        _live(monkeypatch, {"status": "failed", "percent": 60})

        progress = await _reads.get_processing_progress(
            progress_world["db"], progress_world["material_id"]
        )
        assert progress is not None
        assert progress.error_message == "extraction failed on page 3"


class TestTheLiveStageLine:
    """Long stages loop internally and would otherwise look frozen.

    The knowledge-graph build makes one LLM call per chunk, so it can sit at
    a single percent for minutes. The sub-progress detail is what tells the
    teacher it is alive rather than wedged.
    """

    async def test_a_stage_and_a_detail_are_joined(
        self, monkeypatch: pytest.MonkeyPatch, progress_world: dict[str, Any]
    ) -> None:
        _live(
            monkeypatch,
            {
                "status": "processing",
                "percent": 80,
                "stage_label": "Building knowledge graph",
                "detail": "42/85",
            },
        )

        progress = await _reads.get_processing_progress(
            progress_world["db"], progress_world["material_id"]
        )
        assert progress is not None
        assert progress.latest_log_line == "Building knowledge graph · 42/85"

    @pytest.mark.parametrize(
        ("snapshot", "expected"),
        [
            ({"stage_label": "Chunking", "detail": None}, "Chunking"),
            ({"stage_label": None, "detail": "42/85"}, "42/85"),
            ({"stage_label": None, "detail": None}, None),
            ({}, None),
        ],
    )
    async def test_a_half_present_line_is_not_padded_with_a_separator(
        self,
        monkeypatch: pytest.MonkeyPatch,
        progress_world: dict[str, Any],
        snapshot: dict[str, Any],
        expected: str | None,
    ) -> None:
        """A dangling "·" would read as a truncated message."""
        _live(monkeypatch, {"status": "processing", "percent": 50, **snapshot})

        progress = await _reads.get_processing_progress(
            progress_world["db"], progress_world["material_id"]
        )
        assert progress is not None
        assert progress.latest_log_line == expected


class TestWhenThereIsNoProgressToReport:
    async def test_an_unknown_material_has_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            _reads, "get_material_for_authoring", AsyncMock(return_value=None)
        )
        assert await _reads.get_processing_progress(object(), uuid4()) is None

    async def test_a_material_with_no_current_version_has_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Nothing has been uploaded yet, so there is nothing in flight."""
        monkeypatch.setattr(
            _reads,
            "get_material_for_authoring",
            AsyncMock(return_value=_material(uuid4(), None)),
        )
        assert await _reads.get_processing_progress(object(), uuid4()) is None

    async def test_a_dangling_current_version_has_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The pointer outlived the row it points at."""
        monkeypatch.setattr(
            _reads,
            "get_material_for_authoring",
            AsyncMock(return_value=_material(uuid4(), uuid4())),
        )
        db = SimpleNamespace(get=AsyncMock(return_value=None))
        assert await _reads.get_processing_progress(db, uuid4()) is None

    async def test_a_version_with_no_job_yet_reads_as_zero(
        self, monkeypatch: pytest.MonkeyPatch, progress_world: dict[str, Any]
    ) -> None:
        """Queued but not yet claimed. Zero is the honest number, and the
        alternative is a null the progress bar cannot render."""
        monkeypatch.setattr(
            _reads, "get_latest_processing_job", AsyncMock(return_value=None)
        )
        _live(monkeypatch, None)

        progress = await _reads.get_processing_progress(
            progress_world["db"], progress_world["material_id"]
        )
        assert progress is not None
        assert progress.progress_percent == 0


class TestTheKnowledgeGraphDegradesInsteadOfFailing:
    """Neo4j is optional, and the panel it feeds is a sidebar.

    Every unhappy path returns a well-formed graph so the SPA renders a
    hint. The ``enabled`` flag is what distinguishes "your deployment does
    not have this" from "it is on but has nothing to show", which are
    different messages to a teacher.
    """

    async def test_a_disabled_deployment_says_so(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            _reads, "get_settings", lambda: SimpleNamespace(knowledge_graph_enabled=False)
        )
        lesson_id = uuid4()

        graph = await _reads.get_lesson_knowledge_graph(object(), lesson_id)

        assert graph.enabled is False
        assert graph.lesson_id == lesson_id
        assert graph.nodes == []
        assert graph.edges == []

    async def test_an_unreachable_neo4j_stays_enabled_but_empty(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Enabled and empty, not disabled: the feature is configured and
        the teacher should be told it is temporarily unavailable rather
        than that their deployment lacks it.
        """
        import abridgeai.infrastructure.neo4j as neo4j_module

        monkeypatch.setattr(
            _reads, "get_settings", lambda: SimpleNamespace(knowledge_graph_enabled=True)
        )

        def _explode() -> Any:
            raise ConnectionError("neo4j is unreachable")

        monkeypatch.setattr(neo4j_module, "graph_client", _explode)

        graph = await _reads.get_lesson_knowledge_graph(object(), uuid4())

        assert graph.enabled is True
        assert graph.nodes == []
        assert graph.edges == []


class TestTheStreamUrlHeader:
    async def test_a_quote_in_the_title_is_stripped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The title is teacher-supplied and lands inside a quoted HTTP
        header value. An unescaped quote closes the filename early, and the
        browser saves the document under a truncated name.
        """
        captured: dict[str, Any] = {}
        target = SimpleNamespace(
            title='Week 1: "Intro" notes.pdf', material_version_id=uuid4()
        )
        monkeypatch.setattr(
            _reads,
            "get_authoring_stream_target_for_material",
            AsyncMock(return_value=target),
        )
        monkeypatch.setattr(
            _reads, "get_settings", lambda: SimpleNamespace(s3_url_ttl_seconds=3600)
        )

        async def _create_stream_url(obj: Any, *, response_headers: dict[str, str]):
            captured["headers"] = response_headers
            return "https://files.example.edu/signed", None

        monkeypatch.setattr(_reads, "create_stream_url", _create_stream_url)

        result = await _reads.get_authoring_stream_url(object(), uuid4())

        assert result is not None
        assert '"' not in captured["headers"]["Content-Disposition"].split("filename=")[1][1:-1]
        assert "Intro" in captured["headers"]["Content-Disposition"]
        assert captured["headers"]["Content-Disposition"].startswith("inline;"), (
            "a teacher previewing a material views it in place rather than "
            "downloading it, unlike the learner-facing download path"
        )

    async def test_an_unresolvable_material_has_no_url(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Missing, soft-deleted, or with no storage object behind its
        current version. The router maps ``None`` to a 404."""
        monkeypatch.setattr(
            _reads,
            "get_authoring_stream_target_for_material",
            AsyncMock(return_value=None),
        )
        assert await _reads.get_authoring_stream_url(object(), uuid4()) is None

    async def test_the_expiry_is_reported_as_an_absolute_time(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The SPA refreshes the URL before it lapses, so it needs a wall
        clock instant rather than a duration it would have to anchor."""
        target = SimpleNamespace(title="notes.pdf", material_version_id=uuid4())
        monkeypatch.setattr(
            _reads,
            "get_authoring_stream_target_for_material",
            AsyncMock(return_value=target),
        )
        monkeypatch.setattr(
            _reads, "get_settings", lambda: SimpleNamespace(s3_url_ttl_seconds=900)
        )

        async def _create_stream_url(_obj: Any, *, response_headers: dict[str, str]):
            del response_headers
            return "https://files.example.edu/signed", None

        monkeypatch.setattr(_reads, "create_stream_url", _create_stream_url)

        result = await _reads.get_authoring_stream_url(object(), uuid4())

        assert result is not None
        assert result.expires_at.tzinfo is not None
        assert result.material_version_id == target.material_version_id


class TestTheLessonSummary:
    async def test_a_lesson_with_no_materials_is_a_row_of_zeroes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Returning nothing would make the SPA null-check before rendering
        an empty state it can already draw from zeroes.
        """
        counts = {
            "materials_total": 0,
            "versions_total": 0,
            "pending_versions": 0,
            "processing_versions": 0,
            "completed_versions": 0,
            "failed_versions": 0,
        }
        monkeypatch.setattr(
            _reads, "get_lesson_processing_summary", AsyncMock(return_value=counts)
        )
        lesson_id = uuid4()

        summary = await _reads.get_lesson_processing_summary_view(object(), lesson_id)

        assert summary.lesson_id == lesson_id
        assert summary.materials_total == 0
        assert summary.failed_versions == 0

    async def test_the_counts_are_passed_through_verbatim(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The five in-flight statuses are already collapsed into one bucket
        by the query. Re-deriving any of these here would give the badge a
        second opinion about the same lesson.
        """
        counts = {
            "materials_total": 5,
            "versions_total": 7,
            "pending_versions": 1,
            "processing_versions": 2,
            "completed_versions": 3,
            "failed_versions": 1,
        }
        monkeypatch.setattr(
            _reads, "get_lesson_processing_summary", AsyncMock(return_value=counts)
        )

        summary = await _reads.get_lesson_processing_summary_view(object(), uuid4())

        assert summary.model_dump(exclude={"lesson_id"}) == counts
