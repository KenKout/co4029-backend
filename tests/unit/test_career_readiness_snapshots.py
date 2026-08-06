"""Unit tests for career readiness snapshots (FR-6.8, phase-03).

Service layer only — queries + enrollment progress are mocked. Verifies
score derivation/clamping, batch resilience, overview aggregation and
the router wiring call sites.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from abridgeai.features.career_paths.services import readiness as readiness_service

_STUDENT = uuid.uuid4()
_PATH = uuid.uuid4()
_NOW = datetime(2026, 6, 10, 12, 0, 0, tzinfo=UTC)


def _progress(overall: float, formula_version: int = 1) -> SimpleNamespace:
    """Stand-in for ``CareerPathProgressRead``.

    ``formula_version`` is part of the contract now: the snapshot must be
    stamped with the formula that actually produced the score rather than
    inheriting the column default.
    """
    return SimpleNamespace(overall_percent=overall, formula_version=formula_version)


class TestComputeReadinessScore:
    async def test_uses_overall_percent_rounded(self) -> None:
        with patch.object(
            readiness_service.enrollment_service,
            "get_my_path_progress",
            new=AsyncMock(return_value=_progress(66.666)),
        ):
            score, version = await readiness_service.compute_readiness_score(
                AsyncMock(), career_path_id=_PATH, student_id=_STUDENT
            )
        assert score == Decimal("66.67")
        assert version == 1

    async def test_clamps_to_bounds(self) -> None:
        with patch.object(
            readiness_service.enrollment_service,
            "get_my_path_progress",
            new=AsyncMock(return_value=_progress(105.0)),
        ):
            score, _version = await readiness_service.compute_readiness_score(
                AsyncMock(), career_path_id=_PATH, student_id=_STUDENT
            )
        assert score == Decimal("100")

    async def test_returns_the_active_formula_version(self) -> None:
        """The version travels WITH the score so the caller cannot stamp a
        snapshot with a formula it did not use."""
        with patch.object(
            readiness_service.enrollment_service,
            "get_my_path_progress",
            new=AsyncMock(return_value=_progress(50.0, formula_version=2)),
        ):
            score, version = await readiness_service.compute_readiness_score(
                AsyncMock(), career_path_id=_PATH, student_id=_STUDENT
            )
        assert score == Decimal("50.00")
        assert version == 2


def _db_with_savepoints() -> MagicMock:
    """AsyncSession stand-in whose ``begin_nested()`` yields an async CM."""
    db = MagicMock()

    def _nested() -> MagicMock:
        cm = MagicMock()
        cm.__aenter__ = AsyncMock(return_value=cm)
        cm.__aexit__ = AsyncMock(return_value=False)
        return cm

    db.begin_nested = MagicMock(side_effect=lambda: _nested())
    return db


class TestSnapshotBatch:
    async def test_snapshot_all_active_counts_and_survives_failures(self) -> None:
        """A mid-batch failure must not poison later pairs (SAVEPOINT per pair)."""
        pairs = [(_STUDENT, _PATH), (uuid.uuid4(), _PATH), (uuid.uuid4(), _PATH)]
        snapshot = AsyncMock(side_effect=[Decimal("10"), RuntimeError("boom"), Decimal("30")])
        db = _db_with_savepoints()
        with (
            patch.object(
                readiness_service.readiness_queries,
                "list_active_enrollment_pairs",
                new=AsyncMock(return_value=pairs),
            ),
            patch.object(readiness_service, "snapshot_enrollment", new=snapshot),
        ):
            written = await readiness_service.snapshot_all_active_enrollments(db)
        assert written == 2
        assert snapshot.await_count == 3
        # One SAVEPOINT per pair — including the failing one.
        assert db.begin_nested.call_count == 3

    async def test_snapshot_enrollment_persists_computed_score(self) -> None:
        insert = AsyncMock()
        with (
            patch.object(
                readiness_service.enrollment_service,
                "get_my_path_progress",
                new=AsyncMock(return_value=_progress(42.0)),
            ),
            patch.object(
                readiness_service.enrollment_service,
                "sync_enrollment_completion",
                new=AsyncMock(return_value=False),
            ),
            patch.object(readiness_service.readiness_queries, "insert_snapshot", new=insert),
        ):
            score = await readiness_service.snapshot_enrollment(
                AsyncMock(), career_path_id=_PATH, student_id=_STUDENT
            )
        assert score == Decimal("42.00")
        insert.assert_awaited_once()
        assert insert.await_args.kwargs["readiness_score"] == Decimal("42.00")

    async def test_snapshot_stamps_the_version_actually_used(self) -> None:
        """Guards review point #2: the write must pass ``formula_version``
        EXPLICITLY. Relying on the column default (1) would mislabel every
        snapshot taken after the formula flipped to 2."""
        insert = AsyncMock()
        with (
            patch.object(
                readiness_service.enrollment_service,
                "get_my_path_progress",
                new=AsyncMock(return_value=_progress(42.0, formula_version=2)),
            ),
            patch.object(
                readiness_service.enrollment_service,
                "sync_enrollment_completion",
                new=AsyncMock(return_value=False),
            ),
            patch.object(readiness_service.readiness_queries, "insert_snapshot", new=insert),
        ):
            await readiness_service.snapshot_enrollment(
                AsyncMock(), career_path_id=_PATH, student_id=_STUDENT
            )
        assert insert.await_args.kwargs["formula_version"] == 2


class TestOverview:
    async def test_overview_averages_latest_scores(self) -> None:
        rows = [
            {
                "student_id": _STUDENT,
                "student_email": "a@test.local",
                "readiness_score": Decimal("80.00"),
                "captured_at": _NOW,
            },
            {
                "student_id": uuid.uuid4(),
                "student_email": "b@test.local",
                "readiness_score": Decimal("40.00"),
                "captured_at": _NOW,
            },
        ]
        with patch.object(
            readiness_service.readiness_queries,
            "latest_snapshots_for_path",
            new=AsyncMock(return_value=rows),
        ):
            overview = await readiness_service.get_path_readiness_overview(AsyncMock(), _PATH)
        assert overview.student_count == 2
        assert overview.average_score == 60.0
        assert overview.students[0].student_email == "a@test.local"

    async def test_overview_empty_path(self) -> None:
        with patch.object(
            readiness_service.readiness_queries,
            "latest_snapshots_for_path",
            new=AsyncMock(return_value=[]),
        ):
            overview = await readiness_service.get_path_readiness_overview(AsyncMock(), _PATH)
        assert overview.student_count == 0
        assert overview.average_score is None


class TestRouterWiring:
    async def test_learner_history_route_calls_service(self) -> None:
        from abridgeai.features.career_paths.routers import learner as learner_router

        history = AsyncMock(return_value=[])
        with patch.object(learner_router.readiness_service, "get_my_readiness_history", history):
            await learner_router.get_my_readiness_history(
                _PATH, SimpleNamespace(user_id=_STUDENT), AsyncMock()
            )
        history.assert_awaited_once()
        assert history.await_args.kwargs["career_path_id"] == _PATH

    async def test_management_overview_route_calls_service(self) -> None:
        from abridgeai.features.career_paths.routers import authoring as authoring_router

        overview = AsyncMock(return_value=SimpleNamespace())
        with (
            patch.object(
                authoring_router.readiness_service, "get_path_readiness_overview", overview
            ),
            # FR-2.6 org-scope gate (phase-07) — covered by its own tests.
            patch.object(authoring_router, "_ensure_caller_in_path_org", new=AsyncMock()),
        ):
            await authoring_router.get_path_readiness_overview(
                _PATH, SimpleNamespace(user_id=_STUDENT), AsyncMock()
            )
        overview.assert_awaited_once()


class TestWorkerTask:
    async def test_task_commits_and_returns_count(self) -> None:
        from abridgeai.features.career_paths.workers.readiness import (
            snapshot_career_readiness_task,
        )

        db = AsyncMock()
        ctx_mgr = MagicMock()
        ctx_mgr.__aenter__ = AsyncMock(return_value=db)
        ctx_mgr.__aexit__ = AsyncMock(return_value=False)
        session_factory = MagicMock(return_value=ctx_mgr)
        with (
            patch(
                "abridgeai.features.career_paths.workers.readiness.get_sessionmaker",
                return_value=session_factory,
            ),
            patch(
                "abridgeai.features.career_paths.workers.readiness.readiness_service.snapshot_all_active_enrollments",
                new=AsyncMock(return_value=5),
            ),
        ):
            count = await snapshot_career_readiness_task({})
        assert count == 5
        db.commit.assert_awaited_once()
