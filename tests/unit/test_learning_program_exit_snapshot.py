"""Exit snapshots use the single stage-aware career-path progress formula."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from abridgeai.features.learning_programs import queries


@pytest.mark.asyncio
async def test_build_exit_snapshot_uses_stage_aware_progress_and_keeps_raw_counts() -> None:
    """The query delegates its percentage to the stage-aware public API."""
    course_ids = [uuid4() for _ in range(7)]
    rows = [
        {"course_id": course_id, "completed": index < 3}
        for index, course_id in enumerate(course_ids)
    ]
    mappings = SimpleNamespace(all=lambda: rows)
    db = SimpleNamespace(
        execute=AsyncMock(
            return_value=SimpleNamespace(mappings=lambda: mappings),
        )
    )
    attempt = SimpleNamespace(
        career_path_id=uuid4(),
        career_path_version_id=uuid4(),
    )
    student_id = uuid4()
    stage_progress = AsyncMock(return_value=100.0)

    with patch.object(queries, "career_paths_api", create=True) as career_paths_api:
        career_paths_api.get_version_progress_percent_for_user = stage_progress
        snapshot = await queries.build_exit_snapshot(
            db,  # type: ignore[arg-type]
            student_id=student_id,
            attempt=attempt,  # type: ignore[arg-type]
        )

    assert snapshot["overall_percent"] == 100.0
    assert "formula_version" not in snapshot
    assert snapshot["completed_courses"] == 3
    assert snapshot["total_courses"] == 7
    stage_progress.assert_awaited_once_with(
        db,
        version_id=attempt.career_path_version_id,
        student_id=student_id,
    )
