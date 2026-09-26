"""Exit snapshots use the single stage-aware career-path progress formula."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from abridgeai.features.learning_programs import queries
from abridgeai.features.learning_programs.schemas import PathExitSnapshot


@pytest.mark.asyncio
async def test_build_exit_snapshot_uses_stage_aware_progress_and_keeps_raw_counts() -> None:
    """The query delegates its percentage to the stage-aware public API."""
    course_ids = [uuid4() for _ in range(7)]
    rows = [
        {
            "course_id": course_id,
            "title": f"Course {index + 1}",
            "slug": f"course-{index + 1}",
            "completion_percent": 100.0 if index < 3 else 40.0,
            "satisfied": index < 3,
        }
        for index, course_id in enumerate(course_ids)
    ]
    db = SimpleNamespace()
    attempt = SimpleNamespace(
        career_path_id=uuid4(),
        career_path_version_id=uuid4(),
    )
    student_id = uuid4()
    stage_progress = AsyncMock(return_value=100.0)

    with patch.object(queries, "career_paths_api", create=True) as career_paths_api:
        career_paths_api.get_version_course_progress_for_user = AsyncMock(return_value=rows)
        career_paths_api.get_version_progress_percent_for_user = stage_progress
        snapshot = await queries.build_exit_snapshot(
            db,  # type: ignore[arg-type]
            student_id=student_id,
            attempt=attempt,  # type: ignore[arg-type]
        )

    assert snapshot["overall_percent"] == 100.0
    assert snapshot["schema_version"] == 2
    assert "formula_version" not in snapshot
    assert snapshot["completed_courses"] == 3
    assert snapshot["total_courses"] == 7
    assert snapshot["courses"][3] == {
        "course_id": str(course_ids[3]),
        "title": "Course 4",
        "slug": "course-4",
        "progress_percent": 40.0,
        "completed": False,
    }
    career_paths_api.get_version_course_progress_for_user.assert_awaited_once_with(
        db,
        version_id=attempt.career_path_version_id,
        student_id=student_id,
    )
    stage_progress.assert_awaited_once_with(
        db,
        version_id=attempt.career_path_version_id,
        student_id=student_id,
    )


def test_exit_snapshot_schema_keeps_legacy_rows_readable() -> None:
    """Pre-v2 JSONB rows gain safe defaults instead of failing API serialization."""
    path_id = uuid4()
    version_id = uuid4()

    snapshot = PathExitSnapshot.model_validate(
        {
            "career_path_id": str(path_id),
            "career_path_version_id": str(version_id),
            "completed_course_ids": [],
            "completed_courses": 0,
            "total_courses": 2,
            "overall_percent": 0,
            "captured_at": "2026-09-26T10:30:00Z",
        }
    )

    assert snapshot.schema_version == 1
    assert snapshot.courses == []


@pytest.mark.asyncio
async def test_terminalize_live_assessments_closes_both_assessment_types() -> None:
    db = SimpleNamespace(execute=AsyncMock())

    await queries._terminalize_live_assessments_for_enrollments(db, [uuid4()])

    assert db.execute.await_count == 2
    quiz_sql = str(db.execute.await_args_list[0].args[0])
    interview_sql = str(db.execute.await_args_list[1].args[0])
    assert "UPDATE quiz_attempts" in quiz_sql
    assert "status = 'abandoned'" in quiz_sql
    assert "UPDATE interview_sessions" in interview_sql
    assert "status = 'abandoned'" in interview_sql
    assert "AND s.status = 'in_progress'" in interview_sql
