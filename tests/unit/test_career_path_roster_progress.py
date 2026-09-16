"""Career-path roster percentages use the canonical stage-aware formula."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from abridgeai.features.career_paths.services import enrollment


@pytest.mark.asyncio
async def test_roster_progress_uses_stage_aware_percentage() -> None:
    version_id = uuid4()
    student_id = uuid4()
    row = {
        "student_id": student_id,
        "primary_email": "student@example.com",
        "display_name": "Student",
        "avatar_bucket": None,
        "avatar_object_key": None,
        "overall_percent": 42.86,
        "completed_courses": 3,
        "course_count": 7,
    }
    evals = object()
    with (
        patch.object(
            enrollment.authoring_queries,
            "get_published_version",
            new=AsyncMock(return_value=SimpleNamespace(id=version_id)),
        ),
        patch.object(
            enrollment.student_queries,
            "get_roster_path_progress",
            new=AsyncMock(return_value=[row]),
        ),
        patch.object(
            enrollment.stage_service,
            "evaluate_stages",
            new=AsyncMock(return_value=evals),
        ),
        patch.object(
            enrollment.stage_service,
            "path_progress_percent",
            return_value=100.0,
        ) as progress_percent,
    ):
        result = await enrollment.get_roster_progress(AsyncMock(), uuid4())

    assert result[0].overall_percent == 100.0
    progress_percent.assert_called_once_with(evals)
