"""Regression coverage for the teacher material preview stream."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from abridgeai.features.materials.services.authoring import _reads as reads_service


async def test_authoring_stream_is_inline_for_media_player() -> None:
    target = SimpleNamespace(
        bucket="materials",
        object_key="lessons/video.mp4",
        title="Lecture 1",
        material_version_id=uuid4(),
    )
    expires_at = datetime(2026, 9, 14, tzinfo=UTC)
    create_stream = AsyncMock(return_value=("https://storage.test/video", expires_at))

    with (
        patch(
            f"{reads_service.__name__}.get_authoring_stream_target_for_material",
            new=AsyncMock(return_value=target),
        ),
        patch(f"{reads_service.__name__}.create_stream_url", new=create_stream),
    ):
        result = await reads_service.get_authoring_stream_url(AsyncMock(), uuid4())

    assert result is not None
    assert result.url == "https://storage.test/video"
    create_stream.assert_awaited_once_with(
        target,
        response_headers={"Content-Disposition": 'inline; filename="Lecture 1"'},
    )
