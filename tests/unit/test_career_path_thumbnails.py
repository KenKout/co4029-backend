import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest

from abridgeai.features.career_paths.services import authoring
from abridgeai.features.learning_programs import services as program_services


def test_upload_rejects_unsupported_image_before_database_access() -> None:
    with pytest.raises(authoring.ThumbnailUploadError, match="unsupported_thumbnail_type"):
        asyncio.run(
            authoring.upload_career_path_thumbnail(
                AsyncMock(),
                uuid4(),
                data=b"not-an-image",
                content_type="text/plain",
                uploaded_by=uuid4(),
            )
        )


def test_upload_rejects_oversized_thumbnail() -> None:
    with pytest.raises(authoring.ThumbnailUploadError, match="thumbnail_too_large"):
        asyncio.run(
            authoring.upload_career_path_thumbnail(
                AsyncMock(),
                uuid4(),
                data=b"x" * (5 * 1024 * 1024 + 1),
                content_type="image/png",
                uploaded_by=uuid4(),
            )
        )


def test_upload_stores_image_and_updates_path(monkeypatch: pytest.MonkeyPatch) -> None:
    path_id = uuid4()
    actor_id = uuid4()
    path = SimpleNamespace(id=path_id, thumbnail_object_id=None, updated_by=None)
    db = SimpleNamespace(
        flush=AsyncMock(),
        commit=AsyncMock(),
        refresh=AsyncMock(),
    )
    put = AsyncMock()
    insert = Mock()
    result = object()

    monkeypatch.setattr(authoring, "_require_path", AsyncMock(return_value=path))
    monkeypatch.setattr(authoring, "put_object_bytes", put)
    monkeypatch.setattr(
        authoring.authoring_queries, "insert_thumbnail_storage_object", insert
    )
    monkeypatch.setattr(authoring, "get_career_path", AsyncMock(return_value=result))

    actual = asyncio.run(
        authoring.upload_career_path_thumbnail(
            db,
            path_id,
            data=b"png-bytes",
            content_type="image/png",
            uploaded_by=actor_id,
        )
    )

    assert actual is result
    assert path.thumbnail_object_id is not None
    assert path.updated_by == actor_id
    assert "career-path-thumbnails" in put.await_args.args[0].object_key
    insert.assert_called_once()
    db.flush.assert_awaited_once()
    db.commit.assert_awaited_once()


def test_program_path_payload_includes_thumbnail_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path_id = uuid4()
    version_id = uuid4()
    thumbnail_url = "https://storage.example/path-thumbnail.png"
    rows = [
        {
            "career_path_id": path_id,
            "career_path_version_id": uuid4(),
            "career_path_version_no": 2,
            "name": "Software Engineering",
            "slug": "software-engineering",
            "description": None,
            "status": "published",
            "position": 1,
        }
    ]
    monkeypatch.setattr(
        program_services.queries, "list_version_paths", AsyncMock(return_value=rows)
    )
    monkeypatch.setattr(
        program_services.career_paths_api,
        "get_career_path_thumbnail_urls",
        AsyncMock(return_value={path_id: thumbnail_url}),
    )

    result = asyncio.run(program_services._paths_for_version(AsyncMock(), version_id))

    assert result[0].thumbnail_url == thumbnail_url
