import asyncio
import io
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest
from PIL import Image

from abridgeai.features.career_paths.services import authoring
from abridgeai.features.learning_programs import services as program_services
from abridgeai.infrastructure.thumbnails import is_webp


def _png_bytes(width: int = 64, height: int = 48) -> bytes:
    image = Image.new("RGB", (width, height), color=(120, 160, 200))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    image.close()
    return buffer.getvalue()


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


def test_upload_rejects_undecodable_bytes_as_a_client_error() -> None:
    with pytest.raises(authoring.ThumbnailUploadError, match="invalid_thumbnail"):
        asyncio.run(
            authoring.upload_career_path_thumbnail(
                AsyncMock(),
                uuid4(),
                data=b"png-bytes",  # allowed Content-Type, corrupt body
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
            data=_png_bytes(64, 48),
            content_type="image/png",
            uploaded_by=actor_id,
        )
    )

    assert actual is result
    assert path.thumbnail_object_id is not None
    assert path.updated_by == actor_id
    target = put.await_args.args[0]
    stored = put.await_args.args[1]
    # Uploads are re-encoded server-side: stored bytes are WebP under a
    # .webp key, never the verbatim upload.
    assert "career-path-thumbnails" in target.object_key
    assert target.object_key.endswith(".webp")
    assert is_webp(stored)
    assert put.await_args.kwargs["content_type"] == "image/webp"
    insert_kwargs = insert.call_args.kwargs
    assert insert_kwargs["object_key"].endswith(".webp")
    assert insert_kwargs["mime_type"] == "image/webp"
    assert insert_kwargs["size_bytes"] == len(stored)
    db.flush.assert_awaited_once()
    db.commit.assert_awaited_once()


def test_upload_passes_gif_through_verbatim(monkeypatch: pytest.MonkeyPatch) -> None:
    path = SimpleNamespace(id=uuid4(), thumbnail_object_id=None, updated_by=None)
    db = SimpleNamespace(flush=AsyncMock(), commit=AsyncMock(), refresh=AsyncMock())
    gif = io.BytesIO()
    Image.new("P", (32, 32)).save(gif, format="GIF")
    gif_bytes = gif.getvalue()
    put = AsyncMock()

    monkeypatch.setattr(authoring, "_require_path", AsyncMock(return_value=path))
    monkeypatch.setattr(authoring, "put_object_bytes", put)
    monkeypatch.setattr(
        authoring.authoring_queries,
        "insert_thumbnail_storage_object",
        Mock(),
    )
    monkeypatch.setattr(authoring, "get_career_path", AsyncMock(return_value=object()))

    asyncio.run(
        authoring.upload_career_path_thumbnail(
            db,
            path.id,
            data=gif_bytes,
            content_type="image/gif",
            uploaded_by=uuid4(),
        )
    )

    target = put.await_args.args[0]
    assert target.object_key.endswith(".gif")
    assert put.await_args.args[1] == gif_bytes  # animation preserved
    assert put.await_args.kwargs["content_type"] == "image/gif"


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
