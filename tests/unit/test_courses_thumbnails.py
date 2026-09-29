"""Course thumbnail upload: same pipeline contract as the career-path twin.

Pins the observable behaviour of
``courses.services.authoring.upload_course_thumbnail`` after the
Lighthouse-driven re-encode change: stored bytes are WebP under a
``.webp`` key (never the verbatim upload), corrupt bodies fail as the
existing 422 ``ThumbnailUploadError``, and GIFs pass through verbatim.
"""

from __future__ import annotations

import asyncio
import io
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from PIL import Image

from abridgeai.features.courses.services import authoring
from abridgeai.infrastructure.thumbnails import is_webp


def _image_bytes(width: int, height: int, format: str) -> bytes:
    image = Image.new("RGB", (width, height), color=(90, 130, 190))
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    image.close()
    return buffer.getvalue()


class _FakeSession:
    """The slice of AsyncSession upload_course_thumbnail actually uses."""

    def __init__(self) -> None:
        self.flush = AsyncMock()
        self.commit = AsyncMock()
        self.refresh = AsyncMock()
        self.added: list[Any] = []

    def add(self, obj: Any) -> None:
        self.added.append(obj)


def _patch_upload_guts(monkeypatch: pytest.MonkeyPatch, *, course: object) -> _FakeSession:
    """Stub everything upload_course_thumbnail touches around storage."""
    dto = SimpleNamespace(thumbnail_url=None)
    monkeypatch.setattr(authoring, "_require_course", AsyncMock(return_value=course))
    monkeypatch.setattr(
        authoring, "CourseAuthoring", SimpleNamespace(model_validate=lambda _c: dto)
    )
    monkeypatch.setattr(authoring, "_mint_thumbnail_url", AsyncMock(return_value=None))
    return _FakeSession()


def test_upload_rejects_undecodable_bytes_as_a_client_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    course = SimpleNamespace(id=uuid4(), thumbnail_object_id=None)
    _patch_upload_guts(monkeypatch, course=course)

    with pytest.raises(authoring.ThumbnailUploadError, match="invalid_thumbnail"):
        asyncio.run(
            authoring.upload_course_thumbnail(
                AsyncMock(),
                course.id,
                data=b"jpeg-bytes",  # allowed Content-Type, corrupt body
                content_type="image/jpeg",
                uploaded_by=uuid4(),
            )
        )


def test_upload_rejects_before_database_access(monkeypatch: pytest.MonkeyPatch) -> None:
    require = AsyncMock()
    monkeypatch.setattr(authoring, "_require_course", require)

    with pytest.raises(authoring.ThumbnailUploadError, match="unsupported_thumbnail_type"):
        asyncio.run(
            authoring.upload_course_thumbnail(
                AsyncMock(),
                uuid4(),
                data=b"anything",
                content_type="text/html",
                uploaded_by=uuid4(),
            )
        )
    require.assert_not_awaited()


def test_upload_stores_reencoded_webp(monkeypatch: pytest.MonkeyPatch) -> None:
    course_id = uuid4()
    course = SimpleNamespace(id=course_id, thumbnail_object_id=None)
    db = _patch_upload_guts(monkeypatch, course=course)
    put = AsyncMock()
    monkeypatch.setattr(authoring, "put_object_bytes", put)

    asyncio.run(
        authoring.upload_course_thumbnail(
            db,
            course_id,
            data=_image_bytes(3000, 2000, "JPEG"),
            content_type="image/jpeg",
            uploaded_by=uuid4(),
        )
    )

    target = put.await_args.args[0]
    stored = put.await_args.args[1]
    assert "course-thumbnails" in target.object_key
    assert target.object_key.endswith(".webp")
    assert put.await_args.kwargs["content_type"] == "image/webp"
    # Re-encoded, capped at 1280px — the LCP fix in one assertion pair.
    assert is_webp(stored)
    with Image.open(io.BytesIO(stored)) as reloaded:
        assert max(reloaded.size) <= 1280
    assert course.thumbnail_object_id is not None
    db.commit.assert_awaited_once()


def test_upload_passes_gif_through_verbatim(monkeypatch: pytest.MonkeyPatch) -> None:
    course = SimpleNamespace(id=uuid4(), thumbnail_object_id=None)
    db = _patch_upload_guts(monkeypatch, course=course)
    put = AsyncMock()
    monkeypatch.setattr(authoring, "put_object_bytes", put)
    gif = _image_bytes(64, 64, "GIF")

    asyncio.run(
        authoring.upload_course_thumbnail(
            db,
            course.id,
            data=gif,
            content_type="image/gif",
            uploaded_by=uuid4(),
        )
    )

    target = put.await_args.args[0]
    assert target.object_key.endswith(".gif")
    assert put.await_args.args[1] == gif  # animation preserved
    assert put.await_args.kwargs["content_type"] == "image/gif"
