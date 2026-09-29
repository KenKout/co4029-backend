"""The thumbnail pipeline: what gets stored, and how it is delivered.

Two contracts live in :mod:`abridgeai.infrastructure.thumbnails`, both
born from the Lighthouse finding that thumbnail delivery dominated LCP
on /dashboard (~1.8 MiB est. savings):

* **Encode** — every non-GIF upload is re-encoded to WebP capped at
  1280 px. These tests pin the observable outcomes (magic bytes, the
  dimension cap, aspect preservation, the no-upscale rule, GIF
  passthrough) rather than exact byte output, which varies by Pillow
  version.
* **Delivery** — one presigned URL per object is memoised in Redis and
  served with ``Cache-Control``. A URL minted per request is a NEW url
  per request (the signature embeds the timestamp), which is precisely
  what defeated the browser cache; the memoisation is what the
  stability tests below assert, using the same fake-cache approach as
  ``tests/unit/courses/test_course_content_cache.py``.
"""

from __future__ import annotations

import io
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import pytest
from PIL import Image
from redis.exceptions import ConnectionError as RedisConnectionError

from abridgeai.core.cache import json_store as json_store_module
from abridgeai.core.cache import keys as keys_module
from abridgeai.infrastructure import thumbnails as thumbnails_module
from abridgeai.infrastructure.thumbnails import (
    THUMBNAIL_CACHE_CONTROL,
    ThumbnailDecodeError,
    encode_thumbnail,
    mint_thumbnail_url,
)


class _FakeCache:
    """Records calls; optionally fails the way a dead Redis would."""

    def __init__(self, raise_on: Exception | None = None) -> None:
        self._raise_on = raise_on
        self.store: dict[str, str] = {}

    async def get(self, key: str) -> Any:
        if self._raise_on:
            raise self._raise_on
        return self.store.get(key)

    async def set(self, key: str, value: str, ex: int | None = None) -> None:
        if self._raise_on:
            raise self._raise_on
        self.store[key] = value


@pytest.fixture
def cache(monkeypatch: pytest.MonkeyPatch) -> _FakeCache:
    client = _FakeCache()
    monkeypatch.setattr(json_store_module, "get_cache", lambda: client)
    return client


def _image_bytes(width: int, height: int, format: str) -> bytes:
    image = Image.new("RGB", (width, height), color=(180, 40, 90))
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    image.close()
    return buffer.getvalue()


def _webp_dimensions(data: bytes) -> tuple[int, int]:
    with Image.open(io.BytesIO(data)) as image:
        return image.size


class TestEncode:
    def test_a_large_jpeg_lands_as_webp_capped_at_1280(self) -> None:
        result = encode_thumbnail(
            _image_bytes(3000, 2000, "JPEG"), content_type="image/jpeg"
        )

        assert result.content_type == "image/webp"
        assert result.extension == "webp"
        assert thumbnails_module.is_webp(result.data)
        assert _webp_dimensions(result.data) == (1280, 853)  # aspect preserved

    def test_a_small_image_is_re_encoded_but_never_upscaled(self) -> None:
        result = encode_thumbnail(
            _image_bytes(400, 300, "PNG"), content_type="image/png"
        )

        assert thumbnails_module.is_webp(result.data)
        assert _webp_dimensions(result.data) == (400, 300)

    def test_portrait_orientation_respects_the_longest_side(self) -> None:
        result = encode_thumbnail(
            _image_bytes(900, 3000, "PNG"), content_type="image/png"
        )

        assert _webp_dimensions(result.data) == (384, 1280)

    def test_gif_passes_through_untouched(self) -> None:
        gif = _image_bytes(2000, 200, "GIF")

        result = encode_thumbnail(gif, content_type="image/gif")

        assert result.data == gif
        assert result.content_type == "image/gif"
        assert result.extension == "gif"

    def test_undecodable_bytes_raise_a_thumbnail_error(self) -> None:
        with pytest.raises(ThumbnailDecodeError, match="invalid_thumbnail"):
            encode_thumbnail(b"png-bytes", content_type="image/png")


def _presign_stub(
    monkeypatch: pytest.MonkeyPatch, url: str
) -> list[tuple[Any, dict[str, str] | None]]:
    """Install a recording presign stub; returns the (target, headers) log."""
    calls: list[tuple[Any, dict[str, str] | None]] = []

    async def _create_stream_url(
        target: Any, *, response_headers: Any = None
    ) -> tuple[str, datetime]:
        calls.append((target, response_headers))
        return url, datetime.now(tz=UTC)

    monkeypatch.setattr(thumbnails_module, "create_stream_url", _create_stream_url)
    return calls


class TestMint:
    async def test_consecutive_mints_return_the_same_url(
        self, cache: _FakeCache, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = _presign_stub(monkeypatch, "https://signed.example/a.webp")

        first = await mint_thumbnail_url("bucket", "course-thumbnails/x/y.webp")
        second = await mint_thumbnail_url("bucket", "course-thumbnails/x/y.webp")

        assert second == first == "https://signed.example/a.webp"
        assert len(calls) == 1  # the second call was served by the cache

    async def test_the_url_carries_cache_control(
        self, cache: _FakeCache, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = _presign_stub(monkeypatch, "https://signed.example/a.webp")

        await mint_thumbnail_url("bucket", "course-thumbnails/x/y.webp")

        assert calls[0][1] == {"Cache-Control": THUMBNAIL_CACHE_CONTROL}

    async def test_a_dead_redis_degrades_to_per_request_minting(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        dead = _FakeCache(raise_on=RedisConnectionError("no route to host"))
        monkeypatch.setattr(json_store_module, "get_cache", lambda: dead)
        calls = _presign_stub(monkeypatch, "https://signed.example/a.webp")

        first = await mint_thumbnail_url("bucket", "course-thumbnails/x/y.webp")
        second = await mint_thumbnail_url("bucket", "course-thumbnails/x/y.webp")

        assert {first, second} == {"https://signed.example/a.webp"}
        assert len(calls) == 2  # every read minted; nothing broke

    async def test_the_cache_ttl_is_clamped_below_the_presign_ttl(
        self, cache: _FakeCache, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        written: dict[str, int] = {}

        async def _set_json(key: str, value: object, *, ttl: int, namespace: str) -> bool:
            written[key] = ttl
            return True

        monkeypatch.setattr(thumbnails_module, "set_json", _set_json)
        monkeypatch.setattr(
            thumbnails_module,
            "get_settings",
            lambda: type("S", (), {"s3_url_ttl_seconds": 120})(),
        )
        calls = _presign_stub(monkeypatch, "https://signed.example/a.webp")

        object_key = f"course-thumbnails/{uuid4()}/{uuid4()}.webp"
        await mint_thumbnail_url("bucket", object_key)

        expected_key = keys_module.THUMBNAIL_URL.format(object_key=object_key)
        assert written[expected_key] == 60  # 120s URL life - 60s safety margin
