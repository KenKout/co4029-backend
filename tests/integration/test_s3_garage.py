"""Live Garage round-trip tests for ``abridgeai.infrastructure.s3``.

Skipped by default. Run against a running Garage container with::

    docker compose up -d garage
    uv run pytest tests/integration/test_s3_garage.py -m s3_live

Configuration is read from ``GARAGE_*`` env vars when present, otherwise
defaults match ``backend/garage.toml`` for local dev.
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass

import httpx
import pytest
from pydantic import SecretStr

from abridgeai.core.config import Settings
from abridgeai.infrastructure.s3 import (
    CompletedPart,
    abort_multipart_upload,
    complete_multipart_upload,
    create_multipart_upload,
    create_upload_url,
    delete_object,
    download_to_temp,
    head_object,
)

pytestmark = pytest.mark.s3_live


@dataclass(frozen=True)
class _Obj:
    bucket: str
    object_key: str


def _live_settings() -> Settings:
    access = os.environ.get("GARAGE_ACCESS_KEY") or os.environ.get("AWS_ACCESS_KEY_ID")
    secret = os.environ.get("GARAGE_SECRET_KEY") or os.environ.get("AWS_SECRET_ACCESS_KEY")
    endpoint = os.environ.get("GARAGE_ENDPOINT_URL") or os.environ.get(
        "AWS_ENDPOINT_URL", "http://localhost:3900"
    )
    public_endpoint = os.environ.get("GARAGE_PUBLIC_ENDPOINT_URL") or os.environ.get(
        "AWS_PUBLIC_ENDPOINT_URL", endpoint
    )
    bucket = os.environ.get("GARAGE_BUCKET") or os.environ.get("S3_BUCKET_NAME", "abridgeai-test")
    if not access or not secret:
        pytest.skip("Garage credentials not provided (GARAGE_ACCESS_KEY/SECRET_KEY)")
    try:
        httpx.get(endpoint, timeout=2.0)
    except httpx.HTTPError:
        pytest.skip(f"Garage is not reachable at {endpoint}")
    return Settings(
        aws_access_key_id=SecretStr(access),
        aws_secret_access_key=SecretStr(secret),
        aws_endpoint_url=endpoint,
        aws_public_endpoint_url=public_endpoint,
        aws_region=os.environ.get("AWS_REGION", "us-east-1"),
        s3_bucket_name=bucket,
        s3_url_ttl_seconds=3600,
    )


async def test_garage_full_lifecycle() -> None:
    settings = _live_settings()
    key = f"abridgeai-tests/t2.3-{uuid.uuid4()}.bin"
    obj = _Obj(bucket=settings.s3_bucket_name, object_key=key)
    payload = b"garage-live-bytes-" * 4096

    upload_url, _ = await create_upload_url(obj, settings=settings)
    async with httpx.AsyncClient() as http:
        put = await http.put(upload_url, content=payload)
    assert put.status_code in (200, 204), put.text

    meta = await head_object(obj, settings=settings)
    assert meta is not None
    assert meta.size == len(payload)

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        out = await download_to_temp(obj, Path(tmp), settings=settings)
        assert out.read_bytes() == payload

    assert await delete_object(obj, settings=settings) is True
    assert await head_object(obj, settings=settings) is None


async def test_garage_multipart_lifecycle() -> None:
    settings = _live_settings()
    key = f"abridgeai-tests/t2.3-multipart-{uuid.uuid4()}.bin"
    obj = _Obj(bucket=settings.s3_bucket_name, object_key=key)
    part_bytes = b"m" * (5 * 1024 * 1024)

    init = await create_multipart_upload(obj, part_count=2, settings=settings)
    completed: list[CompletedPart] = []
    async with httpx.AsyncClient() as http:
        for part in init.parts:
            resp = await http.put(part.url, content=part_bytes)
            assert resp.status_code in (200, 204), resp.text
            completed.append(CompletedPart(part_number=part.part_number, etag=resp.headers["ETag"]))
    await complete_multipart_upload(obj, init.upload_id, completed, settings=settings)

    meta = await head_object(obj, settings=settings)
    assert meta is not None
    assert meta.size == 2 * len(part_bytes)

    await delete_object(obj, settings=settings)

    abort_obj = _Obj(
        bucket=settings.s3_bucket_name,
        object_key=f"abridgeai-tests/t2.3-abort-{uuid.uuid4()}.bin",
    )
    abort_init = await create_multipart_upload(abort_obj, part_count=1, settings=settings)
    await abort_multipart_upload(abort_obj, abort_init.upload_id, settings=settings)
    assert await head_object(abort_obj, settings=settings) is None
