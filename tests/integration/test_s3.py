"""Integration tests for ``abridgeai.infrastructure.s3`` against ``moto``.

Covers the 9 QA scenarios from the plan (T2.3 §3267-3354):

1. Direct-upload flow (zero backend bandwidth)
2. Public vs internal endpoint split
3. AWS S3 production mode (no endpoint override)
4. ``head_object`` verifies upload completion
5. Multipart upload + abort cleanup
6. ``delete_object`` idempotent
7. Worker ``download_to_temp`` uses internal endpoint
8. Misconfiguration raises ``S3NotConfiguredError``

Live-Garage round-trip lives in ``test_s3_garage.py`` (marked
``@pytest.mark.s3_live``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import httpx
import pytest
from moto.server import ThreadedMotoServer
from pydantic import SecretStr

from abridgeai.core.config import Settings
from abridgeai.infrastructure import s3 as s3_module
from abridgeai.infrastructure.errors import S3NotConfiguredError, S3NotFoundError
from abridgeai.infrastructure.s3 import (
    CompletedPart,
    abort_multipart_upload,
    complete_multipart_upload,
    create_multipart_upload,
    create_stream_url,
    create_upload_url,
    delete_object,
    download_to_temp,
    head_object,
    put_object_bytes,
)

BUCKET = "abridgeai-test"


@dataclass(frozen=True)
class _Obj:
    bucket: str
    object_key: str


@pytest.fixture
def moto_server() -> ThreadedMotoServer:
    server = ThreadedMotoServer(port=0)
    server.start()
    host, port = server.get_host_and_port()
    server._host = host  # type: ignore[attr-defined]
    server._port = port  # type: ignore[attr-defined]
    yield server
    server.stop()


def _endpoint(server: ThreadedMotoServer) -> str:
    return f"http://{server._host}:{server._port}"  # type: ignore[attr-defined]


def _settings(
    server: ThreadedMotoServer | None,
    *,
    public_endpoint: str | None = None,
    internal_endpoint: str | None = None,
    creds: bool = True,
) -> Settings:
    if server is not None:
        if internal_endpoint is None:
            internal_endpoint = _endpoint(server)
        if public_endpoint is None:
            public_endpoint = _endpoint(server)
    return Settings(
        aws_access_key_id=SecretStr("AKIAIOSFODNN7EXAMPLE") if creds else None,
        aws_secret_access_key=SecretStr("wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY")
        if creds
        else None,
        aws_endpoint_url=internal_endpoint,
        aws_public_endpoint_url=public_endpoint,
        aws_region="us-east-1",
        s3_bucket_name=BUCKET,
        s3_url_ttl_seconds=3600,
    )


async def _ensure_bucket(settings: Settings) -> None:
    import aioboto3

    session = aioboto3.Session()
    async with session.client(
        "s3",
        endpoint_url=settings.aws_endpoint_url,
        aws_access_key_id=settings.aws_access_key_id.get_secret_value(),  # type: ignore[union-attr]
        aws_secret_access_key=settings.aws_secret_access_key.get_secret_value(),  # type: ignore[union-attr]
        region_name=settings.aws_region,
    ) as client:
        try:
            await client.create_bucket(Bucket=BUCKET)
        except client.exceptions.BucketAlreadyOwnedByYou:
            pass
        except client.exceptions.BucketAlreadyExists:
            pass


async def test_direct_upload_zero_backend_bandwidth(moto_server: ThreadedMotoServer) -> None:
    settings = _settings(moto_server)
    await _ensure_bucket(settings)
    obj = _Obj(bucket=BUCKET, object_key="materials/upload-flow.bin")
    payload = b"x" * (1024 * 64)

    url, expires_at = await create_upload_url(
        obj, content_type="application/octet-stream", settings=settings
    )

    parsed = urlparse(url)
    assert parsed.scheme == "http"
    assert parsed.hostname == moto_server._host  # type: ignore[attr-defined]
    assert parsed.port == moto_server._port  # type: ignore[attr-defined]
    assert "X-Amz-Signature" in parsed.query
    assert expires_at.tzinfo is not None

    async with httpx.AsyncClient() as http:
        put = await http.put(
            url, content=payload, headers={"Content-Type": "application/octet-stream"}
        )
    assert put.status_code in (200, 204), put.text

    meta = await head_object(obj, settings=settings)
    assert meta is not None
    assert meta.size == len(payload)


async def test_endpoint_split_public_vs_internal(moto_server: ThreadedMotoServer) -> None:
    public = _endpoint(moto_server)
    internal = public.replace("http://", "http://internal-")
    settings = _settings(
        moto_server,
        public_endpoint=public,
        internal_endpoint=internal,
    )

    bucket_settings = _settings(moto_server)
    await _ensure_bucket(bucket_settings)
    obj = _Obj(bucket=BUCKET, object_key="materials/endpoint-split.bin")

    upload_url, _ = await create_upload_url(obj, settings=settings)
    stream_url, _ = await create_stream_url(obj, settings=settings)

    upload_host = urlparse(upload_url).hostname or ""
    stream_host = urlparse(stream_url).hostname or ""
    assert upload_host.startswith(urlparse(public).hostname or "")
    assert stream_host.startswith(urlparse(public).hostname or "")
    assert "internal-" not in upload_host
    assert "internal-" not in stream_host


async def test_aws_default_endpoint_virtual_host_addressing() -> None:
    settings = _settings(None, public_endpoint=None, internal_endpoint=None)
    obj = _Obj(bucket=BUCKET, object_key="materials/aws-mode.bin")

    url, _ = await create_upload_url(obj, settings=settings)
    parsed = urlparse(url)
    assert parsed.hostname is not None
    assert "amazonaws.com" in parsed.hostname
    assert parsed.hostname.startswith(BUCKET) or BUCKET in parsed.hostname.split(".")[0]


async def test_head_verifies_upload_and_delete_clears(moto_server: ThreadedMotoServer) -> None:
    settings = _settings(moto_server)
    await _ensure_bucket(settings)
    obj = _Obj(bucket=BUCKET, object_key="materials/head-flow.bin")

    url, _ = await create_upload_url(obj, settings=settings)
    payload = b"hello-head"
    async with httpx.AsyncClient() as http:
        await http.put(url, content=payload)

    meta = await head_object(obj, settings=settings)
    assert meta is not None
    assert meta.size == len(payload)

    deleted = await delete_object(obj, settings=settings)
    assert deleted is True
    assert await head_object(obj, settings=settings) is None


async def test_multipart_lifecycle_complete_and_abort(moto_server: ThreadedMotoServer) -> None:
    settings = _settings(moto_server)
    await _ensure_bucket(settings)

    obj_complete = _Obj(bucket=BUCKET, object_key="materials/multipart-complete.bin")
    init = await create_multipart_upload(obj_complete, part_count=2, settings=settings)
    assert len(init.parts) == 2
    part_bytes = b"a" * (5 * 1024 * 1024)
    completed: list[CompletedPart] = []
    async with httpx.AsyncClient() as http:
        for part in init.parts:
            resp = await http.put(part.url, content=part_bytes)
            assert resp.status_code in (200, 204), resp.text
            etag = resp.headers["ETag"]
            completed.append(CompletedPart(part_number=part.part_number, etag=etag))
    await complete_multipart_upload(obj_complete, init.upload_id, completed, settings=settings)
    final = await head_object(obj_complete, settings=settings)
    assert final is not None
    assert final.size == len(part_bytes) * 2

    obj_abort = _Obj(bucket=BUCKET, object_key="materials/multipart-abort.bin")
    init_abort = await create_multipart_upload(obj_abort, part_count=1, settings=settings)
    await abort_multipart_upload(obj_abort, init_abort.upload_id, settings=settings)

    assert await head_object(obj_abort, settings=settings) is None


async def test_delete_object_idempotent(moto_server: ThreadedMotoServer) -> None:
    settings = _settings(moto_server)
    await _ensure_bucket(settings)
    obj = _Obj(bucket=BUCKET, object_key="materials/idempotent.bin")

    url, _ = await create_upload_url(obj, settings=settings)
    async with httpx.AsyncClient() as http:
        await http.put(url, content=b"some bytes")

    assert await delete_object(obj, settings=settings) is True
    assert await delete_object(obj, settings=settings) is True
    assert await delete_object(obj, settings=settings) is True


async def test_worker_download_uses_internal_endpoint(
    moto_server: ThreadedMotoServer,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(moto_server)
    await _ensure_bucket(settings)
    obj = _Obj(bucket=BUCKET, object_key="materials/worker.bin")

    upload_url, _ = await create_upload_url(obj, settings=settings)
    payload = b"worker-bytes-" * 1024
    async with httpx.AsyncClient() as http:
        await http.put(upload_url, content=payload)

    captured: dict[str, object] = {}
    real_kwargs = s3_module._client_kwargs

    def spy(s: Settings, endpoint: str | None) -> dict:
        captured["endpoint"] = endpoint
        return real_kwargs(s, endpoint)

    monkeypatch.setattr(s3_module, "_client_kwargs", spy)

    out = await download_to_temp(obj, tmp_path, settings=settings)
    assert out.exists()
    assert out.read_bytes() == payload
    assert captured["endpoint"] == settings.aws_endpoint_url


async def test_no_creds_raises_s3_not_configured() -> None:
    settings = _settings(None, creds=False)
    obj = _Obj(bucket=BUCKET, object_key="materials/never.bin")

    with pytest.raises(S3NotConfiguredError):
        await create_upload_url(obj, settings=settings)
    with pytest.raises(S3NotConfiguredError):
        await create_stream_url(obj, settings=settings)
    with pytest.raises(S3NotConfiguredError):
        await head_object(obj, settings=settings)
    with pytest.raises(S3NotConfiguredError):
        await delete_object(obj, settings=settings)
    with pytest.raises(S3NotConfiguredError):
        await create_multipart_upload(obj, settings=settings)


async def test_server_side_put_stores_bytes_and_content_type(
    moto_server: ThreadedMotoServer,
) -> None:
    """The one upload path that does NOT go through a presigned URL.

    Avatars and thumbnails are small and server-validated, so the backend
    writes them directly rather than issuing a URL and waiting for the
    browser. It is the only writer that must use the INTERNAL endpoint: the
    bytes never leave the cluster.
    """
    settings = _settings(moto_server)
    await _ensure_bucket(settings)
    obj = _Obj(bucket=BUCKET, object_key="avatars/server-side.png")

    await put_object_bytes(obj, b"PNG-bytes", content_type="image/png", settings=settings)

    meta = await head_object(obj, settings=settings)
    assert meta is not None
    assert meta.size == len(b"PNG-bytes")
    assert meta.content_type == "image/png"


async def test_server_side_put_uses_the_internal_endpoint(
    moto_server: ThreadedMotoServer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    public = _endpoint(moto_server)
    settings = _settings(
        moto_server,
        public_endpoint=public.replace("http://", "http://public-"),
        internal_endpoint=public,
    )
    await _ensure_bucket(_settings(moto_server))

    captured: dict[str, object] = {}
    real_kwargs = s3_module._client_kwargs

    def spy(s: Settings, endpoint: str | None) -> dict:
        captured["endpoint"] = endpoint
        return real_kwargs(s, endpoint)

    monkeypatch.setattr(s3_module, "_client_kwargs", spy)

    await put_object_bytes(
        _Obj(bucket=BUCKET, object_key="avatars/internal.png"), b"x", settings=settings
    )
    assert captured["endpoint"] == settings.aws_endpoint_url


async def test_stream_url_forwards_only_the_three_supported_headers(
    moto_server: ThreadedMotoServer,
) -> None:
    """These control how the browser treats the streamed object.

    ``Content-Disposition`` is what makes a material download under its real
    filename instead of opening inline under an opaque object key, so it is
    signed into the URL rather than sent as a request header (a presigned GET
    is opened by the browser, which sends no headers of ours). Anything
    outside the supported three is dropped: an unsigned response header S3
    does not recognise invalidates the signature.
    """
    settings = _settings(moto_server)
    await _ensure_bucket(settings)
    obj = _Obj(bucket=BUCKET, object_key="materials/report.pdf")

    url, _ = await create_stream_url(
        obj,
        response_headers={
            "Content-Disposition": 'attachment; filename="Week 1.pdf"',
            "Content-Type": "application/pdf",
            "Cache-Control": "private, max-age=60",
            "X-Unsupported": "dropped",
        },
        settings=settings,
    )

    query = parse_qs(urlparse(url).query)
    assert query["response-content-disposition"] == ['attachment; filename="Week 1.pdf"']
    assert query["response-content-type"] == ["application/pdf"]
    assert query["response-cache-control"] == ["private, max-age=60"]
    assert not [key for key in query if "unsupported" in key.lower()]


async def test_stream_url_without_headers_signs_no_response_overrides(
    moto_server: ThreadedMotoServer,
) -> None:
    settings = _settings(moto_server)
    await _ensure_bucket(settings)

    url, _ = await create_stream_url(
        _Obj(bucket=BUCKET, object_key="materials/plain.bin"), settings=settings
    )

    query = parse_qs(urlparse(url).query)
    assert not [key for key in query if key.startswith("response-")]


async def test_download_of_a_missing_object_is_not_found_not_a_generic_failure(
    moto_server: ThreadedMotoServer,
    tmp_path: Path,
) -> None:
    """A worker distinguishes "the file is gone" from "S3 is unwell".

    The first means the job can never succeed and should be abandoned; the
    second is worth retrying. Collapsing them into one error would have the
    worker retry a deleted material until it exhausted its attempts.
    """
    settings = _settings(moto_server)
    await _ensure_bucket(settings)

    with pytest.raises(S3NotFoundError):
        await download_to_temp(
            _Obj(bucket=BUCKET, object_key="materials/never-uploaded.bin"),
            tmp_path,
            settings=settings,
        )


async def test_download_creates_the_destination_directory(
    moto_server: ThreadedMotoServer,
    tmp_path: Path,
) -> None:
    """Workers hand in a per-job temp directory that does not exist yet."""
    settings = _settings(moto_server)
    await _ensure_bucket(settings)
    obj = _Obj(bucket=BUCKET, object_key="materials/nested-dest.bin")
    await put_object_bytes(obj, b"payload", settings=settings)

    dest = tmp_path / "job" / "artifacts"
    out = await download_to_temp(obj, dest, settings=settings)

    assert out.parent == dest
    assert out.name == "nested-dest.bin", "the object key's basename, not the whole key"
    assert out.read_bytes() == b"payload"


async def test_multipart_argument_guards_refuse_before_touching_s3() -> None:
    """Both are caller mistakes, and both are cheaper to catch locally.

    A zero-part upload would initialise a multipart upload and presign
    nothing, leaving an orphaned upload S3 bills for until a lifecycle rule
    reaps it. An empty completion would close an upload with no parts.
    """
    settings = _settings(None, creds=False)
    obj = _Obj(bucket=BUCKET, object_key="materials/guard.bin")

    with pytest.raises(ValueError, match="part_count"):
        await create_multipart_upload(obj, part_count=0, settings=settings)
    with pytest.raises(ValueError, match="at least one part"):
        await complete_multipart_upload(obj, "upload-id", [], settings=settings)


async def test_abort_of_an_unknown_upload_is_silent(
    moto_server: ThreadedMotoServer,
) -> None:
    """Abort is the cleanup path, often reached from an error handler.

    Raising here would mask the failure that prompted the cleanup with a
    second failure about the cleanup itself.
    """
    settings = _settings(moto_server)
    await _ensure_bucket(settings)

    await abort_multipart_upload(
        _Obj(bucket=BUCKET, object_key="materials/no-such-upload.bin"),
        "not-a-real-upload-id",
        settings=settings,
    )


async def test_put_object_bytes_refuses_without_credentials() -> None:
    settings = _settings(None, creds=False)
    with pytest.raises(S3NotConfiguredError):
        await put_object_bytes(
            _Obj(bucket=BUCKET, object_key="materials/never.bin"), b"x", settings=settings
        )
