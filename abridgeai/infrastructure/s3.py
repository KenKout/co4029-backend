"""Async S3 / Garage object-storage helpers (T2.3).

Architectural intent (preserved from legacy ``backend/app/integrations/s3.py``):

* **Direct upload**: backend issues a presigned ``PUT`` URL; the browser
  uploads bytes straight to S3. Backend never proxies upload traffic.
* **Direct streaming**: backend issues a presigned ``GET`` URL; the browser
  streams straight from S3.
* **Worker downloads**: ARQ workers call ``download_to_temp`` to fetch
  source files from the *internal* endpoint (low-latency, in-cluster).

Endpoint resolution
-------------------

* ``_browser_endpoint`` = ``aws_public_endpoint_url`` || ``aws_endpoint_url``
  || ``None`` — used for presigned URLs (must be reachable from the user's
  browser).
* ``_internal_endpoint`` = ``aws_endpoint_url`` || ``None`` — used for
  server-side calls (HEAD, multipart admin, downloads).
* When the resolved endpoint is ``None`` (production AWS) ``aioboto3`` falls
  back to the regional AWS endpoint and uses **virtual-host** addressing.
  When it is set (Garage) we force **path-style** addressing.

Client lifecycle
----------------

Every call opens a fresh ``aioboto3.Session().client("s3")`` async context
manager. ``boto3``'s underlying HTTP pool keeps the cost cheap and avoids
the cross-event-loop state issues a singleton client would introduce.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

import aioboto3
from botocore.config import Config
from botocore.exceptions import ClientError

from abridgeai.core.config import Settings, get_settings
from abridgeai.infrastructure.errors import (
    S3NotConfiguredError,
    S3NotFoundError,
    S3PermissionError,
    S3UploadError,
)


class StorageObject(Protocol):
    """Minimal duck-typed view of an ORM ``StorageObject`` row.

    Production callers pass the SQLAlchemy model from ``features.identity``
    (T1.x); tests pass a dataclass. Only ``bucket`` and ``object_key`` are
    consumed here.
    """

    bucket: str
    object_key: str


@dataclass(frozen=True)
class ObjectMetadata:
    size: int
    content_type: str | None
    etag: str | None


@dataclass(frozen=True)
class PresignedPart:
    part_number: int
    url: str


@dataclass(frozen=True)
class MultipartInit:
    upload_id: str
    parts: list[PresignedPart]


@dataclass(frozen=True)
class CompletedPart:
    part_number: int
    etag: str


_DEFAULT_PART_COUNT = 1
_NOT_FOUND_CODES: frozenset[str] = frozenset({"404", "NoSuchKey", "NotFound"})
_FORBIDDEN_CODES: frozenset[str] = frozenset({"403", "AccessDenied", "Forbidden"})


def _utcnow() -> datetime:
    return datetime.now(tz=UTC)


def _s3_enabled(settings: Settings) -> bool:
    return bool(settings.aws_access_key_id and settings.aws_secret_access_key)


def _ensure_configured(settings: Settings) -> None:
    if not _s3_enabled(settings):
        raise S3NotConfiguredError(
            "S3 / Garage is not configured: set AWS_ACCESS_KEY_ID and "
            "AWS_SECRET_ACCESS_KEY (and optionally AWS_ENDPOINT_URL for "
            "Garage local dev)."
        )


def _browser_endpoint(settings: Settings) -> str | None:
    return settings.aws_public_endpoint_url or settings.aws_endpoint_url or None


def _internal_endpoint(settings: Settings) -> str | None:
    return settings.aws_endpoint_url or None


def _addressing_style(endpoint: str | None) -> str:
    return "path" if endpoint else "virtual"


def _client_config(settings: Settings, endpoint: str | None) -> Config:
    return Config(
        signature_version="s3v4",
        s3={"addressing_style": _addressing_style(endpoint)},
        region_name=settings.aws_region,
    )


def _session() -> aioboto3.Session:
    return aioboto3.Session()


def _client_kwargs(settings: Settings, endpoint: str | None) -> dict[str, Any]:
    access_key = settings.aws_access_key_id
    secret_key = settings.aws_secret_access_key
    return {
        "endpoint_url": endpoint,
        "aws_access_key_id": access_key.get_secret_value() if access_key else None,
        "aws_secret_access_key": secret_key.get_secret_value() if secret_key else None,
        "region_name": settings.aws_region,
        "config": _client_config(settings, endpoint),
    }


def _error_code(exc: ClientError) -> str:
    return str(exc.response.get("Error", {}).get("Code", ""))


def _http_status(exc: ClientError) -> int:
    return int(exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0))


def _translate_client_error(exc: ClientError, *, action: str) -> S3UploadError:
    code = _error_code(exc)
    status = _http_status(exc)
    if code in _NOT_FOUND_CODES or status == 404:
        raise S3NotFoundError(f"{action}: object not found ({code or status})") from exc
    if code in _FORBIDDEN_CODES or status == 403:
        raise S3PermissionError(f"{action}: access denied ({code or status})") from exc
    return S3UploadError(f"{action}: {code or status or exc!s}")


async def create_upload_url(
    storage_object: StorageObject,
    content_type: str | None = None,
    *,
    settings: Settings | None = None,
) -> tuple[str, datetime]:
    """Presigned PUT URL for direct browser → S3 single-shot upload."""

    settings = settings or get_settings()
    _ensure_configured(settings)
    endpoint = _browser_endpoint(settings)
    expires_in = settings.s3_url_ttl_seconds
    expires_at = _utcnow() + timedelta(seconds=expires_in)

    params: dict[str, Any] = {
        "Bucket": storage_object.bucket,
        "Key": storage_object.object_key,
    }
    if content_type:
        params["ContentType"] = content_type

    async with _session().client("s3", **_client_kwargs(settings, endpoint)) as client:
        try:
            url = await client.generate_presigned_url(
                ClientMethod="put_object",
                Params=params,
                ExpiresIn=expires_in,
            )
        except ClientError as exc:
            raise _translate_client_error(exc, action="create_upload_url") from exc
    return url, expires_at


async def create_stream_url(
    storage_object: StorageObject,
    *,
    response_headers: dict[str, str] | None = None,
    settings: Settings | None = None,
) -> tuple[str, datetime]:
    """Presigned GET URL for direct browser → S3 streaming."""

    settings = settings or get_settings()
    _ensure_configured(settings)
    endpoint = _browser_endpoint(settings)
    expires_in = settings.s3_url_ttl_seconds
    expires_at = _utcnow() + timedelta(seconds=expires_in)

    params: dict[str, Any] = {
        "Bucket": storage_object.bucket,
        "Key": storage_object.object_key,
    }
    if response_headers:
        if "Content-Disposition" in response_headers:
            params["ResponseContentDisposition"] = response_headers["Content-Disposition"]
        if "Content-Type" in response_headers:
            params["ResponseContentType"] = response_headers["Content-Type"]

    async with _session().client("s3", **_client_kwargs(settings, endpoint)) as client:
        try:
            url = await client.generate_presigned_url(
                ClientMethod="get_object",
                Params=params,
                ExpiresIn=expires_in,
            )
        except ClientError as exc:
            raise _translate_client_error(exc, action="create_stream_url") from exc
    return url, expires_at


async def head_object(
    storage_object: StorageObject,
    *,
    settings: Settings | None = None,
) -> ObjectMetadata | None:
    """Verify the object exists and fetch size/content-type/etag.

    Returns ``None`` for missing objects so the ``/complete`` endpoint can
    distinguish "client never finished the upload" from a real error.
    """

    settings = settings or get_settings()
    _ensure_configured(settings)
    endpoint = _internal_endpoint(settings)

    async with _session().client("s3", **_client_kwargs(settings, endpoint)) as client:
        try:
            response = await client.head_object(
                Bucket=storage_object.bucket,
                Key=storage_object.object_key,
            )
        except ClientError as exc:
            code = _error_code(exc)
            status = _http_status(exc)
            if code in _NOT_FOUND_CODES or status == 404:
                return None
            if code in _FORBIDDEN_CODES or status == 403:
                raise S3PermissionError(f"head_object: access denied ({code})") from exc
            raise S3UploadError(f"head_object: {code or status or exc!s}") from exc

    return ObjectMetadata(
        size=int(response.get("ContentLength", 0)),
        content_type=response.get("ContentType"),
        etag=response.get("ETag"),
    )


async def delete_object(
    storage_object: StorageObject,
    *,
    settings: Settings | None = None,
) -> bool:
    """Permanent delete from S3. Idempotent: returns ``True`` even if absent."""

    settings = settings or get_settings()
    _ensure_configured(settings)
    endpoint = _internal_endpoint(settings)

    async with _session().client("s3", **_client_kwargs(settings, endpoint)) as client:
        try:
            await client.delete_object(
                Bucket=storage_object.bucket,
                Key=storage_object.object_key,
            )
        except ClientError as exc:
            code = _error_code(exc)
            status = _http_status(exc)
            if code in _NOT_FOUND_CODES or status == 404:
                return True
            if code in _FORBIDDEN_CODES or status == 403:
                raise S3PermissionError(f"delete_object: access denied ({code})") from exc
            raise S3UploadError(f"delete_object: {code or status or exc!s}") from exc
    return True


async def create_multipart_upload(
    storage_object: StorageObject,
    *,
    part_count: int = _DEFAULT_PART_COUNT,
    content_type: str | None = None,
    settings: Settings | None = None,
) -> MultipartInit:
    """Initialise a multipart upload and presign ``part_count`` PUT URLs."""

    if part_count < 1:
        raise ValueError("part_count must be >= 1")
    settings = settings or get_settings()
    _ensure_configured(settings)
    browser_endpoint = _browser_endpoint(settings)
    internal_endpoint = _internal_endpoint(settings)
    expires_in = settings.s3_url_ttl_seconds

    init_kwargs: dict[str, Any] = {
        "Bucket": storage_object.bucket,
        "Key": storage_object.object_key,
    }
    if content_type:
        init_kwargs["ContentType"] = content_type

    async with _session().client(
        "s3", **_client_kwargs(settings, internal_endpoint)
    ) as admin_client:
        try:
            init = await admin_client.create_multipart_upload(**init_kwargs)
        except ClientError as exc:
            raise _translate_client_error(exc, action="create_multipart_upload") from exc
        upload_id = init["UploadId"]

    async with _session().client(
        "s3", **_client_kwargs(settings, browser_endpoint)
    ) as browser_client:
        parts: list[PresignedPart] = []
        for part_number in range(1, part_count + 1):
            try:
                url = await browser_client.generate_presigned_url(
                    ClientMethod="upload_part",
                    Params={
                        "Bucket": storage_object.bucket,
                        "Key": storage_object.object_key,
                        "UploadId": upload_id,
                        "PartNumber": part_number,
                    },
                    ExpiresIn=expires_in,
                )
            except ClientError as exc:
                raise _translate_client_error(
                    exc, action="create_multipart_upload.presign_part"
                ) from exc
            parts.append(PresignedPart(part_number=part_number, url=url))

    return MultipartInit(upload_id=upload_id, parts=parts)


async def complete_multipart_upload(
    storage_object: StorageObject,
    upload_id: str,
    parts: list[CompletedPart],
    *,
    settings: Settings | None = None,
) -> None:
    """Finalise a multipart upload with the parts the client uploaded."""

    if not parts:
        raise ValueError("complete_multipart_upload requires at least one part")
    settings = settings or get_settings()
    _ensure_configured(settings)
    endpoint = _internal_endpoint(settings)

    async with _session().client("s3", **_client_kwargs(settings, endpoint)) as client:
        try:
            await client.complete_multipart_upload(
                Bucket=storage_object.bucket,
                Key=storage_object.object_key,
                UploadId=upload_id,
                MultipartUpload={
                    "Parts": [{"PartNumber": p.part_number, "ETag": p.etag} for p in parts],
                },
            )
        except ClientError as exc:
            raise _translate_client_error(exc, action="complete_multipart_upload") from exc


async def abort_multipart_upload(
    storage_object: StorageObject,
    upload_id: str,
    *,
    settings: Settings | None = None,
) -> None:
    """Cancel a multipart upload and free orphaned parts."""

    settings = settings or get_settings()
    _ensure_configured(settings)
    endpoint = _internal_endpoint(settings)

    async with _session().client("s3", **_client_kwargs(settings, endpoint)) as client:
        try:
            await client.abort_multipart_upload(
                Bucket=storage_object.bucket,
                Key=storage_object.object_key,
                UploadId=upload_id,
            )
        except ClientError as exc:
            code = _error_code(exc)
            status = _http_status(exc)
            if code in _NOT_FOUND_CODES or status == 404:
                return
            raise _translate_client_error(exc, action="abort_multipart_upload") from exc


async def download_to_temp(
    storage_object: StorageObject,
    dest_dir: Path,
    *,
    settings: Settings | None = None,
) -> Path:
    """Worker helper: download the object to ``dest_dir``, return the path.

    Uses the *internal* endpoint so worker → S3 traffic stays in-cluster.
    Caller owns cleanup of the destination file.
    """

    settings = settings or get_settings()
    _ensure_configured(settings)
    endpoint = _internal_endpoint(settings)
    dest_dir.mkdir(parents=True, exist_ok=True)  # noqa: ASYNC240  # short blocking call; aioboto3.download_file itself wraps sync I/O
    dest_path = dest_dir / Path(storage_object.object_key).name

    async with _session().client("s3", **_client_kwargs(settings, endpoint)) as client:
        try:
            await client.download_file(
                Bucket=storage_object.bucket,
                Key=storage_object.object_key,
                Filename=str(dest_path),
            )
        except ClientError as exc:
            raise _translate_client_error(exc, action="download_to_temp") from exc
    return dest_path


__all__ = [
    "CompletedPart",
    "MultipartInit",
    "ObjectMetadata",
    "PresignedPart",
    "S3NotConfiguredError",
    "StorageObject",
    "abort_multipart_upload",
    "complete_multipart_upload",
    "create_multipart_upload",
    "create_stream_url",
    "create_upload_url",
    "delete_object",
    "download_to_temp",
    "head_object",
]
