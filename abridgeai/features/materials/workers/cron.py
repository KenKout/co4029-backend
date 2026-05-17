"""Cron task — abort orphaned multipart uploads (T4.5 §4947 / §4955).

Multipart uploads that the client never finalised cost ongoing storage
in S3 / Garage. AWS treats incomplete multipart parts as billable
storage; Garage does not, but the parts still occupy disk on the
cluster nodes. Hourly (configured in :mod:`abridgeai.workers.arq_app`)
this task lists every multipart upload older than the configured TTL
and calls ``abort_multipart_upload`` to free the bytes.

Storage strategy decision
-------------------------
T4.5 considered three options for tracking orphan uploads:

1. Add ``pending_multipart_upload_id`` to ``learning_material_versions``
   via a new Alembic migration. Cost: schema churn for an
   operational concern.
2. Stash ``upload_id`` in Redis with TTL. Cost: extra storage layer
   to maintain; lost on Redis flush.
3. **Query S3 directly** via ``ListMultipartUploads``. Cost: one S3
   API call per cron run; no schema, no extra state.

We picked (3) — S3 already knows about every in-flight multipart
upload it issued an upload_id for. Calling ``ListMultipartUploads``
returns every upload along with its ``Initiated`` timestamp; rows
older than the TTL are aborted. T2.3 did not expose a helper for
``ListMultipartUploads`` so this task uses ``aioboto3`` directly —
documented as a follow-up candidate to fold into ``infrastructure/s3.py``.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import aioboto3
from botocore.exceptions import ClientError

from abridgeai.core.config import get_settings
from abridgeai.core.observability import (
    bind_request_context,
    clear_request_context,
    get_logger,
)
from abridgeai.infrastructure.errors import S3NotConfiguredError

_logger = get_logger(__name__)

_DEFAULT_ORPHAN_TTL_HOURS: int = 24


async def cleanup_orphaned_uploads_task(ctx: dict[str, Any]) -> None:
    """Abort multipart uploads older than 24h that never reached complete.

    Listed parts older than the TTL are aborted via
    :func:`abridgeai.infrastructure.s3.abort_multipart_upload`. Errors
    on individual aborts are logged and skipped — one stuck upload
    must not stop the rest of the run from being cleaned up.

    Reads :attr:`Settings.s3_bucket_name` for the target bucket.
    Surfaces nothing on the return path (cron tasks are fire-and-forget);
    counts and timing are emitted via structured logs.
    """
    _ = ctx
    bind_request_context(task="cleanup_orphaned_uploads")
    try:
        await _run_cleanup(ttl_hours=_DEFAULT_ORPHAN_TTL_HOURS)
    finally:
        clear_request_context()


async def _run_cleanup(*, ttl_hours: int) -> None:
    settings = get_settings()
    access_key = settings.aws_access_key_id
    secret_key = settings.aws_secret_access_key
    if access_key is None or secret_key is None:
        raise S3NotConfiguredError("S3 credentials are not configured for orphan cleanup")

    cutoff = datetime.now(tz=UTC) - timedelta(hours=ttl_hours)
    bucket = settings.s3_bucket_name
    endpoint = settings.aws_endpoint_url

    aborted = 0
    skipped = 0
    listed = 0

    session = aioboto3.Session()
    async with session.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key.get_secret_value(),
        aws_secret_access_key=secret_key.get_secret_value(),
        region_name=settings.aws_region,
    ) as client:
        paginator = client.get_paginator("list_multipart_uploads")
        async for page in paginator.paginate(Bucket=bucket):
            for upload in page.get("Uploads", []) or []:
                listed += 1
                initiated = upload.get("Initiated")
                if initiated is None:
                    continue
                if initiated.tzinfo is None:
                    initiated = initiated.replace(tzinfo=UTC)
                if initiated > cutoff:
                    skipped += 1
                    continue
                try:
                    await client.abort_multipart_upload(
                        Bucket=bucket,
                        Key=upload["Key"],
                        UploadId=upload["UploadId"],
                    )
                    aborted += 1
                except ClientError as exc:
                    _logger.warning(
                        "orphan_upload_abort_failed",
                        key=upload.get("Key"),
                        upload_id=upload.get("UploadId"),
                        error=str(exc),
                    )

    _logger.info(
        "orphan_uploads_cleanup_completed",
        listed=listed,
        aborted=aborted,
        skipped=skipped,
        bucket=bucket,
        ttl_hours=ttl_hours,
    )


__all__ = ["cleanup_orphaned_uploads_task"]
