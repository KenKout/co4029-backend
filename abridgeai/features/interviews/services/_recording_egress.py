"""LiveKit Egress seam for interview audio recording.

Everything that talks to the LiveKit Egress API — start, stop, inspect, and
the reported-location → S3-key normalization — lives here so
``services/recording.py`` keeps the orchestration/policy side readable and
stays under the 800-LOC cap (``tests/integration/test_interviews_metric.py``).

Pure provider seam: no DB access, no state. Every function raises on failure
(``_stop_egress_quietly`` / ``_fetch_egress_info`` convert provider errors
into ``False``/``(False, None)`` so reconciliation can retry) — the callers in
``services/recording.py`` own the never-break-the-interview semantics.

The names are re-exported from ``services/recording.py``; tests patch them
there and production call sites resolve them through that module's namespace.
"""

from __future__ import annotations

import logging
from urllib.parse import urlsplit

from livekit import api as lk_api
from livekit.api.egress_service import EgressInfo  # type: ignore[attr-defined]

from abridgeai.core.config import Settings

logger = logging.getLogger(__name__)


class RecordingUnavailableError(Exception):
    """Raised (internally) when recording cannot start — never escapes the module."""


def recording_bucket(settings: Settings) -> str:
    """Bucket recordings land in — the same S3/Garage bucket, recording prefix."""
    return settings.s3_bucket_name


async def _start_audio_egress(*, room_name: str, destination: str, settings: Settings) -> str:
    """Start a room-composite AUDIO-ONLY Egress to the private prefix.

    Raises on any provider error; the caller owns the failure semantics.
    """
    if not (settings.livekit_ws_url and settings.livekit_api_key and settings.livekit_api_secret):
        raise RecordingUnavailableError("LiveKit credentials are not configured")
    bucket = recording_bucket(settings)
    lkapi = lk_api.LiveKitAPI(
        url=settings.livekit_ws_url,
        api_key=settings.livekit_api_key.get_secret_value(),
        api_secret=settings.livekit_api_secret.get_secret_value(),
    )
    try:
        # RoomCompositeEgress REJECTS a room that has never been created
        # (twirp 404 "requested room does not exist"). The recording starts
        # with the token mint — BEFORE any client has joined — so create the
        # room here (idempotent) with a generous empty timeout; the egress
        # keeps the room alive as a hidden participant until recording ends.
        await lkapi.room.create_room(
            lk_api.CreateRoomRequest(
                name=room_name,
                empty_timeout=settings.interview_recording_empty_timeout_seconds,
            )
        )
        request = lk_api.RoomCompositeEgressRequest(
            room_name=room_name,
            audio_only=True,
            file_outputs=[
                lk_api.EncodedFileOutput(
                    filepath=destination,
                    file_type=lk_api.EncodedFileType.MP3
                    if settings.interview_recording_format == "mp3"
                    else lk_api.EncodedFileType.OGG,
                    s3=lk_api.S3Upload(
                        bucket=bucket,
                        region=settings.aws_region,
                        endpoint=settings.aws_endpoint_url,
                        access_key=settings.aws_access_key_id.get_secret_value()
                        if settings.aws_access_key_id
                        else "",
                        secret=settings.aws_secret_access_key.get_secret_value()
                        if settings.aws_secret_access_key
                        else "",
                        force_path_style=bool(settings.aws_endpoint_url),
                    ),
                )
            ],
        )
        info = await lkapi.egress.start_room_composite_egress(request)
    finally:
        await lkapi.aclose()
    egress_id = info.egress_id
    if not egress_id:
        raise RecordingUnavailableError("LiveKit returned no egress id")
    return egress_id


async def _stop_egress_quietly(egress_id: str, settings: Settings) -> bool:
    """Stop an Egress job; return False so reconciliation can retry failures."""
    if not (
        settings.livekit_ws_url
        and settings.livekit_api_key
        and settings.livekit_api_secret
    ):
        return False
    lkapi = lk_api.LiveKitAPI(
        url=settings.livekit_ws_url,
        api_key=settings.livekit_api_key.get_secret_value(),
        api_secret=settings.livekit_api_secret.get_secret_value(),
    )
    try:
        await lkapi.egress.stop_egress(lk_api.StopEgressRequest(egress_id=egress_id))
    except Exception:  # noqa: BLE001 -- reconciliation retries the stop
        logger.info("interview.recording.stop_egress_deferred", extra={"egress_id": egress_id})
        return False
    finally:
        await lkapi.aclose()
    return True


async def _fetch_egress_info(
    egress_id: str, settings: Settings
) -> tuple[bool, EgressInfo | None]:
    """Return ``(provider_reachable, egress_info_or_none)``."""
    if not (
        settings.livekit_ws_url
        and settings.livekit_api_key
        and settings.livekit_api_secret
    ):
        return False, None
    lkapi = lk_api.LiveKitAPI(
        url=settings.livekit_ws_url,
        api_key=settings.livekit_api_key.get_secret_value(),
        api_secret=settings.livekit_api_secret.get_secret_value(),
    )
    try:
        response = await lkapi.egress.list_egress(
            lk_api.ListEgressRequest(egress_id=egress_id)
        )
    except Exception:  # noqa: BLE001 -- provider down: unknown, not dead
        return False, None
    finally:
        await lkapi.aclose()
    for item in response.items:
        if item.egress_id == egress_id:
            return True, item
    return True, None


def _normalize_reported_key(reported_location: str, *, expected_bucket: str) -> str:
    """Convert LiveKit's filename/location into an S3 key and reject foreign URI hosts.

    LiveKit versions have returned a bare key, ``s3://bucket/key``, or a full
    HTTP URL (the S3 endpoint + bucket + key — observed with the Garage
    endpoint on egress v1.13). Handle all three: the URL form is only
    accepted when its host matches the configured S3 endpoint host AND its
    first path segment is the expected bucket, so a foreign bucket is still
    rejected; the key is what remains of the path after the bucket.
    """
    value = reported_location.strip()
    if value.startswith("s3://"):
        parsed = urlsplit(value)
        if parsed.netloc != expected_bucket:
            return ""
        return parsed.path.lstrip("/")
    bucket_prefix = expected_bucket.strip("/") + "/"
    if value.startswith(bucket_prefix):
        return value[len(bucket_prefix) :]
    if "://" in value:
        parsed = urlsplit(value)
        segments = [seg for seg in parsed.path.split("/") if seg]
        endpoint_host = ""
        from abridgeai.core.config import get_settings

        endpoint = get_settings().aws_endpoint_url or ""
        if endpoint:
            endpoint_host = urlsplit(endpoint).netloc
        host_ok = bool(endpoint_host) and parsed.netloc == endpoint_host
        bucket_ok = bool(segments) and segments[0] == expected_bucket
        if host_ok and bucket_ok:
            return "/".join(segments[1:])
        return ""
    return value.lstrip("/")


__all__ = ["RecordingUnavailableError", "recording_bucket"]
