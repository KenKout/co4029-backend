"""Interview audio recording — start, stop, validate, attach, reconcile.

The orchestration seam between the LiveKit Egress provider and the interview
feature. Mirrors the organisation of ``services/real_time.py``: everything
provider-shaped (Egress API calls, webhook signature verification) lives here,
routes and workers stay thin, and the DB writes go through
``queries/recordings.py`` conditional updates.

Contract the whole module defends:

* **Recording never breaks the interview.** Every provider interaction is
  best-effort from the interview's perspective — a failed Egress start, a
  dead webhook, a reconciliation miss leaves a dead recording row and a log,
  never a failed session. Conversely nothing here ever raises out of
  :func:`start_recording_for_session` / :func:`stop_recording_for_session`.
* **No consent, no recording.** A row is claimed only with the candidate's
  affirmative consent (policy version + timestamp recorded on the row), and
  the claim happens only when the feature flag is on.
* **Provider truth, validated.** The canonical ``storage_objects`` row is
  derived from what LiveKit reports it wrote (bucket + key, validated against
  the expected prefix, ``head_object``-checked for size/MIME), never from
  what we asked for.

Webhook signature verification uses LiveKit's ``WebhookReceiver`` with the
same API key/secret that mint participant tokens — the deployment has one
LiveKit credential pair.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

from livekit import api as lk_api
from livekit.api.egress_service import EgressInfo
from livekit.api.webhook import WebhookEvent

from abridgeai.core.config import Settings, get_settings
from abridgeai.core.db import AsyncSession  # type: ignore[attr-defined]
from abridgeai.features.interviews.models import InterviewSession
from abridgeai.features.interviews.queries import recordings as recordings_queries
from abridgeai.features.interviews.queries import sessions as sessions_queries
from abridgeai.features.interviews.services._recording_egress import (
    RecordingUnavailableError,
    _fetch_egress_info,
    _normalize_reported_key,
    _start_audio_egress,
    _stop_egress_quietly,
    recording_bucket,
)
from abridgeai.infrastructure import s3

logger = logging.getLogger(__name__)

# Callback events this module consumes. LiveKit delivers several event types;
# anything unlisted is acknowledged (200) and ignored.
_EGRESS_EVENTS = frozenset({"egress_started", "egress_updated", "egress_ended", "egress_failed"})

# Allowed audio MIME types for the Egress output (mirrors the configured
# container format). Anything else on the object is treated as provider error
# and never attached for playback.
_ALLOWED_MIME_BY_FORMAT: dict[str, tuple[str, ...]] = {
    "mp3": ("audio/mpeg", "audio/mp3"),
    "ogg": ("audio/ogg", "application/ogg"),
}
_EXTENSION_BY_FORMAT = {"mp3": "mp3", "ogg": "ogg"}
# A zero-byte or absurdly small object is an Egress failure, not a recording.
_MIN_AUDIO_BYTES = 1024




def build_destination_file(
    *,
    session_id: UUID,
    recording_id: UUID,
    fmt: str,
    settings: Settings,
) -> str:
    """Deterministic Egress output key under the private recording prefix.

    ``<prefix>/<session>/<recording>/audio.<ext>`` — the recording id keeps the
    path stable even if a session's Egress is ever restarted into a second
    file, and the prefix is what both the bucket policy and the object
    validation key on.
    """
    ext = _EXTENSION_BY_FORMAT.get(fmt, "mp3")
    prefix = settings.interview_recording_s3_prefix.strip("/")
    return f"{prefix}/{session_id}/{recording_id}/audio.{ext}"




async def persist_recording_consent(
    db: AsyncSession,
    *,
    session: InterviewSession,
    accepted: bool,
    policy_version: str | None,
    room_name: str,
    settings: Settings | None = None,
) -> bool:
    """Persist a learner decision and claim its one recording row if eligible."""
    settings = settings or get_settings()
    consented = accepted and policy_version == settings.interview_recording_policy_version
    if session.recording_consent_status is None or consented:
        consented_at = datetime.now(tz=UTC)
        session.recording_consent_status = "accepted" if consented else "declined"
        session.recording_consent_policy_version = policy_version
        session.recording_consented_at = consented_at
        session.recording_consent_scope = "audio_only"
        if consented and settings.interview_recording_enabled:
            await recordings_queries.claim_recording(
                db,
                session_id=session.id,
                room_name=room_name,
                consent_policy_version=settings.interview_recording_policy_version,
                consented_at=consented_at,
            )
    return consented


async def start_recording_for_session(
    db: AsyncSession,
    *,
    session_id: UUID,
    student_id: UUID,
    room_name: str,
    consented: bool,
    settings: Settings | None = None,
) -> None:
    """Claim + start the session's recording after the realtime room exists.

    Called from the realtime-token flow once ``room_name`` is deterministic.
    All failure paths are swallowed INTO the recording row + logs; the
    interview continues regardless (flag off / no consent / provider error).

    Consent is an explicit boolean the caller derives from the candidate's
    stored acceptance (DTO ``interview-recording-consent``), matched against
    the CURRENT policy version. Without it this is a no-op — LiveKit is never
    called, and no row is created.
    """
    settings = settings or get_settings()
    if not settings.interview_recording_enabled or not consented:
        return

    now = datetime.now(tz=UTC)
    recording_id = await recordings_queries.claim_recording(
        db,
        session_id=session_id,
        room_name=room_name,
        consent_policy_version=settings.interview_recording_policy_version,
        consented_at=now,
    )
    if recording_id is None:
        return
    # Consent endpoint creates the pending row before the room is opened. A
    # later token mint must start that row; a retry after the provider already
    # accepted Egress must not start a second job.
    recording = await recordings_queries.get_recording_for_session(db, session_id)
    if recording is None or recording.status != "pending" or recording.egress_id:
        return
    if recording.consent_policy_version != settings.interview_recording_policy_version:
        logger.warning(
            "interview.recording.stale_consent",
            extra={"session_id": str(session_id)},
        )
        return

    destination = build_destination_file(
        session_id=session_id,
        recording_id=recording_id,
        fmt=settings.interview_recording_format,
        settings=settings,
    )
    try:
        egress_id = await _start_audio_egress(
            room_name=room_name,
            destination=destination,
            settings=settings,
        )
    except Exception as exc:  # noqa: BLE001 -- recording must never fail the interview
        await recordings_queries.mark_recording_failed(
            db, session_id=session_id, reason=f"egress_start: {exc}"
        )
        logger.exception(
            "interview.recording.start_failed",
            extra={"session_id": str(session_id), "recording_id": str(recording_id)},
        )
        return

    claimed = await recordings_queries.attach_egress(
        db,
        recording_id,
        egress_id=egress_id,
        destination_file=destination,
    )
    if not claimed:
        # Another caller won the pending→active transition (UNIQUE egress_id
        # backstops this). Stop OUR egress so the room does not record twice.
        await _stop_egress_quietly(egress_id, settings)
        logger.warning(
            "interview.recording.duplicate_start_suppressed",
            extra={"session_id": str(session_id), "recording_id": str(recording_id)},
        )
    logger.info(
        "interview.recording.started",
        extra={
            "session_id": str(session_id),
            "recording_id": str(recording_id),
            "egress_id": egress_id,
        },
    )


async def stop_recording_for_session(db: AsyncSession, *, session_id: UUID) -> None:
    """Idempotently stop the session's recording on ANY terminal path.

    Called from ``submit_session``, realtime finalization and the expiry
    sweep — never from a client disconnect. A row that is pending (start call
    died before the Egress id was stamped) is failed outright: reconciliation
    has no job id to repair, and a room whose interview ended must not keep
    recording.
    """
    settings = get_settings()
    if not settings.interview_recording_enabled:
        return
    recording = await recordings_queries.get_recording_for_session(db, session_id)
    if recording is None or recording.status in ("complete", "failed", "cancelled", "expired"):
        return
    if recording.status == "active" and recording.egress_id:
        stopped = await _stop_egress_quietly(recording.egress_id, settings)
        if not stopped:
            logger.warning(
                "interview.recording.stop_deferred",
                extra={"session_id": str(session_id), "egress_id": recording.egress_id},
            )
            return
    await recordings_queries.mark_recording_cancelled(
        db, session_id=session_id, reason="stopped"
    )
    logger.info(
        "interview.recording.stopped",
        extra={"session_id": str(session_id), "recording_id": str(recording.id)},
    )


def verify_webhook_signature(body: str, auth_token: str | None) -> WebhookEvent:
    """Verify a LiveKit webhook delivery and decode its event.

    Raises ``ValueError`` for a missing/invalid/stale signature — the router
    maps that to 401 BEFORE any event data is used. Uses the deployment's
    LiveKit key/secret pair; ``TokenVerifier`` carries the skew leeway.
    """
    settings = get_settings()
    if not (
        settings.livekit_api_key
        and settings.livekit_api_secret
        and body
        and auth_token
    ):
        raise ValueError("missing webhook credentials or signature")
    verifier = lk_api.TokenVerifier(
        settings.livekit_api_key.get_secret_value(),
        settings.livekit_api_secret.get_secret_value(),
        leeway=timedelta(seconds=settings.livekit_webhook_max_skew_seconds),
    )
    receiver = lk_api.WebhookReceiver(verifier)
    return receiver.receive(body, auth_token)


async def handle_egress_webhook(db: AsyncSession, event: WebhookEvent) -> str:
    """Persist the outcome of one verified Egress callback (idempotent).

    Tolerates duplicate and out-of-order deliveries: every state change is a
    conditional UPDATE keyed on the CURRENT status, so a replayed ``ended``
    after the row completed is a no-op, and an ``ended`` that beats its own
    ``started`` to the inbox simply finds the row the ended-handler can drive
    directly. Returns the event verb (for logging/tests).
    """
    event_name = event.event or ""
    if event_name not in _EGRESS_EVENTS or event.egress_info is None:
        return event_name

    egress_id = event.egress_info.egress_id
    status = int(event.egress_info.status)
    recording = await recordings_queries.get_recording_by_egress_id(db, egress_id)
    if recording is None:
        # Not ours (another room/evironment sharing this LiveKit project), or
        # the start call failed before stamping the id — both are no-ops.
        logger.info(
            "interview.recording.callback_unknown_egress",
            extra={"egress_id": egress_id, "event": event_name},
        )
        return event_name

    started = status == int(lk_api.EgressStatus.EGRESS_STARTING) or status == int(
        lk_api.EgressStatus.EGRESS_ACTIVE
    )
    ended_ok = status == int(lk_api.EgressStatus.EGRESS_COMPLETE)
    ended_bad = status in (
        int(lk_api.EgressStatus.EGRESS_FAILED),
        int(lk_api.EgressStatus.EGRESS_ABORTED),
        int(lk_api.EgressStatus.EGRESS_LIMIT_REACHED),
    )

    if started:
        await recordings_queries.mark_egress_active_if_pending_id(db, egress_id=egress_id)
        return event_name

    if ended_bad:
        reason = event.egress_info.error or f"egress status {status}"
        await recordings_queries.mark_recording_failed(db, egress_id=egress_id, reason=reason)
        logger.warning(
            "interview.recording.egress_failed",
            extra={"egress_id": egress_id, "session_id": str(recording.session_id)},
        )
        return event_name

    if ended_ok:
        await _attach_completed_output(db, egress_id=egress_id, info=event.egress_info)
    return event_name


async def reconcile_recordings(  # noqa: C901 - provider states + bounded retry policy
    db: AsyncSession, *, settings: Settings | None = None
) -> int:
    """Repair unresolved Egress jobs whose output has appeared in S3.

    Registered on the ARQ sweep (``workers/recordings.py``). For each row
    stuck ``pending``/``active`` past the grace window, list the provider's
    Egress job: a terminal job is processed exactly like its webhook
    (attachment or failure), a live job is left alone, an unknown job is
    failed after the bounded retry ceiling. Returns the number of rows driven
    to a terminal state.
    """
    settings = settings or get_settings()
    if not settings.interview_recording_enabled:
        return 0
    now = datetime.now(tz=UTC)
    candidates = await recordings_queries.list_unresolved_recordings(
        db, older_than_minutes=5
    )
    settled = 0
    for recording in candidates:
        if recording.egress_id is None or recording.egress_id == "":
            # Start never completed: nothing to interrogate. Charge an attempt
            # so a permanently broken start is bounded, and fail at ceiling.
            attempt = await recordings_queries.charge_reconcile_attempt(
                db, recording_id=recording.id, now=now,
                max_attempts=settings.interview_recording_reconcile_max_attempts,
            )
            if (
                attempt is not None
                and attempt >= settings.interview_recording_reconcile_max_attempts
            ):
                await recordings_queries.mark_recording_failed(
                    db,
                    session_id=recording.session_id,
                    reason="reconcile: egress id never stamped",
                )
                settled += 1
            continue

        lookup_ok, info = await _fetch_egress_info(recording.egress_id, settings)
        if not lookup_ok:
            # Provider API down — inspect later WITHOUT spending an attempt
            # (see stamp_reconcile_error; weather must not eat the budget).
            await recordings_queries.stamp_reconcile_error(
                db,
                recording_id=recording.id,
                error="egress lookup unavailable",
                now=now,
            )
            continue
        if info is None:
            # A healthy provider returned no such job. This is bounded rather
            # than treated as an outage: a deleted/expired provider job must
            # eventually become a durable failed recording.
            attempt = await recordings_queries.charge_reconcile_attempt(
                db,
                recording_id=recording.id,
                now=now,
                max_attempts=settings.interview_recording_reconcile_max_attempts,
            )
            if (
                attempt is not None
                and attempt >= settings.interview_recording_reconcile_max_attempts
            ):
                await recordings_queries.mark_recording_failed(
                    db,
                    egress_id=recording.egress_id,
                    reason="reconcile: provider egress not found",
                )
                settled += 1
            continue

        status = int(info.status)
        if status in (
            int(lk_api.EgressStatus.EGRESS_STARTING),
            int(lk_api.EgressStatus.EGRESS_ACTIVE),
            int(lk_api.EgressStatus.EGRESS_ENDING),
        ):
            await recordings_queries.stamp_reconcile_error(
                db, recording_id=recording.id, error="egress still running", now=now
            )
            continue

        if status in (
            int(lk_api.EgressStatus.EGRESS_FAILED),
            int(lk_api.EgressStatus.EGRESS_ABORTED),
            int(lk_api.EgressStatus.EGRESS_LIMIT_REACHED),
        ):
            await recordings_queries.mark_recording_failed(
                db,
                egress_id=recording.egress_id,
                reason=info.error or f"egress status {status}",
            )
            settled += 1
            continue

        if status == int(lk_api.EgressStatus.EGRESS_COMPLETE):
            attached = await _attach_completed_output(
                db, egress_id=recording.egress_id, info=info
            )
            if attached:
                settled += 1
                continue
        # Still unresolved after this pass: charge the bounded attempt.
        attempt = await recordings_queries.charge_reconcile_attempt(
            db,
            recording_id=recording.id,
            now=now,
            max_attempts=settings.interview_recording_reconcile_max_attempts,
        )
        if attempt is not None and attempt >= settings.interview_recording_reconcile_max_attempts:
            await recordings_queries.mark_recording_failed(
                db,
                egress_id=recording.egress_id,
                reason="reconcile: retry ceiling reached",
            )
            settled += 1
    return settled


async def expire_retention_due_recordings(
    db: AsyncSession, *, settings: Settings | None = None, limit: int = 100
) -> int:
    """Delete expired audio + tombstone (30-day retention, application side).

    The object is deleted FIRST (idempotent) — the tombstone transaction only
    commits once the bytes are actually gone. Returns the number of recordings
    expired.
    """
    settings = settings or get_settings()
    now = datetime.now(tz=UTC)
    due = await recordings_queries.list_expired_recordings(db, now=now, limit=limit)
    expired = 0
    for recording in due:
        if recording.storage_object_id is None:
            continue
        object_row = await recordings_queries.get_storage_object_details(
            db, recording.storage_object_id
        )
        if object_row is not None:
            try:
                await s3.delete_object(
                    _ObjectView(
                        bucket=object_row["bucket"],
                        object_key=object_row["object_key"],
                    ),
                    settings=settings,
                )
            except Exception:  # noqa: BLE001 -- retry next sweep; bytes must go first
                logger.exception(
                    "interview.recording.retention_delete_failed",
                    extra={"recording_id": str(recording.id)},
                )
                continue
        session_id = recording.session_id
        storage_object_id = recording.storage_object_id
        if storage_object_id is None:
            continue
        if await recordings_queries.expire_recording(db, recording_id=recording.id, now=now):
            await recordings_queries.clear_session_pointer_if_equal(
                db, session_id, storage_object_id
            )
            expired += 1
            logger.info(
                "interview.recording.retention_expired",
                extra={"recording_id": str(recording.id), "session_id": str(session_id)},
            )
    return expired


async def get_playback_url(
    db: AsyncSession, *, session_id: UUID
) -> dict[str, Any] | None:
    """Teacher playback payload for one session, or ``None`` when not recorded.

    Availability states: ``not_recorded`` (no row), ``processing``
    (pending/active), ``available`` (complete + object), ``failed``,
    ``expired``. For ``available`` the only sensitive field returned is a
    short-lived signed stream URL — bucket/key/storage ids stay server-side.
    """
    settings = get_settings()
    recording = await recordings_queries.get_recording_for_session(db, session_id)
    if recording is None:
        return {"state": "not_recorded"}
    if recording.status in ("pending", "active"):
        return {"state": "processing"}
    if recording.status == "failed":
        return {"state": "failed"}
    if recording.status == "expired":
        return {"state": "expired"}
    # complete
    if recording.storage_object_id is None:  # defensive; tombstone CHECK forbids
        return {"state": "failed"}
    row = await recordings_queries.get_storage_object_details(
        db, recording.storage_object_id
    )
    if row is None:
        return {"state": "failed"}
    view = _ObjectView(bucket=row["bucket"], object_key=row["object_key"])
    safe_name = (row["original_filename"] or "interview-recording").replace('"', "")
    try:
        url, _ = await s3.create_stream_url(
            view,
            response_headers={
                "Content-Disposition": f'inline; filename="{safe_name}"',
                "Content-Type": recording.mime_type or "audio/mpeg",
                "Cache-Control": "no-store",
            },
            settings=settings,
        )
    except s3.S3NotConfiguredError:
        return {"state": "failed"}
    return {
        "state": "available",
        "stream_url": url,
        "expires_at": datetime.now(tz=UTC)
        + timedelta(seconds=settings.s3_url_ttl_seconds),
        "duration_seconds": recording.duration_seconds,
        "mime_type": recording.mime_type,
        "recorded_at": recording.completed_at or recording.created_at,
    }


async def _attach_completed_output(db: AsyncSession, *, egress_id: str, info: EgressInfo) -> bool:
    """Validate the Egress output and attach it as the canonical storage object.

    The reported file must live under the configured recording prefix, exist
    (``head_object``), be non-trivially sized, and carry an allowed audio MIME.
    On any mismatch the recording is failed — a corrupt or foreign object is
    never wired to playback. Returns True when the row and session pointer
    reached ``complete``.
    """
    settings = get_settings()
    recording = await recordings_queries.get_recording_by_egress_id(db, egress_id)
    if recording is None or recording.status == "complete":
        return recording is not None  # idempotent replay: already done

    file_results = list(getattr(info, "file_results", None) or [])
    if not file_results:
        await recordings_queries.mark_recording_failed(
            db, egress_id=egress_id, reason="completed with no file output"
        )
        return False
    result = file_results[0]
    prefix = settings.interview_recording_s3_prefix.strip("/")
    bucket = recording_bucket(settings)
    reported_location = result.location or result.filename or ""
    filename = _normalize_reported_key(
        reported_location,
        expected_bucket=bucket,
    )
    expected_prefix = (
        f"{prefix}/{recording.session_id}/{recording.id}/"
    )
    if not filename.startswith(expected_prefix):
        await recordings_queries.mark_recording_failed(
            db,
            egress_id=egress_id,
            reason=f"output outside recording prefix: {reported_location}",
        )
        return False

    # LiveKit may rewrite the container (e.g. mp4 requested → mp3 produced);
    # trust the REPORTED key for the object lookup, but only after the exact
    # session/recording prefix check above.
    fmt = settings.interview_recording_format
    allowed = _ALLOWED_MIME_BY_FORMAT.get(fmt, _ALLOWED_MIME_BY_FORMAT["mp3"])
    ext = filename.rsplit(".", 1)[-1].lower()
    mime = _MIME_BY_EXT.get(ext, "audio/mpeg")

    view = _ObjectView(bucket=bucket, object_key=filename)
    try:
        meta = await s3.head_object(view, settings=settings)
    except Exception as exc:  # noqa: BLE001 -- attach is best-effort; reconcile retries
        await recordings_queries.stamp_reconcile_error(
            db, recording_id=recording.id, error=f"head_object: {exc}", now=datetime.now(tz=UTC)
        )
        return False
    if meta is None:
        # Output not visible yet (S3 propagation) — leave unresolved for the
        # reconciler instead of failing a recording whose bytes may be seconds
        # away.
        await recordings_queries.stamp_reconcile_error(
            db,
            recording_id=recording.id,
            error="output not in bucket yet",
            now=datetime.now(tz=UTC),
        )
        return False
    if meta.size < _MIN_AUDIO_BYTES or meta.content_type not in allowed:
        await recordings_queries.mark_recording_failed(
            db,
            egress_id=egress_id,
            reason=f"output failed validation (size={meta.size}, ct={meta.content_type})",
        )
        return False

    storage_object_id = await recordings_queries.upsert_storage_object(
        db,
        bucket=bucket,
        object_key=filename,
        mime_type=mime,
        size_bytes=meta.size,
    )
    retention = datetime.now(tz=UTC) + timedelta(
        days=settings.interview_recording_retention_days
    )
    attached = await recordings_queries.complete_with_output(
        db,
        egress_id=egress_id,
        storage_object_id=storage_object_id,
        mime_type=mime,
        size_bytes=meta.size,
        duration_seconds=(result.duration / 1e9) if result.duration else None,
        retention_delete_at=retention,
    )
    if not attached:
        return False
    # Mirror the playback pointer onto the session — only when blank or equal
    # (never overwrite a pointer a repair just wrote; see queries helper).
    await sessions_queries.set_recording_pointer_if_absent(
        db, session_id=recording.session_id, storage_object_id=storage_object_id
    )
    logger.info(
        "interview.recording.attached",
        extra={
            "egress_id": egress_id,
            "session_id": str(recording.session_id),
            "size_bytes": meta.size,
        },
    )
    return True


class _ObjectView:
    """Duck-typed ``s3.StorageObject`` (bucket+key only) — see ``s3.Protocol``."""

    def __init__(self, *, bucket: str, object_key: str) -> None:
        self.bucket = bucket
        self.object_key = object_key


_MIME_BY_EXT: dict[str, str] = {
    "mp3": "audio/mpeg",
    "ogg": "audio/ogg",
    "mp4": "audio/mp4",
    "m4a": "audio/mp4",
    "wav": "audio/wav",
}


__all__ = [
    "RecordingUnavailableError",
    "build_destination_file",
    "expire_retention_due_recordings",
    "get_playback_url",
    "handle_egress_webhook",
    "reconcile_recordings",
    "start_recording_for_session",
    "stop_recording_for_session",
    "verify_webhook_signature",
]
