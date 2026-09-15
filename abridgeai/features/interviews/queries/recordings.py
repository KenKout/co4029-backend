"""Data accessors for the interview audio-recording lifecycle.

Conventions (see the feature's other query modules):

* SELECT / UPDATE via the ORM (single-feature tables; no cross-feature
  metadata walk is possible here — the only FK out is to ``storage_objects``,
  and it is written as a bare UUID column value, never resolved as a
  relationship).
* The claim (insert) is a conditional INSERT — a plain ``insert()``
  statement with ``on_conflict_do_nothing`` — so two concurrent
  realtime-token mints cannot both create a recording row for the same
  session (UNIQUE ``session_id``). ``SELECT-then-INSERT`` is a race; the DB
  constraint is the guarantee.

Terminal-state transitions (``complete``/``failed``/``cancelled``/``expired``,
attachment, tombstoning) are conditional UPDATEs keyed on the CURRENT status:
a duplicate or out-of-order Egress callback that replays after the retention
sweeper expired the row must not resurrect it or double-write the pointer.
Every mutating helper returns the outcome it decided, so callers can log it.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import ColumnElement, select, text, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.features.interviews.models import InterviewRecording

TERMINAL_RECORDING_STATUSES = ("complete", "failed", "cancelled", "expired")


async def get_recording_for_session(
    db: AsyncSession, session_id: UUID
) -> InterviewRecording | None:
    stmt = select(InterviewRecording).where(InterviewRecording.session_id == session_id)
    return (await db.execute(stmt)).scalar_one_or_none()


async def get_recording_by_egress_id(
    db: AsyncSession, egress_id: str
) -> InterviewRecording | None:
    stmt = select(InterviewRecording).where(InterviewRecording.egress_id == egress_id)
    return (await db.execute(stmt)).scalar_one_or_none()


async def claim_recording(
    db: AsyncSession,
    *,
    session_id: UUID,
    room_name: str,
    consent_policy_version: str,
    consented_at: datetime,
    consent_scope: str = "audio_only",
) -> UUID | None:
    """Create this session's recording row (``pending``) if absent.

    Returns the recording id in both cases: a newly inserted row, or the
    existing pending row created by the consent endpoint. A row already
    active/complete is also returned so the start service can treat it as an
    idempotent rejoin and avoid a second provider job.
    """
    recording_id = uuid4()
    stmt = (
        pg_insert(InterviewRecording)
        .values(
            id=recording_id,
            session_id=session_id,
            room_name=room_name,
            status="pending",
            consent_policy_version=consent_policy_version,
            consented_at=consented_at,
            consent_scope=consent_scope,
        )
        .on_conflict_do_nothing(index_elements=["session_id"])
        .returning(InterviewRecording.id)
    )
    row = (await db.execute(stmt)).scalar_one_or_none()
    await db.flush()
    if row is not None:
        return row
    existing = await get_recording_for_session(db, session_id)
    return existing.id if existing is not None else None


async def attach_egress(
    db: AsyncSession,
    recording_id: UUID,
    *,
    egress_id: str,
    destination_file: str,
) -> bool:
    """Move ``pending → active`` and stamp the Egress identity.

    Conditional on ``status = 'pending'``: exactly one caller wins the
    transition even if two paths race to start the same recording (the second
    caller's start call is the duplicate the UNIQUE ``egress_id`` backstops).
    """
    now = datetime.now(tz=UTC)
    stmt = (
        update(InterviewRecording)
        .where(
            InterviewRecording.id == recording_id,
            InterviewRecording.status == "pending",
        )
        .values(
            egress_id=egress_id,
            destination_file=destination_file,
            status="active",
            updated_at=now,
        )
    )
    result = await db.execute(stmt)
    await db.flush()
    return bool(result.rowcount)


async def mark_egress_active_if_pending_id(
    db: AsyncSession,
    *,
    egress_id: str,
) -> bool:
    """A callback/inspection observed the Egress actually running.

    Only upgrades a row that somehow stayed ``pending`` after its start call —
    never touches a row in a later state.
    """
    now = datetime.now(tz=UTC)
    stmt = (
        update(InterviewRecording)
        .where(
            InterviewRecording.egress_id == egress_id,
            InterviewRecording.status == "pending",
        )
        .values(status="active", updated_at=now)
    )
    result = await db.execute(stmt)
    await db.flush()
    return bool(result.rowcount)


async def mark_recording_failed(
    db: AsyncSession,
    *,
    egress_id: str | None = None,
    session_id: UUID | None = None,
    reason: str,
) -> bool:
    """Terminal failure — conditional on a non-terminal current status."""
    now = datetime.now(tz=UTC)
    predicate = _identity_predicate(egress_id=egress_id, session_id=session_id)
    stmt = (
        update(InterviewRecording)
        .where(
            predicate,
            InterviewRecording.status.in_(("pending", "active")),
        )
        .values(status="failed", last_error=_truncate(reason), updated_at=now)
    )
    result = await db.execute(stmt)
    await db.flush()
    return bool(result.rowcount)


async def mark_recording_cancelled(
    db: AsyncSession,
    *,
    session_id: UUID,
    reason: str | None = None,
) -> bool:
    """Stop-before-output terminal state (declined consent / provider teardown)."""
    now = datetime.now(tz=UTC)
    values: dict[str, Any] = {"status": "cancelled", "updated_at": now}
    if reason:
        values["last_error"] = _truncate(reason)
    stmt = (
        update(InterviewRecording)
        .where(
            InterviewRecording.session_id == session_id,
            InterviewRecording.status.in_(("pending", "active")),
        )
        .values(**values)
    )
    result = await db.execute(stmt)
    await db.flush()
    return bool(result.rowcount)


async def complete_with_output(
    db: AsyncSession,
    *,
    egress_id: str,
    storage_object_id: UUID,
    mime_type: str,
    size_bytes: int,
    duration_seconds: float | None,
    retention_delete_at: datetime,
) -> bool:
    """Attach the validated output: recording → ``complete`` + FK, session pointer.

    Idempotent and single-flight: the UPDATE only lands while the row is still
    ``active`` (or ``pending`` for a start-then-immediately-ended Egress), and
    a ``complete`` row is left untouched. Callers can therefore replay the
    same Egress-end callback safely. The caller mirrors the session pointer in
    the SAME transaction when this returns True (see
    ``services.recording.attach_output``), which is also what keeps the
    ``no pointer overwrite`` invariant: an existing session pointer is only
    written when it is NULL or equal.
    """
    now = datetime.now(tz=UTC)
    stmt = (
        update(InterviewRecording)
        .where(
            InterviewRecording.egress_id == egress_id,
            InterviewRecording.status.in_(("pending", "active")),
        )
        .values(
            status="complete",
            storage_object_id=storage_object_id,
            mime_type=mime_type,
            size_bytes=size_bytes,
            duration_seconds=duration_seconds,
            completed_at=now,
            retention_delete_at=retention_delete_at,
            updated_at=now,
        )
    )
    result = await db.execute(stmt)
    await db.flush()
    return bool(result.rowcount)


async def clear_session_pointer_if_equal(
    db: AsyncSession, session_id: UUID, storage_object_id: UUID
) -> None:
    """Null the session playback pointer, but ONLY while it still names ``storage_object_id``.

    Retention deletes through this helper can never clobber a pointer a
    concurrent repair legitimately re-attached (evaluations repair rows the
    same way — see ``queries.sessions`` recovery helpers).
    """
    from sqlalchemy import update as _update  # noqa: PLC0415  — localised re-import

    from abridgeai.features.interviews.models import InterviewSession  # noqa: PLC0415

    stmt = (
        _update(InterviewSession)
        .where(
            InterviewSession.id == session_id,
            InterviewSession.recording_object_id == storage_object_id,
        )
        .values(recording_object_id=None, updated_at=datetime.now(tz=UTC))
    )
    await db.execute(stmt)
    await db.flush()


async def expire_recording(
    db: AsyncSession,
    *,
    recording_id: UUID,
    now: datetime,
) -> bool:
    """Retention tombstone: ``complete → expired`` + clear the output FK.

    The S3 delete happens BEFORE this call (the caller deletes first, then
    commits the tombstone — deleting the row's pointer only after the bytes
    are actually gone keeps replay available until the deletion succeeded).
    """
    stmt = (
        update(InterviewRecording)
        .where(
            InterviewRecording.id == recording_id,
            InterviewRecording.status == "complete",
        )
        .values(
            status="expired",
            storage_object_id=None,
            deleted_at=now,
            updated_at=now,
        )
    )
    result = await db.execute(stmt)
    await db.flush()
    return bool(result.rowcount)


async def charge_reconcile_attempt(
    db: AsyncSession,
    *,
    recording_id: UUID,
    now: datetime,
    max_attempts: int,
) -> int | None:
    """Increment the reconcile counter (bounded), return the NEW count.

    Returns ``None`` when the row must NOT be re-driven this pass: it is
    terminal, or already at the attempt ceiling. Conditional-UPDATE based so
    two concurrent sweeps cannot both cross the ceiling (same reasoning as the
    evaluation recovery stamp).
    """
    stmt = (
        update(InterviewRecording)
        .where(
            InterviewRecording.id == recording_id,
            InterviewRecording.status.in_(("pending", "active")),
            InterviewRecording.reconcile_attempts < max_attempts,
        )
        .values(
            reconcile_attempts=InterviewRecording.reconcile_attempts + 1,
            last_reconcile_at=now,
            updated_at=now,
        )
        .returning(InterviewRecording.reconcile_attempts)
    )
    row = (await db.execute(stmt)).scalar_one_or_none()
    await db.flush()
    return row


async def stamp_reconcile_error(
    db: AsyncSession,
    *,
    recording_id: UUID,
    error: str,
    now: datetime,
) -> None:
    """Record the last inspection error WITHOUT consuming an attempt.

    Inspection failures (S3 unreachable, provider API down) are infrastructure
    weather — they must not silently exhaust the repair budget that a real
    lost-callback repair needs. Mirrors the evaluation recovery's
    refund-on-dispatch-failure reasoning.
    """
    stmt = (
        update(InterviewRecording)
        .where(InterviewRecording.id == recording_id)
        .values(last_error=_truncate(error), last_reconcile_at=now, updated_at=now)
    )
    await db.execute(stmt)
    await db.flush()


async def get_storage_object_details(
    db: AsyncSession, storage_object_id: UUID
) -> dict[str, Any] | None:
    row = (
        await db.execute(
            text(
                "SELECT bucket, object_key, mime_type, size_bytes, original_filename "
                "FROM storage_objects WHERE id = :id"
            ),
            {"id": str(storage_object_id)},
        )
    ).mappings().first()
    return dict(row) if row is not None else None


async def upsert_storage_object(
    db: AsyncSession,
    *,
    bucket: str,
    object_key: str,
    mime_type: str,
    size_bytes: int,
) -> UUID:
    existing = (
        await db.execute(
            text("SELECT id FROM storage_objects WHERE bucket = :b AND object_key = :k"),
            {"b": bucket, "k": object_key},
        )
    ).first()
    if existing is not None:
        object_id = existing.id
        await db.execute(
            text(
                "UPDATE storage_objects SET size_bytes = :s, mime_type = :m, "
                "updated_at = NOW() WHERE id = :id"
            ),
            {"s": size_bytes, "m": mime_type, "id": str(object_id)},
        )
        await db.flush()
        return object_id

    object_id = uuid4()
    await db.execute(
        text(
            """
            INSERT INTO storage_objects
                (id, bucket, object_key, original_filename, mime_type,
                 size_bytes, uploaded_by, uploaded_at, created_at, updated_at)
            VALUES
                (:id, :bucket, :object_key, :original_filename, :mime_type,
                 :size_bytes, NULL, NOW(), NOW(), NOW())
            """
        ),
        {
            "id": object_id,
            "bucket": bucket,
            "object_key": object_key,
            "original_filename": object_key.rsplit("/", 1)[-1],
            "mime_type": mime_type,
            "size_bytes": size_bytes,
        },
    )
    await db.flush()
    return object_id


async def list_unresolved_recordings(
    db: AsyncSession, *, older_than_minutes: int, limit: int = 50
) -> list[InterviewRecording]:
    """Rows stuck in ``pending``/``active`` long enough to suspect a lost callback."""
    cutoff = datetime.now(tz=UTC) - timedelta(minutes=max(1, older_than_minutes))
    stmt = (
        select(InterviewRecording)
        .where(
            InterviewRecording.status.in_(("pending", "active")),
            InterviewRecording.updated_at < cutoff,
        )
        .order_by(InterviewRecording.updated_at.asc())
        .limit(limit)
    )
    return list((await db.execute(stmt)).scalars().all())


async def list_expired_recordings(
    db: AsyncSession, *, now: datetime, limit: int = 100
) -> list[InterviewRecording]:
    """Complete recordings whose retention deadline has passed."""
    stmt = (
        select(InterviewRecording)
        .where(
            InterviewRecording.status == "complete",
            InterviewRecording.retention_delete_at.is_not(None),
            InterviewRecording.retention_delete_at <= now,
        )
        .order_by(InterviewRecording.retention_delete_at.asc())
        .limit(limit)
    )
    return list((await db.execute(stmt)).scalars().all())


def _identity_predicate(
    *,
    egress_id: str | None,
    session_id: UUID | None,
) -> ColumnElement[bool]:
    if egress_id is not None:
        return InterviewRecording.egress_id == egress_id
    if session_id is not None:
        return InterviewRecording.session_id == session_id
    raise ValueError("mark_recording_failed requires egress_id or session_id")


def _truncate(reason: str) -> str:
    return reason if len(reason) <= 2000 else reason[:2000]



__all__ = [
    "TERMINAL_RECORDING_STATUSES",
    "attach_egress",
    "charge_reconcile_attempt",
    "claim_recording",
    "clear_session_pointer_if_equal",
    "complete_with_output",
    "expire_recording",
    "get_recording_by_egress_id",
    "get_recording_for_session",
    "list_expired_recordings",
    "list_unresolved_recordings",
    "mark_egress_active_if_pending_id",
    "mark_recording_cancelled",
    "mark_recording_failed",
    "stamp_reconcile_error",
]
