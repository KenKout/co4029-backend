"""Durable typed-turn receipts on the native interview path.

The window between "the agent has your text" (the ACK) and "the answer is
durable" (the transcript row + graded fold committed) is a whole LLM grading
call wide. Before this coordinator the ACK meant "in RAM": a crash inside the
window took the answer and the candidate's transcript entry with it, while the
client believed the turn was settled — its snapshot-driven draft clearing would
then never fire, but a reload would show the answer missing from history.

The receipt makes the ordering honest, reusing the unique idempotency index
from migration 0023 (``uq_interview_msg_turn_key`` on
``interview_session_messages.metadata_json->>'turn_key'``):

1. ``begin()`` reserves the key synchronously (closing gate + single-flight),
   BEFORE any await, so a finish cannot slip between reserve and receipt.
2. ``persist_receipt()`` inserts (or loads) the typed answer row with
   ``turn_state='received'`` and commits.
3. ONLY then is the ACK published — it now means "your answer is durable".
4. The graded fold runs; ``mark_applied()`` CASes the row received→applied
   under the processing token and commits in the SAME transaction as the
   runtime-state save, so state and receipt can never disagree.
5. A retry after a crash continues from the receipt state:
   received → re-fold (the work never landed); applied → re-ack + confirm
   snapshot only, never a second fold.

The receipts ride ``interview_session_messages.metadata_json``:
``source='native_agent'``, ``kind='answer'``, ``turn_key``, ``turn_state``,
``bank_question_id`` — no schema change, invisible to old code paths that read
role/kind.
"""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Protocol
from uuid import UUID

from abridgeai.core.security import utcnow

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

# How long a fold "owns" a received receipt. Generous over the grading probe
# (~1s) plus DB work; only matters when two copies of the same turn race — the
# ledger normally dedupes before this is consulted.
PROCESSING_LEASE_SECONDS = 300

_RECEIVED = "received"
_APPLIED = "applied"
_FAILED = "failed"


class TypedTurnReceiptError(RuntimeError):
    """The receipt could not be persisted and the turn must not be ACKed."""


def typed_receipt_metadata(
    *,
    turn_key: str,
    bank_question_id: UUID | None,
    turn_state: str = _RECEIVED,
    processing_token: UUID | None = None,
    processing_expires_at: str | None = None,
) -> dict[str, Any]:
    """The metadata_json block that marks a row as a typed-turn receipt."""
    meta: dict[str, Any] = {
        "source": "native_agent",
        "kind": "answer",
        "turn_key": turn_key[:128],
        "turn_state": turn_state,
    }
    if bank_question_id is not None:
        meta["bank_question_id"] = str(bank_question_id)
    if processing_token is not None:
        meta["processing_token"] = str(processing_token)
    if processing_expires_at is not None:
        meta["processing_expires_at"] = processing_expires_at
    return meta


async def persist_receipt(
    db: AsyncSession,
    *,
    session_id: UUID,
    session_question_id: UUID | None,
    bank_question_id: UUID | None,
    text: str,
    turn_key: str,
) -> tuple[Any, bool]:  # noqa: ANN401 - returns the caller's session's ORM row
    """Insert the typed answer row, or load the existing one. Returns (row, created).

    Commits before returning: the caller publishes the ACK only after this
    returns, so an ACK always means the receipt is DURABLE. The unique index
    makes the insert-or-load atomic — two copies of the same turn race to
    insert and the loser loads the winner's row instead.

    LINKAGE (owned here, not by the caller): ``session_question_id`` may arrive
    as a BANK question id (the typed door captures it pre-fold from the live
    state) or None. When it is not already a resolvable session-question link,
    resolve it HERE via the shared transcript resolver — created on demand
    inside the caller's transaction, concurrent creators converging by
    constraint + reload. A typed receipt stored with a NULL linkage is
    invisible to the evaluator's candidate filter, so the answer could be
    acked/applied yet never graded; resolving at receipt time closes that gap.

    Distinguishing bank ids from session ids on read would need a lookup that
    fails soft; instead the resolver is IDEMPOTENT — a bank id that already has
    a session-question row resolves to it, and one that doesn't gets its row
    created, so passing either id through is safe.
    """
    from sqlalchemy import select  # noqa: PLC0415

    from abridgeai.features.interviews.models import (  # noqa: PLC0415
        InterviewSessionMessage,
    )
    from abridgeai.features.interviews.realtime.native_transcript import (  # noqa: PLC0415
        resolve_session_question,
    )

    if session_question_id is None and bank_question_id is not None:
        session_question_id = await resolve_session_question(
            db, session_id, bank_question_id
        )

    token = uuid_token()
    row = InterviewSessionMessage(
        session_id=session_id,
        session_question_id=session_question_id,
        role="user",
        content_text=text.strip(),
        metadata_json=typed_receipt_metadata(
            turn_key=turn_key,
            bank_question_id=bank_question_id,
            processing_token=token,
            processing_expires_at=(
                utcnow() + timedelta(seconds=PROCESSING_LEASE_SECONDS)
            ).isoformat(),
        ),
    )
    db.add(row)
    try:
        await db.commit()
        return row, True
    except Exception:
        # A concurrent copy won the unique index: load ITS row and continue
        # with that receipt instead. Rollback first — the session is poisoned
        # by the IntegrityError.
        await db.rollback()
    existing = (
        await db.execute(
            select(InterviewSessionMessage).where(
                InterviewSessionMessage.session_id == session_id,
                InterviewSessionMessage.role == "user",
                InterviewSessionMessage.metadata_json["turn_key"].as_string()
                == turn_key[:128],
            )
        )
    ).scalar_one_or_none()
    if existing is None:
        raise TypedTurnReceiptError(
            "typed-turn receipt insert raced and no winner row was found"
        )
    return existing, False


def uuid_token() -> UUID:
    """A fresh processing token (owner identity for the received→applied CAS)."""
    from uuid import uuid4  # noqa: PLC0415

    return uuid4()


def receipt_state(row: Any) -> str | None:  # noqa: ANN401 - ORM row or store dict
    """The ``turn_state`` of a receipt row (or None for non-receipt rows)."""
    meta = row if isinstance(row, dict) else getattr(row, "metadata_json", None) or {}
    state = meta.get("turn_state") if isinstance(meta, dict) else None
    return state if isinstance(state, str) else None


def receipt_is_owned_by_current_caller(
    row: Any, token: UUID | None  # noqa: ANN401 - ORM row from the caller's session
) -> bool:
    """Whether the processing token on a RECEIVED receipt is still fresh.

    A duplicate that lands while the first copy is mid-fold sees the winner's
    token with an unexpired lease: it must wait/re-ack, not fold.
    """
    meta = getattr(row, "metadata_json", None) or {}
    if not isinstance(meta, dict):
        return False
    raw_expires = meta.get("processing_expires_at")
    if not isinstance(raw_expires, str):
        return False
    try:
        expires_at = raw_expires
        from datetime import datetime  # noqa: PLC0415

        parsed = datetime.fromisoformat(expires_at)
    except ValueError:
        return False
    now = utcnow()
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=now.tzinfo)
    return parsed > now


async def load_receipt_by_key(
    db: AsyncSession, *, session_id: UUID, turn_key: str
) -> Any:  # noqa: ANN401 - ORM row from the caller's session
    """The receipt row for a turn key, or None."""
    from sqlalchemy import select  # noqa: PLC0415

    from abridgeai.features.interviews.models import (  # noqa: PLC0415
        InterviewSessionMessage,
    )

    return (
        await db.execute(
            select(InterviewSessionMessage).where(
                InterviewSessionMessage.session_id == session_id,
                InterviewSessionMessage.role == "user",
                InterviewSessionMessage.metadata_json["turn_key"].as_string()
                == turn_key[:128],
            )
        )
    ).scalar_one_or_none()


async def mark_receipt_applied(
    db: AsyncSession,
    row: Any,  # noqa: ANN401 - ORM row from persist_receipt
    *,
    processing_token: str | None,
) -> bool:
    """CAS one receipt received→applied, owner-checked by the processing token.

    Returns True when THIS caller won the transition. False means another call
    already settled the receipt (or the token did not match — a superseded
    folder), and the caller must not fold again.
    """
    from sqlalchemy import update  # noqa: PLC0415

    from abridgeai.features.interviews.models import (  # noqa: PLC0415
        InterviewSessionMessage,
    )

    if not processing_token:
        return False
    meta = dict(getattr(row, "metadata_json", None) or {})
    if meta.get("turn_state") != _APPLIED:
        meta["turn_state"] = _APPLIED
        meta["applied_at"] = utcnow().isoformat()
    result = await db.execute(
        update(InterviewSessionMessage)
        .where(
            InterviewSessionMessage.id == row.id,
            InterviewSessionMessage.metadata_json["turn_state"].as_string() == _RECEIVED,
            InterviewSessionMessage.metadata_json["processing_token"].as_string()
            == processing_token,
        )
        .values(metadata_json=meta)
    )
    await db.commit()
    rowcount: Any = getattr(result, "rowcount", 0)
    return bool(rowcount > 0)


async def mark_receipt_failed(
    db: AsyncSession,
    row: Any,  # noqa: ANN401 - ORM row from persist_receipt
    *,
    processing_token: str | None,
    error_class: str,
) -> bool:
    """CAS one receipt received→failed, owner-checked by the processing token.

    A failed receipt is RETRYABLE, not terminal: the client resends with the
    SAME turn_key and the next callback reclaims the lease and folds again.
    Returns True when THIS caller won the transition.
    """
    from sqlalchemy import update  # noqa: PLC0415

    from abridgeai.features.interviews.models import (  # noqa: PLC0415
        InterviewSessionMessage,
    )

    if not processing_token:
        return False
    meta = dict(getattr(row, "metadata_json", None) or {})
    if meta.get("turn_state") != _RECEIVED:
        return False
    meta["turn_state"] = _FAILED
    # Allowlisted error class only, truncated — never a raw exception string
    # (it can carry prompt or DB detail) and never the answer text itself.
    meta["last_error_class"] = error_class[:64]
    meta["failed_at"] = utcnow().isoformat()
    result = await db.execute(
        update(InterviewSessionMessage)
        .where(
            InterviewSessionMessage.id == row.id,
            InterviewSessionMessage.metadata_json["turn_state"].as_string() == _RECEIVED,
            InterviewSessionMessage.metadata_json["processing_token"].as_string()
            == processing_token,
        )
        .values(metadata_json=meta)
    )
    await db.commit()
    rowcount: Any = getattr(result, "rowcount", 0)
    return bool(rowcount > 0)


async def reclaim_receipt_failed(
    db: AsyncSession,
    *,
    session_id: UUID,
    turn_key: str,
) -> tuple[Any, bool]:  # noqa: ANN401 - ORM row
    """CAS a FAILED receipt back to received under a FRESH processing token.

    This is the retry path's ownership handover: the previous fold died, the
    lease/token is stale, and the new caller takes the row over. Applied rows
    are immutable; received rows under a live lease stay with their owner.
    """

    from abridgeai.features.interviews.models import (  # noqa: PLC0415
        InterviewSessionMessage,
    )

    row = await load_receipt_by_key(db, session_id=session_id, turn_key=turn_key)
    if row is None:
        return None, False
    meta = dict(getattr(row, "metadata_json", None) or {})
    if meta.get("turn_state") != _FAILED:
        return row, False
    token = uuid_token()
    meta["turn_state"] = _RECEIVED
    meta["processing_token"] = str(token)
    meta["processing_expires_at"] = (
        utcnow() + timedelta(seconds=PROCESSING_LEASE_SECONDS)
    ).isoformat()
    meta["reclaimed_at"] = utcnow().isoformat()
    meta.pop("last_error_class", None)
    from sqlalchemy import update  # noqa: PLC0415

    result = await db.execute(
        update(InterviewSessionMessage)
        .where(
            InterviewSessionMessage.id == row.id,
            InterviewSessionMessage.metadata_json["turn_state"].as_string() == _FAILED,
        )
        .values(metadata_json=meta)
    )
    await db.commit()
    rowcount: Any = getattr(result, "rowcount", 0)
    return row, bool(rowcount > 0)


__all__ = [
    "PROCESSING_LEASE_SECONDS",
    "mark_receipt_failed",
    "reclaim_receipt_failed",
    "InMemoryTypedTurnStore",
    "TypedTurnStore",
    "TypedTurnReceiptError",
    "load_receipt_by_key",
    "persist_receipt",
    "mark_receipt_applied",
    "receipt_is_owned_by_current_caller",
    "receipt_state",
    "typed_receipt_metadata",
]


class TypedTurnStore(Protocol):
    """The persistence surface the typed door needs, injectable for tests.

    The production implementation (:func:`persist_receipt`) talks to Postgres
    through the caller's session factory; unit tests supply an in-memory store
    so the callback's ORDERING contract can be tested without a database.
    """

    async def persist(  # noqa: ANN401 - the row type is the implementation's own
        self,
        *,
        session_id: UUID,
        session_question_id: UUID | None,
        bank_question_id: UUID | None,
        text: str,
        turn_key: str,
    ) -> tuple[Any, bool]:
        """Insert or load the durable receipt. Returns (row, created)."""
        ...

    async def mark_applied(
        self,
        row: Any,  # noqa: ANN401 - the row type is the implementation's own
        *,
        state_extra: dict[str, Any] | None = None
    ) -> bool:
        """CAS the row received→applied under the processing token."""
        ...

    async def mark_failed(
        self,
        row: Any,  # noqa: ANN401 - the row type is the implementation's own
        *,
        error_class: str,
    ) -> bool:
        """CAS the row received→failed (owner-checked). A retry reclaims it."""
        ...

    async def lookup(
        self,
        *,
        session_id: UUID,
        turn_key: str,
    ) -> Any | None:  # noqa: ANN401 - the row type is the implementation's own
        """The durable receipt for a turn key, or None. Read-only."""
        ...

    async def reclaim(
        self,
        *,
        session_id: UUID,
        turn_key: str,
    ) -> tuple[Any | None, bool]:  # noqa: ANN401 - the row type is the implementation's own
        """CAS failed→received under a fresh token. (row, claimed)."""
        ...


class InMemoryTypedTurnStore:
    """A store that keeps receipts in a dict — for unit tests only."""

    def __init__(self) -> None:
        self.rows: dict[str, dict[str, Any]] = {}
        self.fail_next_persist = False

    async def persist(
        self,
        *,
        session_id: UUID,
        session_question_id: UUID | None,
        bank_question_id: UUID | None,
        text: str,
        turn_key: str,
    ) -> tuple[dict[str, Any], bool]:
        del session_id, session_question_id, bank_question_id
        if self.fail_next_persist:
            self.fail_next_persist = False
            raise TypedTurnReceiptError("injected receipt failure")
        key = turn_key[:128]
        existing = self.rows.get(key)
        if existing is not None:
            return existing, False
        token = str(uuid_token())
        row = {
            "turn_key": key,
            "text": text.strip(),
            "turn_state": _RECEIVED,
            "processing_token": token,
            "processing_expires_at": (
                utcnow() + timedelta(seconds=PROCESSING_LEASE_SECONDS)
            ).isoformat(),
        }
        self.rows[key] = row
        return row, True

    async def mark_applied(
        self,
        row: Any,  # noqa: ANN401 - in-memory row
        *,
        state_extra: dict[str, Any] | None = None
    ) -> bool:
        if row["turn_state"] != _RECEIVED:
            return False
        row["turn_state"] = _APPLIED
        if state_extra:
            row.update(state_extra)
        return True

    async def mark_failed(
        self,
        row: Any,  # noqa: ANN401 - in-memory row
        *,
        error_class: str,
    ) -> bool:
        if row["turn_state"] != _RECEIVED:
            return False
        row["turn_state"] = _FAILED
        # Allowlisted class only, truncated — never a raw exception string.
        row["last_error_class"] = error_class[:64]
        return True

    async def lookup(
        self,
        *,
        session_id: UUID,
        turn_key: str,
    ) -> dict[str, Any] | None:
        del session_id
        return self.rows.get(turn_key[:128])

    async def reclaim(
        self,
        *,
        session_id: UUID,
        turn_key: str,
    ) -> tuple[dict[str, Any] | None, bool]:
        """Failed (or expired-lease) receipt → received under a FRESH token.

        Returns (row, claimed). claimed=False means someone else owns it or the
        state did not permit reclaiming (applied is immutable).
        """
        del session_id
        row = self.rows.get(turn_key[:128])
        if row is None or row["turn_state"] != _FAILED:
            return row, False
        row["turn_state"] = _RECEIVED
        row["processing_token"] = str(uuid_token())
        row["processing_expires_at"] = (
            utcnow() + timedelta(seconds=PROCESSING_LEASE_SECONDS)
        ).isoformat()
        row.pop("last_error_class", None)
        return row, True
