"""Persistence layer for the adaptive-interviewer runtime state (Phase 1).

Wraps :class:`InterviewRuntimeState` with lazy initialisation, typed
(de)serialization via :mod:`orchestrator.state`, and a version-guarded save
that prevents double-advancement on retried REST / duplicate LiveKit callbacks.

Concurrency contract
---------------------
* ``load_or_init`` returns the current typed state, creating the row lazily
  for a session that predates the adaptive flag (or never had one). The DB
  ``state_version`` starts at 0.
* ``save`` takes the ``expected_version`` the caller read. If the row's
  version has moved on (a concurrent turn already advanced), the save is
  rejected with :class:`StaleStateError` and the caller must reload — this is
  the optimistic lock that guarantees one advancement per turn.
* ``last_turn_idempotency_key`` lets a caller detect a duplicate replay of the
  SAME student turn (same key) and short-circuit without re-running the LLM
  pipeline or advancing twice.

This module does NO policy and NO LLM work — it is pure persistence so it can
be unit-tested against a real session row without any orchestrator wiring.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import select

from abridgeai.features.interviews.models import InterviewRuntimeState
from abridgeai.features.interviews.orchestrator.state import (
    InterviewPhase,
    InterviewRuntimeStateData,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


class StaleStateError(RuntimeError):
    """Raised when a version-guarded save loses the optimistic-lock race.

    The caller read the state at version N, but by commit time the row was
    already at a newer version (a concurrent/duplicate turn advanced it). The
    caller should reload and decide whether the turn is a duplicate replay.
    """

    def __init__(self, session_id: UUID, expected: int, actual: int) -> None:
        super().__init__(
            f"stale interview runtime state for session {session_id}: "
            f"expected version {expected}, found {actual}"
        )
        self.session_id = session_id
        self.expected = expected
        self.actual = actual


class LoadedRuntimeState:
    """A typed state payload plus the DB bookkeeping the caller needs to save.

    ``version`` is the optimistic-lock counter the caller read; pass it back to
    :func:`save` as ``expected_version``. ``last_turn_idempotency_key`` is the
    key of the previously-processed turn (None if the session has taken no turn
    yet) so the caller can detect a duplicate replay.
    """

    __slots__ = ("data", "last_turn_idempotency_key", "version")

    def __init__(
        self,
        data: InterviewRuntimeStateData,
        version: int,
        last_turn_idempotency_key: str | None,
    ) -> None:
        self.data = data
        self.version = version
        self.last_turn_idempotency_key = last_turn_idempotency_key


async def _get_row(db: AsyncSession, session_id: UUID) -> InterviewRuntimeState | None:
    stmt = select(InterviewRuntimeState).where(InterviewRuntimeState.session_id == session_id)
    return (await db.execute(stmt)).scalar_one_or_none()


async def load_or_init(db: AsyncSession, session_id: UUID) -> LoadedRuntimeState:
    """Return the typed runtime state for a session, creating it lazily.

    A freshly-created row starts at ``state_version=0`` with a default
    (opening-phase) payload. Does NOT commit — the caller owns the transaction
    boundary so state creation joins whatever unit of work is in flight.
    """
    row = await _get_row(db, session_id)
    if row is None:
        data = InterviewRuntimeStateData()
        row = InterviewRuntimeState(
            session_id=session_id,
            phase=data.phase.value,
            state_version=0,
            last_turn_idempotency_key=None,
            state_json=data.to_dict(),
        )
        db.add(row)
        await db.flush()
        return LoadedRuntimeState(data=data, version=0, last_turn_idempotency_key=None)

    data = InterviewRuntimeStateData.from_dict(row.state_json)
    return LoadedRuntimeState(
        data=data,
        version=row.state_version,
        last_turn_idempotency_key=row.last_turn_idempotency_key,
    )


async def load_readonly(db: AsyncSession, session_id: UUID) -> LoadedRuntimeState | None:
    """Return the typed runtime state, or ``None`` when no row exists yet.

    Unlike :func:`load_or_init`, this NEVER creates a row — it is a pure read.
    The security guard uses this so that merely *assessing* a turn (especially
    in shadow mode, or on the legacy path where the adaptive orchestrator never
    runs) does not materialize a runtime-state row that wouldn't otherwise
    exist. Callers that need to persist counters fall back to ``load_or_init``
    only when they are actually going to write.
    """
    row = await _get_row(db, session_id)
    if row is None:
        return None
    data = InterviewRuntimeStateData.from_dict(row.state_json)
    return LoadedRuntimeState(
        data=data,
        version=row.state_version,
        last_turn_idempotency_key=row.last_turn_idempotency_key,
    )


async def save(
    db: AsyncSession,
    session_id: UUID,
    data: InterviewRuntimeStateData,
    *,
    expected_version: int,
    turn_idempotency_key: str | None = None,
) -> int:
    """Persist ``data`` with an optimistic-lock check; return the new version.

    Rejects the write with :class:`StaleStateError` when the row's current
    version differs from ``expected_version`` (a concurrent turn won the race).
    On success bumps ``state_version`` by one, mirrors ``phase`` into its hot
    column, and records ``turn_idempotency_key`` as the last processed turn.

    Whole-dict reassignment of ``state_json`` is intentional: the ORM only
    reliably flags a JSONB column dirty on reassignment, not in-place mutation.

    The version check lives INSIDE the UPDATE's WHERE clause, not in a preceding
    SELECT. A read-then-write pair is not atomic under READ COMMITTED: while one
    transaction holds the row lock with an uncommitted UPDATE, a second one still
    SELECTs the OLD version, so its guard passes; its UPDATE then blocks on the
    lock and lands anyway once the first commits. That silently discarded a turn
    — reproduced against this database as ``state_version`` stuck at 1 after two
    turns, with the second turn's ``last_turn_idempotency_key`` overwriting the
    first and its state (hint counters, security attempt counts) lost.

    Postgres re-evaluates the WHERE of a blocked UPDATE against the *committed*
    row once the lock is released, so ``state_version = :expected`` fails there
    and the write matches zero rows — which is how the race is now detected.
    """
    from sqlalchemy import update  # noqa: PLC0415

    # A bulk UPDATE bypasses the ORM's identity map. Expire any instance this
    # session already loaded BEFORE issuing it, so a later read in the same
    # transaction refetches instead of serving pre-save column values.
    stale_instance = await _get_row(db, session_id)
    if stale_instance is not None:
        db.expire(stale_instance)

    values: dict[str, object] = {
        "state_json": data.to_dict(),
        "phase": data.phase.value,
        "state_version": expected_version + 1,
    }
    if turn_idempotency_key is not None:
        values["last_turn_idempotency_key"] = turn_idempotency_key

    result = await db.execute(
        update(InterviewRuntimeState)
        .where(
            InterviewRuntimeState.session_id == session_id,
            InterviewRuntimeState.state_version == expected_version,
        )
        .values(**values)
        .returning(InterviewRuntimeState.state_version)
    )
    new_version = result.scalar_one_or_none()
    if new_version is None:
        # Zero rows matched: either no row exists (caller skipped load_or_init)
        # or the version moved on. Re-read to report which, and to give the
        # caller the actual version for its duplicate-replay decision.
        row = await _get_row(db, session_id)
        raise StaleStateError(
            session_id, expected_version, row.state_version if row is not None else -1
        )

    return int(new_version)


def is_duplicate_turn(loaded: LoadedRuntimeState, turn_idempotency_key: str | None) -> bool:
    """True when this turn key was already processed (a replay/duplicate).

    A ``None`` key can never be a duplicate (the caller opted out of
    idempotency for this turn).
    """
    if turn_idempotency_key is None:
        return False
    return loaded.last_turn_idempotency_key == turn_idempotency_key


__all__ = [
    "InterviewPhase",
    "LoadedRuntimeState",
    "StaleStateError",
    "is_duplicate_turn",
    "load_or_init",
    "save",
]
