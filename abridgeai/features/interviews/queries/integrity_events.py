"""Concurrency-safe persistence for browser integrity events.

The ingest endpoint used to read ``integrity_score`` into Python, skip ids it
had just SELECTed, and ``db.add`` the rest — a read-then-write batch that lost
updates whenever two batches overlapped (both read the same base score; the
later commit decided) and turned a duplicate ``client_event_id`` into a raw
unique-violation 500. These helpers push the decisions into single SQL
statements instead:

* inserts are bulk ``ON CONFLICT DO NOTHING`` with ``RETURNING``, so a replayed
  id that raced the dedupe SELECT degrades to "already recorded" instead of a
  500, and the request learns exactly which rows IT contributed;
* the score is applied ADDITIVELY in SQL (``score = score + delta``) from the
  rows that actually inserted — a concurrent winner's contribution is always
  underneath, so no commit can drag the total backwards;
* the one-shot warning is a conditional UPDATE on ``integrity_warning_issued
  IS DISTINCT FROM true`` evaluated against the CURRENT persisted score —
  exactly one request's statement can match (a blocked one re-evaluates the
  predicate against the committed winner), which makes the flag transition and
  its ``warning_issued`` evidence row exactly-once without any explicit lock.

All SQLAlchemy stays in ``queries/`` per the services-no-sql contract; the
router composes these with the pure scoring helpers in
``schemas/integrity.py``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from sqlalchemy import select, text, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.features.interviews.models import (
    AssessmentIntegrityEvent,
    InterviewSession,
)
from abridgeai.features.interviews.schemas.integrity import (
    IntegrityEventItem,
    coerce_client_event_id,
    is_scored_event,
    weight_for,
)


async def _stage_rows(
    db: AsyncSession,
    session_id: UUID,
    student_id: UUID,
    events: list[IntegrityEventItem],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Split one batch into unscored + scored rows, dropping recorded ids.

    The scored rows are staged for ONE bulk insert; per-row ``db.add`` would
    turn a duplicate that raced the dedupe SELECT into a unique-violation 500
    at commit. An id repeated inside the same batch is staged once — the first
    occurrence owns the score contribution.
    """
    scored_rows: list[dict[str, object]] = []
    unscored_rows: list[dict[str, object]] = []
    for item in events:
        common: dict[str, object] = {
            "assessment_kind": "interview",
            "interview_session_id": session_id,
            "student_id": student_id,
            "event_type": item.event_type,
            "severity": item.severity,
            "metadata_json": dict(item.metadata),
        }
        if not is_scored_event(item.event_type):
            # reconnect / disconnect: recorded for the teacher timeline, never
            # scored (a network blip is not an integrity signal).
            unscored_rows.append(common)
            continue
        client_event_id = coerce_client_event_id(item.metadata.get("client_event_id"))
        if await scored_event_exists(db, session_id, client_event_id):
            # Idempotent retry: the batch was already persisted (and scored).
            continue
        if client_event_id is not None and any(
            row["client_event_id"] == client_event_id for row in scored_rows
        ):
            # Duplicate id inside ONE batch: first occurrence wins.
            continue
        common["client_event_id"] = client_event_id
        scored_rows.append(common)
    return scored_rows, unscored_rows


async def record_batch(
    db: AsyncSession,
    *,
    session_id: UUID,
    student_id: UUID,
    policy: dict[str, int],
    events: list[IntegrityEventItem],
    warned_already: bool,
) -> tuple[int, bool, int]:
    """Ingest one batch: insert, score additively, fire the one-shot warning.

    Returns ``(score_after, warning_issued_by_this_request, threshold)``. The
    caller commits; this function only stages and executes statements in the
    caller's transaction. ``warned_already`` is the flag value the caller read
    when it loaded the session — a session that already warned never re-warns.
    """
    threshold = int(policy.get("score_threshold", 0) or 0)
    scored_rows, unscored_rows = await _stage_rows(db, session_id, student_id, events)

    # Single bulk insert. ON CONFLICT DO NOTHING makes a lost dedupe race a
    # no-op; RETURNING tells us which rows THIS request contributed, so the
    # score delta is derived from durable reality, never from the stale read.
    inserted_ids = await insert_events_ignore_conflicts(db, scored_rows)
    if unscored_rows:
        await insert_events_ignore_conflicts(db, unscored_rows)

    # Additive-in-SQL scoring: the delta is the weight sum of the staged rows
    # that actually inserted, applied on top of whatever any concurrent batch
    # has already committed (``score = score + delta``), so neither commit can
    # erase or regress the other. ``weight_for`` returns 0 for anything the
    # snapshot does not know — policy-only scoring, as before.
    delta = sum(
        weight_for(str(row["event_type"]), policy)
        for row in scored_rows
        if row["client_event_id"] in inserted_ids
    )
    score_after = await add_integrity_score(db, session_id, delta)

    # One-shot warning: exactly one racing request's conditional UPDATE can
    # flip the flag (the loser re-evaluates the WHERE against the committed
    # winner and matches nothing), so the evidence row is written only by the
    # request that actually fired the transition.
    warning_issued = False
    if not warned_already and delta > 0 and await try_flag_integrity_warning(
        db, session_id, threshold, datetime.now(tz=UTC)
    ):
        warning_issued = True
        # Server-generated evidence row: the client cannot write this event
        # type into the timeline (its own warning_issued posts are recorded
        # but never score and never set the flag).
        await insert_events_ignore_conflicts(
            db,
            [
                {
                    "assessment_kind": "interview",
                    "interview_session_id": session_id,
                    "student_id": student_id,
                    "event_type": "warning_issued",
                    "severity": "warning",
                    "metadata_json": {
                        "integrity_score_after": score_after,
                        "integrity_score_threshold": threshold,
                        "integrity_weight_tab_switch": int(
                            policy.get("tab_switch", 0) or 0
                        ),
                        "integrity_weight_focus_lost": int(
                            policy.get("focus_lost", 0) or 0
                        ),
                        "integrity_weight_fullscreen_exit": int(
                            policy.get("fullscreen_exit", 0) or 0
                        ),
                    },
                }
            ],
        )
    return score_after, warning_issued, threshold


async def scored_event_exists(
    db: AsyncSession, session_id: UUID, client_event_id: UUID | None
) -> bool:
    """Whether this client id is ALREADY recorded for the session.

    The pre-insert dedupe stays as the common path (an ordinary retry reads a
    hit and skips the insert entirely); the unique index remains the authority
    for the racy case the SELECT cannot see.
    """
    if client_event_id is None:
        return False
    stmt = (
        select(AssessmentIntegrityEvent.id)
        .where(
            AssessmentIntegrityEvent.interview_session_id == session_id,
            AssessmentIntegrityEvent.client_event_id == client_event_id,
        )
        .limit(1)
    )
    return (await db.execute(stmt)).scalar_one_or_none() is not None


async def insert_events_ignore_conflicts(
    db: AsyncSession, rows: list[dict[str, Any]]
) -> set[UUID | None]:
    """Bulk-insert event rows; a lost insert race is a no-op, not a 500.

    ``ON CONFLICT DO NOTHING`` targets the partial unique index
    ``uq_integrity_events_session_client`` (migration 0113) — the same
    constraint the ORM flush used to surface as a raw IntegrityError. Returns
    the client ids that THIS statement actually inserted (a row whose
    ``(session_id, client_event_id)`` a concurrent batch just committed is
    skipped and absent from the set; id-less rows always insert and are
    represented as ``None``).
    """
    if not rows:
        return set()
    stmt = pg_insert(AssessmentIntegrityEvent).values(rows).on_conflict_do_nothing(
        index_elements=["interview_session_id", "client_event_id"],
        # The unique index is PARTIAL (client_event_id IS NOT NULL, migration
        # 0113) — Postgres refuses a conflict target that does not carry the
        # index's own predicate.
        index_where=text("client_event_id IS NOT NULL"),
    ).returning(AssessmentIntegrityEvent.client_event_id)
    result = await db.execute(stmt)
    return {row[0] for row in result.all()}


async def try_flag_integrity_warning(
    db: AsyncSession,
    session_id: UUID,
    threshold: int,
    now: datetime,
) -> bool:
    """Claim the one-shot warning transition; True only for exactly one caller.

    Call this AFTER this request's delta has been applied to the persisted
    score: the WHERE tests the CURRENT total against the threshold, so the
    request whose contribution carried the total across is the one that
    matches. Two overlapping batches cannot both match: the loser blocks on
    the winner's row lock, re-evaluates the predicate against the committed
    row, matches nothing, and reports False — so the caller inserts its
    ``warning_issued`` evidence row only when it really fired the transition.
    """
    if threshold <= 0:
        return False
    result = await db.execute(
        update(InterviewSession)
        .where(
            InterviewSession.id == session_id,
            text("integrity_warning_issued IS DISTINCT FROM true"),
            text("integrity_score >= :threshold"),
        )
        .values(
            integrity_warning_issued=True,
            session_security_flagged=True,
            # Keep the FIRST flagging timestamp: written only when this
            # statement is the one flipping the flag.
            integrity_threshold_flagged_at=text(
                "COALESCE(integrity_threshold_flagged_at, :now)"
            ),
        )
        .returning(InterviewSession.id),
        {"threshold": threshold, "now": now},
    )
    return result.scalar_one_or_none() is not None


async def add_integrity_score(db: AsyncSession, session_id: UUID, delta: int) -> int:
    """Add ``delta`` to the live score in SQL; return the persisted total.

    Additive on the server-side value, so a concurrent batch that commits
    between this request's read and its write stays in the total (no
    lost update, no regression). The returned total includes everything
    committed so far — the honest number to hand back to the client.
    """
    if delta == 0:
        current = (
            await db.execute(
                select(InterviewSession.integrity_score).where(
                    InterviewSession.id == session_id
                )
            )
        ).scalar_one()
        return int(current or 0)
    result = await db.execute(
        update(InterviewSession)
        .where(InterviewSession.id == session_id)
        .values(integrity_score=InterviewSession.integrity_score + delta)
        .returning(InterviewSession.integrity_score)
    )
    return int(result.scalar_one())


async def list_integrity_events_for_session(db: AsyncSession, session_id: UUID) -> list[Any]:
    """Return FR-5.8 proctoring events for one interview session, oldest first.

    Projection source for the teacher gap-report integrity timeline. Scoped to
    ``assessment_kind='interview'``. Empty list when none were recorded (the
    common case for an honest take). Mirrors the quiz-side
    ``list_integrity_events_for_attempt``.
    """
    stmt = (
        select(AssessmentIntegrityEvent)
        .where(
            AssessmentIntegrityEvent.assessment_kind == "interview",
            AssessmentIntegrityEvent.interview_session_id == session_id,
        )
        .order_by(AssessmentIntegrityEvent.created_at.asc())
    )
    return list((await db.execute(stmt)).scalars().all())


__all__ = ["list_integrity_events_for_session"]
