"""Durable recovery-lifecycle bookkeeping on ``interview_sessions``.

Split out of :mod:`sessions` (the LOC gate) — these four writers own the
``internal_summary_json`` ``evaluation_recovery`` sub-key and the evaluation
claim columns, and NOTHING else may write either:

* ``stamp`` charges an attempt and records a dispatching job — atomically
  guarded by verdict-null / claim-expired / no-active-job / under-ceiling;
* ``refund`` rolls a failed dispatch back, scoped to the attempt it wrote;
* ``transition`` moves the CURRENT job's phase (dispatching→queued→running→
  retrying→terminal), CAS-scoped to the job id so a stale worker cannot
  overwrite a newer dispatch.

All four are conditional UPDATEs against the live row: the sweep's candidate
list is minutes old by the time they run, so the row itself is the only honest
arbiter. See services/lifecycle.py for the driver.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

_RECOVERY_ATTEMPTS_PATH = "'{evaluation_recovery,attempts}'"
# A legacy / hand-edited row could hold a non-numeric there. Mirror the Python
# reader's tolerance (``services.evaluation_state.recovery_attempts``) instead of
# letting one malformed row raise out of the sweep.
_RECOVERY_ATTEMPTS_INT = (
    f"CASE WHEN jsonb_typeof(internal_summary_json #> {_RECOVERY_ATTEMPTS_PATH}) = 'number' "
    f"     THEN (internal_summary_json #>> {_RECOVERY_ATTEMPTS_PATH})::int "
    "     ELSE 0 END"
)

# SQL that reads the durable recovery phase (``evaluation_recovery.current.phase``)
# and answers "is the recorded job still working?". dispatching/queued/running/
# retrying are ACTIVE; succeeded/failed/missing (and no record at all) are not.
_ACTIVE_PHASES_SQL = (
    "jsonb_typeof(internal_summary_json #> '{evaluation_recovery,current,phase}') = 'string' "
    "AND internal_summary_json #>> '{evaluation_recovery,current,phase}' "
    "    IN ('dispatching', 'queued', 'running', 'retrying')"
)
_ACTIVE_PHASES_SQL2 = _ACTIVE_PHASES_SQL


async def stamp_evaluation_recovery_attempt(
    db: AsyncSession,
    session_id: UUID,
    *,
    now: datetime,
    job_id: str | None = None,
) -> int | None:
    """Charge one recovery attempt to this session. Returns the NEW attempt count.

    ``None`` means nothing was charged. Every refusal is decided INSIDE the
    UPDATE so a stale candidate list cannot charge a session that changed:

    * a verdict was published between the candidate query and this call — the
      session is graded, there is nothing to repair;
    * a LIVE claim is held — a grader owns the session right now and a second
      one would race it for the same transcript;
    * a durable ACTIVE recovery job exists (``current.phase`` in
      dispatching/queued/running/retrying) — the last re-drive never settled,
      so the attempt is not the sweep's to spend.

    Why this is SQL and not a Python read-modify-write on the ORM row: the sweep
    loads its candidates in one transaction and then works through them, so its
    ``internal_summary_json`` snapshot goes stale the moment the evaluator it is
    trying to repair publishes. Assigning a whole dict back would blind-overwrite
    the column and wipe the ``total_score`` / ``rubric_aggregated`` /
    ``evaluated_at`` the evaluator just wrote (and resurrect the stale
    ``evaluation_failure`` note it had cleared). ``jsonb_set`` touches ONLY the
    ``evaluation_recovery`` key, so bookkeeping and results never collide.

    Charging RESERVES the dispatch: ``current`` is written as ``dispatching``
    with the deterministic job id in the same update, so a crash between the
    charge and the enqueue leaves evidence rather than a silent attempt. The
    caller CASes the phase to ``queued`` once ARQ accepts the job.
    """

    # The only interpolation is _RECOVERY_ATTEMPTS_INT / _ACTIVE_PHASES_SQL /
    # _ACTIVE_PHASES_SQL2, module-level SQL constants. Every value (session id,
    # timestamp, job id) is a bound parameter.
    next_attempt = f"({_RECOVERY_ATTEMPTS_INT} + 1)"
    job_id_expr = (
        "CAST(:job_id AS text)"
        if job_id is not None
        else
        # Deterministic default: the sweep's per-attempt job id, so the record
        # is still meaningful when the caller does not pass one. The literal
        # separator is a bound parameter because ":recover" inside the SQL
        # string would be parsed as a bind parameter name.
        "'interview-evaluation:' || id::text "
        "|| CAST(:recover_sep AS text) || " + next_attempt
    )
    sql = text(
        "UPDATE interview_sessions "  # noqa: S608 -- only module-level SQL constants interpolated
        "   SET internal_summary_json = jsonb_set("
        "         COALESCE(internal_summary_json, '{}'::jsonb), "
        "         '{evaluation_recovery}', "
        "         COALESCE(internal_summary_json -> 'evaluation_recovery', '{}'::jsonb) "
        "           || jsonb_build_object("
        f"                'attempts', {next_attempt}, "
        # CAST is required: inside jsonb_build_object Postgres has no context to
        # infer a bare parameter's type and rejects the statement outright.
        # Two SEPARATE parameters for the same instant: the claim comparison
        # needs a timestamp the server can order against the timestamptz
        # column, and one parameter keeps one type across the whole statement —
        # so the two text copies get their own string binding, or the
        # timestamptz typing would leak into them ('2026-09-09 14:17:34+00'
        # instead of the ISO form the Python reader parses with fromisoformat).
        "                'last_attempt_at', CAST(:stamp_text AS text), "
        "                'current', jsonb_build_object("
        "                  'attempt', " + next_attempt + ", "
        "                  'job_id', " + job_id_expr + ", "
        "                  'phase', 'dispatching', "
        "                  'dispatched_at', CAST(:stamp_text AS text))), "
        "         true) "
        " WHERE id = :session_id "
        "   AND pass_verdict IS NULL "
        # The claim must be absent OR expired: a live claim means a grader owns
        # the session and the sweep must not start a second one against it.
        "   AND (evaluation_claim_token IS NULL "
        "        OR evaluation_claim_expires_at IS NULL "
        "        OR evaluation_claim_expires_at <= :now) "
        # A durable ACTIVE job refuses the charge; terminal phases do not.
        f"   AND NOT COALESCE(({_ACTIVE_PHASES_SQL}), false) "
        f"RETURNING {_RECOVERY_ATTEMPTS_INT} AS attempts"
    )
    params: dict[str, Any] = {
        "session_id": session_id,
        "now": now.isoformat(),
        "stamp_text": now.isoformat(),
        "job_id": job_id,
        "recover_sep": ":recover-",
    }
    attempts = (await db.execute(sql, params)).scalar_one_or_none()
    await db.commit()
    return int(attempts) if attempts is not None else None


async def refund_evaluation_recovery_attempt(
    db: AsyncSession,
    session_id: UUID,
    *,
    attempt: int,
    previous_recovery: dict[str, Any] | None,
) -> bool:
    """Give back the attempt charged by ``stamp_evaluation_recovery_attempt``.

    Used when the dispatch produced no job at all (Redis down, or ARQ refused the
    ID), where charging the budget would strand the session unevaluated.

    Restores ONLY the ``evaluation_recovery`` sub-object, and only while the count
    we wrote is still the current one. Both bounds matter:

    * the sub-object scope means a verdict published while the enqueue was in
      flight survives the refund — writing back a whole pre-enqueue snapshot of
      ``internal_summary_json`` deleted the results the evaluator had just
      committed;
    * the count guard means a later sweep that has already charged its own
      attempt is not silently un-charged. We then leave our attempt spent, which
      is the safe direction (bounded retries, not an infinite loop).
    """
    import json  # noqa: PLC0415


    # Same shape as stamp_evaluation_recovery_attempt: the interpolation is a
    # module-level SQL constant, not input.
    sql = text(
        "UPDATE interview_sessions "  # noqa: S608 -- SQL constant interpolation only, see stamp helper
        "   SET internal_summary_json = CASE "
        "         WHEN CAST(:previous_recovery AS jsonb) IS NULL "
        "           THEN internal_summary_json - 'evaluation_recovery' "
        "         ELSE jsonb_set(internal_summary_json, '{evaluation_recovery}', "
        # The refund also drops the reserved dispatch record: restoring the
        # pre-charge bookkeeping must not leave a 'current' pointing at a job
        # that never existed.
        "                        (CAST(:previous_recovery AS jsonb) - 'current'), true) "
        "       END "
        " WHERE id = :session_id "
        f"   AND {_RECOVERY_ATTEMPTS_INT} = :attempt "
        "RETURNING id"
    )
    refunded = (
        await db.execute(
            sql,
            {
                "session_id": session_id,
                "attempt": attempt,
                "previous_recovery": (
                    json.dumps(previous_recovery) if previous_recovery is not None else None
                ),
            },
        )
    ).scalar_one_or_none() is not None
    await db.commit()
    return refunded


async def transition_evaluation_recovery_phase(
    db: AsyncSession,
    session_id: UUID,
    *,
    job_id: str,
    from_phase: str,
    to_phase: str,
    extra: dict[str, Any] | None = None,
) -> bool:
    """CAS the durable recovery record along one job's lifecycle.

    ``from_phase`` must still be the CURRENT phase and ``job_id`` must still be
    the CURRENT job for the write to land. Both bounds stop a stale worker (an
    old sweep's task that woke up late, or a superseded job) from relabelling a
    NEWER dispatch's record — the same ownership discipline the evaluation
    claim applies to the verdict, applied to the bookkeeping.

    ``extra`` merges additional fields (``queued_at``, ``next_retry_at``,
    ``error_class``, ...) into ``current`` in the same write.
    """
    import json  # noqa: PLC0415


    extra_json = json.dumps(extra or {})
    sql = text(
        "UPDATE interview_sessions "
        "   SET internal_summary_json = jsonb_set("
        "         internal_summary_json, "
        "         '{evaluation_recovery,current}', "
        "         COALESCE("
        "           internal_summary_json #> '{evaluation_recovery,current}', '{}'::jsonb"
        "         ) "
        "           || CAST(:extra AS jsonb) "
        "           || jsonb_build_object('phase', CAST(:to_phase AS text))) "
        " WHERE id = :session_id "
        "   AND internal_summary_json #>> '{evaluation_recovery,current,phase}' = :from_phase "
        "   AND internal_summary_json #>> '{evaluation_recovery,current,job_id}' = :job_id "
        "RETURNING id"
    )
    updated = (
        await db.execute(
            sql,
            {
                "session_id": session_id,
                "job_id": job_id,
                "from_phase": from_phase,
                "to_phase": to_phase,
                "extra": extra_json,
            },
        )
    ).scalar_one_or_none()
    await db.commit()
    return updated is not None


__all__ = [
    "refund_evaluation_recovery_attempt",
    "stamp_evaluation_recovery_attempt",
    "transition_evaluation_recovery_phase",
]
