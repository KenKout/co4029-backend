"""Public evaluation state for one interview session.

The student-facing DTO used to expose only ``status`` + ``pass_verdict``, which
does not answer the one question every result screen asks: *is a verdict still
coming?* The frontend answered it by guessing —
``status in (completed, timed_out) and pass_verdict is null`` — and got it wrong
in both directions:

* ``status='failed'`` looked final. It is not: it means only that ARQ exhausted
  its retry budget, and ``recover_stalled_evaluations`` re-drives exactly those
  rows. The UI stopped polling and kept an error badge on a session that got a
  verdict thirty seconds later.
* nothing marked the point where the recovery budget runs out, so the naive fix
  (poll every ``failed`` row) would poll forever on genuinely dead sessions.

Both facts already exist server-side. This module derives the answer from them
so there is ONE definition, and exposes it as a small closed vocabulary. The
underlying ``internal_summary_json`` stays teacher-only: only the label crosses
the wire.

The DECISION inputs, in order of authority:

1. a published verdict → ``succeeded`` (``pass_verdict=False`` counts);
2. a durable ACTIVE recovery phase (``evaluation_recovery.current.phase`` in
   dispatching/queued/running/retrying) → ``pending`` — a timestamp heuristic
   cannot distinguish "queued for an hour behind a busy worker" from "dead",
   and guessing dead told the student no verdict was coming while the job sat
   in the queue;
3. a live claim → ``pending`` (direct evidence a grader holds the session);
4. the budget spent with every phase record terminal (succeeded/failed/missing)
   → ``exhausted``;
5. a legacy row at the ceiling with NO phase record stays ``pending`` until the
   lifecycle lazy-reconciles it against ARQ — "unknown" must never be read as
   "dead".
"""

from __future__ import annotations

from typing import Any, Literal

from abridgeai.features.interviews.services.evaluation_claim import EVALUATION_LEASE_SECONDS

# Keep in step with ``recover_stalled_evaluations(max_recovery_attempts=...)``
# and the SQL-side ceiling in ``list_pending_evaluation_sessions``. At this many
# attempts the sweep no longer selects the row, so nothing will re-drive it.
MAX_EVALUATION_RECOVERY_ATTEMPTS = 3

# Kept only for backwards compatibility with the ordering test that pins it
# against the claim lease and ARQ's job timeout. The public-state decision no
# longer consults a dispatch age: the durable phase is the authority.
FINAL_ATTEMPT_SETTLE_SECONDS = EVALUATION_LEASE_SECONDS

# The ACTIVE phases of a durable recovery record: the recorded job has been
# dispatched and has not reached a terminal state yet.
ACTIVE_PHASES = ("dispatching", "queued", "running", "retrying")
TERMINAL_PHASES = ("succeeded", "failed", "missing")

EvaluationState = Literal[
    "not_required",
    "pending",
    "succeeded",
    "exhausted",
]

_TERMINAL_GRADEABLE_STATUSES = ("completed", "timed_out", "failed")


def _recovery_metadata(session: object) -> dict[str, Any]:
    summary = getattr(session, "internal_summary_json", None) or {}
    recovery = summary.get("evaluation_recovery")
    return recovery if isinstance(recovery, dict) else {}


def recovery_attempts(session: object) -> int:
    """Recovery attempts already spent on this session's evaluation."""
    try:
        return int(_recovery_metadata(session).get("attempts") or 0)
    except (TypeError, ValueError):
        return 0


def _current_phase(session: object) -> str | None:
    """The durable phase of the most recent recovery dispatch, if recorded."""
    current = _recovery_metadata(session).get("current")
    if not isinstance(current, dict):
        return None
    phase = current.get("phase")
    return phase if isinstance(phase, str) else None


def derive_evaluation_state(session: object) -> EvaluationState:
    """Whether a verdict exists, is still coming, or will never come.

    * ``succeeded`` — a verdict is published. ``pass_verdict=False`` counts:
      the grader ran to completion and made a judgement.
    * ``pending`` — terminal + ungraded, and the work is not proven over: a
      durable ACTIVE recovery phase, a live claim, or budget left for the
      sweep to pick the session up with.
    * ``exhausted`` — terminal + ungraded, the budget is spent, AND every
      phase record is terminal (or the record was reconciled away). No sweep
      will select it again, so a reader must stop waiting.
    * ``not_required`` — nothing to wait for: still live, ``abandoned`` (no
      gradeable answer), or never reached the assessment (the same refusal
      ``services.evaluation._ungradeable_reason`` applies).
    """
    if getattr(session, "pass_verdict", None) is not None:
        return "succeeded"

    status = getattr(session, "status", None)
    if status not in _TERMINAL_GRADEABLE_STATUSES:
        return "not_required"
    if getattr(session, "assessment_started_at", None) is None:
        return "not_required"

    if recovery_attempts(session) >= MAX_EVALUATION_RECOVERY_ATTEMPTS:
        return "pending" if _final_job_may_still_be_working(session) else "exhausted"
    return "pending"


def _final_job_may_still_be_working(session: object) -> bool:
    """Is the last dispatched job alive, or at least not proven terminal?

    Reads the DURABLE evidence, never a clock: an active phase record means the
    job is still working no matter how long it has been; a live claim is direct
    evidence a grader holds the session. A legacy row with no record (or an
    unrecognised one) conservatively stays pending until the lifecycle
    reconciles it against ARQ.
    """
    phase = _current_phase(session)
    if phase in ACTIVE_PHASES:
        return True
    if phase in TERMINAL_PHASES:
        return False

    # No usable phase record. A live claim is still direct evidence a grader
    # holds the session. Without one the record may be legacy (the lazy
    # reconcile has not reached it) — "unknown" stays pending rather than
    # being read as dead.
    from datetime import datetime  # noqa: PLC0415

    from abridgeai.core.security import utcnow  # noqa: PLC0415

    lease_expires_at = getattr(session, "evaluation_claim_expires_at", None)
    if isinstance(lease_expires_at, datetime) and lease_expires_at > utcnow():
        # A live claim: a grader holds the session right now.
        return True
    # Nothing proves the job terminal and nothing proves it alive. Until the
    # lifecycle reconciles this legacy row against ARQ, "unknown" stays
    # pending — declaring it dead is the exact bug this module exists to stop.
    return True


__all__ = [
    "FINAL_ATTEMPT_SETTLE_SECONDS",
    "MAX_EVALUATION_RECOVERY_ATTEMPTS",
    "ACTIVE_PHASES",
    "TERMINAL_PHASES",
    "EvaluationState",
    "derive_evaluation_state",
    "recovery_attempts",
]
