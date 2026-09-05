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
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Literal

from abridgeai.core.security import utcnow
from abridgeai.features.interviews.services.evaluation_claim import EVALUATION_LEASE_SECONDS

# Keep in step with ``recover_stalled_evaluations(max_recovery_attempts=...)``
# and the SQL-side ceiling in ``list_pending_evaluation_sessions``. At this many
# attempts the sweep no longer selects the row, so nothing will re-drive it.
MAX_EVALUATION_RECOVERY_ATTEMPTS = 3

# How long the job that consumed the LAST attempt may still be working before we
# are willing to call a session dead. The counter is charged BEFORE the enqueue
# (deliberately: a job killed mid-run must still spend its budget), so hitting
# the ceiling means "the final job was dispatched", not "the final job finished".
# The lease ceiling is the right bound: it is already longer than
# ``WorkerSettings.job_timeout``, so no live job can outlast it.
FINAL_ATTEMPT_SETTLE_SECONDS = EVALUATION_LEASE_SECONDS

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


def _last_attempt_at(session: object) -> datetime | None:
    """When the most recent recovery attempt was dispatched, if it is known."""
    raw = _recovery_metadata(session).get("last_attempt_at")
    if isinstance(raw, datetime):
        return raw
    if not isinstance(raw, str):
        return None
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return None


def _final_attempt_may_still_be_running(session: object) -> bool:
    """Is the job that consumed the last attempt plausibly still working?

    The attempt counter is charged BEFORE the enqueue, so reaching the ceiling
    only proves the final job was dispatched. Calling that ``exhausted`` told the
    student "no verdict is coming" while their last grading job was still queued
    or mid-run: the UI stopped polling and they never saw the result that landed
    a minute later.

    A live claim is direct evidence a job holds the session. Otherwise fall back
    to the dispatch timestamp: within the settle window the job may not have
    claimed yet (still queued behind other work).
    """
    now = utcnow()
    lease_expires_at = getattr(session, "evaluation_claim_expires_at", None)
    if isinstance(lease_expires_at, datetime) and lease_expires_at > now:
        return True

    last_attempt_at = _last_attempt_at(session)
    if last_attempt_at is None:
        # No dispatch timestamp (pre-existing row, or hand-edited metadata) —
        # nothing suggests a live job, so do not hold the UI open indefinitely.
        return False
    if last_attempt_at.tzinfo is None:
        # Everything we write is aware (``utcnow().isoformat()``); a naive value
        # can only be legacy/hand-edited data. Read it as UTC rather than raising
        # a TypeError out of a student-facing response serializer.
        last_attempt_at = last_attempt_at.replace(tzinfo=now.tzinfo)
    return now - last_attempt_at < timedelta(seconds=FINAL_ATTEMPT_SETTLE_SECONDS)


def derive_evaluation_state(session: object) -> EvaluationState:
    """Whether a verdict exists, is still coming, or will never come.

    * ``succeeded`` — a verdict is published. ``pass_verdict=False`` counts:
      the grader ran to completion and made a judgement.
    * ``pending`` — terminal + ungraded, and either a job is still working or
      the recovery sweep can still pick it up. Includes ``status='failed'``, and
      includes a session at the attempt ceiling whose FINAL job has not settled
      yet: the counter is charged before the enqueue, so the ceiling alone does
      not mean the last grading run is over.
    * ``exhausted`` — terminal + ungraded, the recovery budget is spent AND the
      job that spent it is no longer running. No sweep will select it again, so
      a reader must stop waiting.
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
        return "pending" if _final_attempt_may_still_be_running(session) else "exhausted"
    return "pending"


__all__ = [
    "FINAL_ATTEMPT_SETTLE_SECONDS",
    "MAX_EVALUATION_RECOVERY_ATTEMPTS",
    "EvaluationState",
    "derive_evaluation_state",
    "recovery_attempts",
]
