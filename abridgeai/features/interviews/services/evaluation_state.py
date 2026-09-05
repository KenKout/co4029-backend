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

from typing import Literal

# Keep in step with ``recover_stalled_evaluations(max_recovery_attempts=...)``
# and the SQL-side ceiling in ``list_pending_evaluation_sessions``. At this many
# attempts the sweep no longer selects the row, so nothing will re-drive it.
MAX_EVALUATION_RECOVERY_ATTEMPTS = 3

EvaluationState = Literal[
    "not_required",
    "pending",
    "succeeded",
    "exhausted",
]

_TERMINAL_GRADEABLE_STATUSES = ("completed", "timed_out", "failed")


def recovery_attempts(session: object) -> int:
    """Recovery attempts already spent on this session's evaluation."""
    summary = getattr(session, "internal_summary_json", None) or {}
    recovery = summary.get("evaluation_recovery") or {}
    try:
        return int(recovery.get("attempts") or 0)
    except (TypeError, ValueError):
        return 0


def derive_evaluation_state(session: object) -> EvaluationState:
    """Whether a verdict exists, is still coming, or will never come.

    * ``succeeded`` — a verdict is published. ``pass_verdict=False`` counts:
      the grader ran to completion and made a judgement.
    * ``pending`` — terminal + ungraded, and either a job is still working or
      the recovery sweep can still pick it up. Includes ``status='failed'``.
    * ``exhausted`` — terminal + ungraded with the recovery budget spent. No
      sweep will select it again, so a reader must stop waiting.
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
        return "exhausted"
    return "pending"


__all__ = [
    "MAX_EVALUATION_RECOVERY_ATTEMPTS",
    "EvaluationState",
    "derive_evaluation_state",
    "recovery_attempts",
]
