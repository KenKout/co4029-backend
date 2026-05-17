"""Spaced-repetition domain events.

Lightweight, in-process domain events emitted by the card-review service.

T7.5.10 + BUG-2 fix
-------------------
Previously, ``emit_card_failed`` ran inside the same flush as the
``CardReview`` insert. That allowed a "ghost notification" race: if the
caller's enclosing transaction rolled back after flush (e.g. raised
inside the same ``async with session.begin():`` block), the structured
log line and any future side-effects would still have fired even though
no ``CardReview`` row was ever durable.

The fix is the **caller-dispatches-after-commit** pattern:

* :class:`CardFailedEvent` remains the value object.
* The card-review service no longer fires anything itself; it appends
  the event to :attr:`CardReviewResult.pending_events`.
* The caller -- typically the quiz/router layer -- is expected to
  ``await db.commit()`` first and then iterate over ``pending_events``
  to dispatch
  :func:`abridgeai.features.spaced_repetition.services.remediation.dispatch_remediation_for_card_failure`.

If the caller never commits (or commits and then crashes before
dispatching), the worst case is a missed notification, which is strictly
preferable to a ghost notification for a non-existent review.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID


@dataclass(frozen=True)
class CardFailedEvent:
    """Emitted when a card review yields ``q == 0`` (full failure)."""

    student_id: UUID
    question_id: UUID
    quiz_attempt_id: UUID | None
    quiz_id: UUID
    timestamp: datetime


__all__ = ["CardFailedEvent"]
