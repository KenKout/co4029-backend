"""Spaced-repetition domain events.

Lightweight, in-process domain events emitted by the card-review service.
For now (T7.5.5) we only emit a structured log line via :mod:`logging`;
T7.5.10 will subscribe a real notification dispatcher to ``CardFailedEvent``.

Keeping the event contract here lets the wiring work happen later without
churning the review-service public API.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CardFailedEvent:
    """Emitted when a card review yields ``q == 0`` (full failure)."""

    student_id: UUID
    question_id: UUID
    quiz_attempt_id: UUID | None
    quiz_id: UUID
    timestamp: datetime


async def emit_card_failed(
    db: AsyncSession,  # noqa: ARG001 — kept for parity with future dispatcher
    event: CardFailedEvent,
) -> None:
    """Emit ``CardFailedEvent`` (currently log-only).

    T7.5.10 wires this to the notifications dispatcher; until then the
    structlog-compatible payload is enough to verify event emission in
    tests and to give SREs a paper trail.
    """
    payload = {k: str(v) if v is not None else None for k, v in asdict(event).items()}
    logger.info("sr.card_failed", extra={"event": "sr.card_failed", **payload})


__all__ = ["CardFailedEvent", "emit_card_failed"]
