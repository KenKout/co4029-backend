"""Assessment integrity-event ingest DTOs (Phase 4).

Client (Phase 5) batches browser activity signals — tab switch, focus loss,
fullscreen exit, (re)connect — and POSTs them for the owning in-progress
session. Events are post-session / teacher review only; never surfaced to the
student. ``event_type`` / ``severity`` literals match the DB CHECK constraints
on ``assessment_integrity_events``.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

IntegrityEventTypeLiteral = Literal[
    "focus_lost",
    "tab_switch",
    "fullscreen_exit",
    "warning_issued",
    "reconnect",
    "disconnect",
]
IntegritySeverityLiteral = Literal["info", "warning", "critical"]

# Cap a single batch so a chatty / malicious client can't flood the table.
MAX_EVENTS_PER_BATCH = 50


class IntegrityEventItem(BaseModel):
    event_type: IntegrityEventTypeLiteral
    severity: IntegritySeverityLiteral = "info"
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)


class IntegrityEventBatchRequest(BaseModel):
    events: list[IntegrityEventItem] = Field(min_length=1, max_length=MAX_EVENTS_PER_BATCH)


__all__ = [
    "IntegrityEventBatchRequest",
    "IntegrityEventItem",
    "IntegritySeverityLiteral",
    "IntegrityEventTypeLiteral",
    "MAX_EVENTS_PER_BATCH",
]
