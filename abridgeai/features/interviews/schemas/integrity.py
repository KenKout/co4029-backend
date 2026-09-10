"""Assessment integrity-event ingest DTOs + server-side scoring rules (Phase 4).

Client (Phase 5) batches browser activity signals — tab switch, focus loss,
fullscreen exit, (re)connect — and POSTs them for the owning in-progress
session. Events are recorded for post-session / teacher review. The SERVER
scores the three browser signals against the per-session policy snapshot
(weights 1..5 + threshold, frozen by ``services.taking.start_session``);
client-supplied metadata and severities are never trusted to influence the
score. ``warning_issued``, ``reconnect`` and ``disconnect`` never score.
``event_type`` / ``severity`` literals match the DB CHECK constraints on
``assessment_integrity_events``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal
from uuid import UUID

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

# The scored browser signals and their DEFAULT weights. The per-session
# snapshot (built from the config in ``taking.integrity_policy_snapshot``) is
# authoritative at scoring time; this map defines the shape and the shipped
# defaults (decision 2026-09-10). Server-only events (warning_issued) and
# connectivity records (reconnect/disconnect) are deliberately absent: they
# can never score, whatever a client claims.
DEFAULT_WEIGHTS: dict[str, int] = {
    "tab_switch": 3,
    "focus_lost": 1,
    "fullscreen_exit": 2,
}

# Canonical snapshot keys — the JSONB snapshot and every scoring lookup key by
# these spellings, so a renamed column cannot silently miss the snapshot.
SNAPSHOT_KEYS: dict[str, str] = {
    "tab_switch": "tab_switch",
    "focus_lost": "focus_lost",
    "fullscreen_exit": "fullscreen_exit",
    "score_threshold": "score_threshold",
}

_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)


class IntegrityEventItem(BaseModel):
    event_type: IntegrityEventTypeLiteral
    severity: IntegritySeverityLiteral = "info"
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)


class IntegrityEventBatchRequest(BaseModel):
    events: list[IntegrityEventItem] = Field(min_length=1, max_length=MAX_EVENTS_PER_BATCH)


class IntegrityEventBatchResponse(BaseModel):
    """Server-authoritative ingest result.

    ``warning_issued`` is True ONLY on the request whose events first crossed
    the threshold — a duplicate/retried crossing batch re-reads the persisted
    ``integrity_warning_issued`` flag and reports False. The client uses this
    to surface exactly one visible warning; it cannot forge one, because the
    flag lives on the server row.
    """

    accepted: int
    integrity_score: int
    integrity_score_threshold: int = 0
    warning_issued: bool = False


def is_scored_event(event_type: str) -> bool:
    """True for the three browser signals that carry weight; False otherwise."""
    return event_type in DEFAULT_WEIGHTS


def weight_for(event_type: str, policy: dict[str, int]) -> int:
    """The snapshot's weight for a scored event, clamped to 1..5.

    A malformed or missing snapshot key degrades to the shipped default rather
    than to 0 — a corrupt snapshot must never silently disable scoring (the
    same direction-safe reading as the cooldown downgrade rule).
    """
    key = SNAPSHOT_KEYS.get(event_type)
    if key is None:
        return 0
    raw = policy.get(key)
    try:
        value = int(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        value = DEFAULT_WEIGHTS[event_type]
    return min(5, max(1, value))


def add_weighted_score(current_score: int, event_type: str, policy: dict[str, int]) -> int:
    """One scored event's contribution. Unscored events add zero by contract."""
    return current_score + weight_for(event_type, policy)


@dataclass(frozen=True)
class ThresholdDecision:
    reaches_threshold: bool


def reaches_threshold(previous_score: int, weight: int, threshold: int) -> bool:
    """Would adding ``weight`` to ``previous_score`` reach ``threshold``?

    The pure pre/post helper the router uses to decide a crossing. Reaching is
    INCLUSIVE (``>=``): the threshold is the flag line, not the flag line + 1.
    """
    if threshold <= 0:
        return False
    return previous_score + weight >= threshold


def warn_once_and_score(score_after: int, threshold: int) -> ThresholdDecision:
    """Whether the NEW score crosses a positive threshold.

    Caller contract (enforced by the router): the persisted
    ``integrity_warning_issued`` flag is checked BEFORE this — a session that
    already warned never warns again, whatever the score does later.
    """
    return ThresholdDecision(reaches_threshold=bool(threshold > 0) and score_after >= threshold)


def coerce_client_event_id(raw: object) -> UUID | None:
    """Parse a client-supplied retry key, or None when absent/invalid.

    Anything that is not a well-formed UUID string becomes None (recorded
    unscored-key) rather than raising — a broken client must not 500 the
    best-effort ingest path.
    """
    if isinstance(raw, UUID):
        return raw
    if isinstance(raw, str) and _UUID_RE.match(raw.strip()):
        try:
            return UUID(raw.strip())
        except ValueError:  # pragma: no cover - regex already validated
            return None
    return None


__all__ = [
    "DEFAULT_WEIGHTS",
    "IntegrityEventBatchRequest",
    "IntegrityEventBatchResponse",
    "IntegrityEventItem",
    "IntegritySeverityLiteral",
    "IntegrityEventTypeLiteral",
    "MAX_EVENTS_PER_BATCH",
    "SNAPSHOT_KEYS",
    "ThresholdDecision",
    "add_weighted_score",
    "coerce_client_event_id",
    "is_scored_event",
    "reaches_threshold",
    "warn_once_and_score",
    "weight_for",
]
