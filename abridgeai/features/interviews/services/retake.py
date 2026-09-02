"""Interview retake policy: cooldown + attempt ceiling (FR-5.3).

Extracted from ``services.taking`` (2026-09-01) to keep that module under the
interviews LOC ratchet. ``taking`` re-imports the names so the public call
sites (``taking_service.compute_retake_status`` and the two exceptions caught
in ``routers.learner_sessions``) keep working unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING
from uuid import UUID

from abridgeai.core.exceptions import AppError
from abridgeai.features.interviews.queries import sessions as sessions_queries

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

# Retake-consuming terminal statuses — only these learner attempts use a
# configured attempt allowance and anchor the cooldown (FR-5.3). ``failed``
# is a system/evaluation failure and remains auditable without penalising the
# learner; ``in_progress`` is handled by the active-session short-circuit.
_RETAKE_CONSUMING_SESSION_STATUSES = ("completed", "timed_out", "abandoned")


class InterviewCooldownActive(AppError):  # noqa: N818  # mirrors quiz CooldownActive
    """FR-5.3 — student tried to start a new attempt inside the config's
    ``cooldown_minutes`` window since their last attempt ended.

    Carries ``retry_after`` so the router can surface a ``Retry-After``
    hint (mapped to HTTP 429). Mirrors the quiz-side
    :class:`~abridgeai.features.quizzes.queries.published.CooldownActive`.
    """

    def __init__(self, config_id: UUID, retry_after: datetime) -> None:
        super().__init__("Interview retake cooldown is still active")
        self.config_id = config_id
        self.retry_after = retry_after


class InterviewMaxAttemptsReached(AppError):  # noqa: N818  # mirrors quiz MaxAttemptsReached
    """FR-5.3 — student has used every attempt allowed by ``max_attempts``.

    Mapped to HTTP 409 by the router.
    """

    def __init__(self, config_id: UUID, max_attempts: int) -> None:
        super().__init__("Interview attempt limit reached")
        self.config_id = config_id
        self.max_attempts = max_attempts


@dataclass(frozen=True)
class RetakeStatus:
    """Proactive retake context for the student-facing results screen.

    Computed from the SAME FR-5.3 rules that gate ``start_session`` — exposed
    on the finish/session responses so the UI can show "N attempts left" and a
    cooldown countdown instead of only learning the ceiling reactively via
    429/409 when the student tries again.

    * ``remaining_attempts`` — ``None`` when ``max_attempts`` is unset.
    * ``retake_available_at`` — ``None`` when no cooldown is active (or unset).
    * ``can_retake`` — attempts remain AND no cooldown is currently blocking.
    """

    remaining_attempts: int | None
    retake_available_at: datetime | None
    can_retake: bool


async def compute_retake_status(
    db: AsyncSession,
    *,
    config_id: UUID,
    student_id: UUID,
) -> RetakeStatus:
    """Read-time retake status (mirrors ``_enforce_retake_policy`` rules)."""
    from abridgeai.features.interviews.models import InterviewConfig  # noqa: PLC0415

    config = await db.get(InterviewConfig, config_id)
    if config is None:
        return RetakeStatus(remaining_attempts=None, retake_available_at=None, can_retake=True)

    # Cooldown window — most recent retake-consuming attempt + cooldown_minutes.
    retake_available_at: datetime | None = None
    cooldown_minutes = config.cooldown_minutes
    if cooldown_minutes is not None and cooldown_minutes > 0:
        last_ended = await sessions_queries.get_last_terminal_ended_at(
            db, student_id, config_id, _RETAKE_CONSUMING_SESSION_STATUSES
        )
        if last_ended is not None:
            if last_ended.tzinfo is None:
                last_ended = last_ended.replace(tzinfo=UTC)
            candidate = last_ended + timedelta(minutes=cooldown_minutes)
            if candidate > datetime.now(UTC):
                retake_available_at = candidate

    # Attempt ceiling — max_attempts minus retake-consuming attempts used.
    remaining_attempts: int | None = None
    max_attempts = config.max_attempts
    if max_attempts is not None and max_attempts > 0:
        used = await sessions_queries.count_terminal_sessions(
            db, student_id, config_id, _RETAKE_CONSUMING_SESSION_STATUSES
        )
        remaining_attempts = max(0, max_attempts - used)

    has_attempts = remaining_attempts is None or remaining_attempts > 0
    can_retake = has_attempts and retake_available_at is None
    return RetakeStatus(
        remaining_attempts=remaining_attempts,
        retake_available_at=retake_available_at,
        can_retake=can_retake,
    )


async def _enforce_retake_policy(
    db: AsyncSession,
    *,
    config_id: UUID,
    student_id: UUID,
) -> None:
    """FR-5.3 — block a new attempt on ``max_attempts`` / ``cooldown_minutes``.

    Loads the config, then applies two gates against the student's
    *retake-consuming* sessions (live sessions never reach here). A ``failed``
    session is a system/evaluation failure, so it remains historical but never
    consumes an attempt or starts a cooldown:

    * **Cooldown** — if ``cooldown_minutes`` is set and the most recent
      retake-consuming attempt ended less than that many minutes ago, raise
      :class:`InterviewCooldownActive` (router → 429).
    * **Max attempts** — if ``max_attempts`` is set and the count of
      retake-consuming attempts already meet it, raise
      :class:`InterviewMaxAttemptsReached` (router → 409).

    NULL / non-positive knobs disable the corresponding gate. The gates
    are read-time only — no data migration normalises historical rows.
    """
    from abridgeai.features.interviews.models import InterviewConfig  # noqa: PLC0415

    config = await db.get(InterviewConfig, config_id)
    if config is None:
        return

    cooldown_minutes = config.cooldown_minutes
    if cooldown_minutes is not None and cooldown_minutes > 0:
        last_ended = await sessions_queries.get_last_terminal_ended_at(
            db, student_id, config_id, _RETAKE_CONSUMING_SESSION_STATUSES
        )
        if last_ended is not None:
            if last_ended.tzinfo is None:
                last_ended = last_ended.replace(tzinfo=UTC)
            retry_after = last_ended + timedelta(minutes=cooldown_minutes)
            if retry_after > datetime.now(UTC):
                raise InterviewCooldownActive(config_id, retry_after)

    max_attempts = config.max_attempts
    if max_attempts is not None and max_attempts > 0:
        used = await sessions_queries.count_terminal_sessions(
            db, student_id, config_id, _RETAKE_CONSUMING_SESSION_STATUSES
        )
        if used >= max_attempts:
            raise InterviewMaxAttemptsReached(config_id, max_attempts)


__all__ = [
    "InterviewCooldownActive",
    "InterviewMaxAttemptsReached",
    "RetakeStatus",
    "_RETAKE_CONSUMING_SESSION_STATUSES",
    "compute_retake_status",
]
