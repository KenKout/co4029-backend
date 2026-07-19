"""DB-touching security-guard handler for the interview flow (Phase S1).

The pure detection logic lives in :mod:`orchestrator.security_logic` and the
types/templates in :mod:`orchestrator.security`. This module owns the *I/O* —
loading/saving runtime-state counters, persisting the student utterance + safe
AI response, and emitting observability events — so ``services.taking`` can
delegate the whole guard step with one call and stay under the god-file cap.

Mirrors the ``intent`` / ``intent_logic`` and ``analysis`` / ``analysis_logic``
split already used across the orchestrator.

Contract (see ``handle_security_guard``):
* READ-ONLY assessment: merely assessing a turn NEVER materializes a
  runtime-state row (protects the legacy/flag-off path). State is created only
  on the enforce+blocking path where counters must persist for escalation.
* Best-effort: any guard failure returns ``None`` (the interview proceeds); the
  output-leakage guard in a later slice backstops the fail-open direction.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from abridgeai.core.db.conflict_mapper import flush_or_conflict
from abridgeai.features.interviews.models import InterviewSessionMessage
from abridgeai.features.interviews.orchestrator import repository as state_repo
from abridgeai.features.interviews.orchestrator import security_logic
from abridgeai.features.interviews.orchestrator.security import (
    ACTION_RESPONSE_KEY,
    BLOCKING_ACTIONS,
    SecurityAction,
    safe_response_text,
)
from abridgeai.features.interviews.realtime import observability as obs

if TYPE_CHECKING:
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession

    from abridgeai.features.interviews.models import (
        InterviewSession,
        InterviewSessionQuestion,
    )

logger = logging.getLogger(__name__)


def _add_student_answer(
    db: AsyncSession,
    session_id: UUID,
    session_question_id: UUID,
    answer_text: str,
    audio_object_id: UUID | None,
) -> None:
    """Persist the student-answer row (kind='answer'); does not flush."""
    db.add(
        InterviewSessionMessage(
            session_id=session_id,
            session_question_id=session_question_id,
            role="user",
            content_text=answer_text,
            audio_object_id=audio_object_id,
            metadata_json={"kind": "answer"},
        )
    )


async def _persist_counters(
    db: AsyncSession,
    session_id: UUID,
    *,
    prior_consecutive: int,
    category_value: str,
    action: SecurityAction,
) -> None:
    """Persist bounded, non-reversible security counters (best-effort).

    Uses ``load_or_init`` (not the read-only load) because a blocked turn DOES
    want counters to persist across turns for escalation — creating the row is
    justified here. This path returns before the normal pipeline runs, so this
    is the sole state write for the turn (no double version bump).
    """
    try:
        to_write = await state_repo.load_or_init(db, session_id)
        data = to_write.data
        data.security_attempt_count = int(getattr(data, "security_attempt_count", 0) or 0) + 1
        data.consecutive_security_attempts = prior_consecutive + 1
        data.last_security_category = category_value
        if action is SecurityAction.WARN_AND_REDIRECT:
            data.security_warning_issued = True
        if action is SecurityAction.END_AND_FLAG:
            data.session_security_flagged = True
        await state_repo.save(db, session_id, data, expected_version=to_write.version)
    except Exception:  # noqa: BLE001 — counters are advisory; never break the turn
        logger.exception("security: failed to persist counters (session=%s)", session_id)


async def maybe_security_block(
    db: AsyncSession,
    *,
    session: InterviewSession,
    current_session_question: InterviewSessionQuestion,
    answer_text: str,
    audio_object_id: UUID | None,
    language: str,
) -> dict[str, Any] | None:
    """Guard entry point that owns its own activation gating.

    Reads the operations-only ``interview_security_guard_mode``: a no-op (returns
    ``None``) when the guard is ``off``; otherwise assesses the turn (``shadow``
    observes only, ``enforce`` may block). Keeping the gating here lets the
    caller in ``services.taking`` stay a single unconditional call.
    """
    from abridgeai.core.config import get_settings  # noqa: PLC0415

    settings = get_settings()
    if not settings.security_guard_active():
        return None
    return await handle_security_guard(
        db,
        session=session,
        current_session_question=current_session_question,
        answer_text=answer_text,
        audio_object_id=audio_object_id,
        language=language,
        enforcing=settings.security_guard_enforcing(),
    )


async def handle_security_guard(
    db: AsyncSession,
    *,
    session: InterviewSession,
    current_session_question: InterviewSessionQuestion,
    answer_text: str,
    audio_object_id: UUID | None,
    language: str,
    enforcing: bool,
) -> dict[str, Any] | None:
    """Assess the utterance for prompt injection; act only when enforcing.

    Returns a legacy-shaped result dict when the turn is BLOCKED (enforce mode +
    a detected attack whose action blocks the academic pipeline): the student
    answer is recorded (faithful transcript), a deterministic safe AI response
    is persisted + returned, and NO academic analysis / evidence / follow-up
    runs. Returns ``None`` when the turn should proceed to the normal pipeline —
    benign input, a non-blocking action, or shadow mode (assess + observe only).
    """
    session_id = session.id
    turn_id = str(current_session_question.id)

    # READ-ONLY load: assessing a turn must never materialize a runtime-state
    # row (that would regress the legacy/flag-off path, which otherwise has no
    # state row). We create/write state only in the enforce+blocking path below.
    try:
        loaded = await state_repo.load_readonly(db, session_id)
        prior_consecutive = (
            int(getattr(loaded.data, "consecutive_security_attempts", 0) or 0)
            if loaded is not None
            else 0
        )
    except Exception:  # noqa: BLE001 — never break the interview on a state read
        prior_consecutive = 0

    # Slice 1 does not auto-end sessions; END_AND_FLAG stays behind platform
    # policy (a later slice wires the teacher-configurable end toggle).
    assessment = security_logic.assess_security(
        answer_text, consecutive_attempts=prior_consecutive, allow_end=False
    )

    # Always emit the assessment (shadow gives false-positive signal before
    # enforce ever blocks anyone). Privacy-safe: no raw content.
    obs.emit(
        obs.EV_SECURITY_ASSESSED,
        session_id=session_id,
        turn_id=turn_id,
        enforcing=enforcing,
        **assessment.to_dict(),
    )

    if not assessment.detected:
        return None

    action = security_logic.choose_action(
        assessment.category, consecutive_attempts=prior_consecutive + 1, allow_end=False
    )

    if not enforcing or action not in BLOCKING_ACTIONS:
        # Shadow mode or a non-blocking action: observe only, run normal
        # pipeline. Deliberately NO state write here — the pipeline's own save
        # owns the version, and writing now would double-bump state_version.
        return None

    # ── ENFORCE + blocking action: block the academic pipeline this turn ──────
    await _persist_counters(
        db,
        session_id,
        prior_consecutive=prior_consecutive,
        category_value=assessment.category.value,
        action=action,
    )

    obs.emit(
        obs.EV_SECURITY_BLOCKED,
        session_id=session_id,
        turn_id=turn_id,
        category=assessment.category.value,
        action=action.value,
        attempt_count=prior_consecutive + 1,
    )

    # Record the student's utterance (faithful transcript) but never analyze it.
    _add_student_answer(db, session_id, current_session_question.id, answer_text, audio_object_id)
    await flush_or_conflict(db)

    response_key = ACTION_RESPONSE_KEY.get(action)
    safe_text = safe_response_text(response_key, language)
    ends_session = action is SecurityAction.END_AND_FLAG

    # Persist the safe AI response as an AI turn (kind='security' so it is
    # distinguishable from academic follow-ups and never counts toward them).
    db.add(
        InterviewSessionMessage(
            session_id=session_id,
            session_question_id=current_session_question.id,
            role="ai",
            content_text=safe_text,
            metadata_json={
                "kind": "security",
                "action": action.value,
                "category": assessment.category.value,
            },
        )
    )
    await flush_or_conflict(db)

    if ends_session:
        obs.emit(
            obs.EV_SECURITY_SESSION_FLAGGED,
            session_id=session_id,
            turn_id=turn_id,
            attempt_count=prior_consecutive + 1,
        )

    # Legacy-shaped result: the safe text rides in followup_text (and
    # ai_turn_text for adaptive clients). No next_question — the student stays on
    # the current question (redirect). is_finished only when policy ended it.
    return {
        "next_question": None,
        "is_finished": ends_session,
        "followup_text": safe_text,
        "ai_turn_text": safe_text,
        "should_narrate": True,
        "should_await_response": not ends_session,
        "should_finish": ends_session,
        "language": language,
        "security_blocked": True,
    }


__all__ = ["handle_security_guard"]
