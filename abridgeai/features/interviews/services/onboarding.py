"""Persisted, ungraded candidate onboarding before interview question one."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import TYPE_CHECKING

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from abridgeai.core.exceptions import AppError, ForbiddenError, NotFoundError
from abridgeai.core.security import CurrentUser, utcnow
from abridgeai.features.interviews.models import InterviewSession, InterviewSessionMessage
from abridgeai.features.interviews.services.ceremony import (
    CeremonyKind,
    ask_preferred_name_text,
    ensure_ceremony_message,
    normalize_language,
    onboarding_ceremony_kind,
    onboarding_retry_text,
    preferred_name_ack_text,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


@dataclass(frozen=True)
class OnboardingResult:
    session: InterviewSession
    ai_text: str | None


def _normalized_words(text: str | None) -> str:
    value = unicodedata.normalize("NFKD", (text or "").casefold())
    value = "".join(char for char in value if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]+", " ", value).strip()


def _language_choice(text: str | None) -> str | None:
    value = f" {_normalized_words(text)} "
    if any(token in value for token in (" vietnamese ", " tieng viet ", " viet nam ")):
        return "vi"
    if any(token in value for token in (" english ", " tieng anh ")):
        return "en"
    return None


def _natural_decision(text: str | None, stage: str | None = None) -> str:
    value = f" {_normalized_words(text)} "
    negative = (
        " not ready ",
        " not yet ",
        " cannot hear ",
        " cant hear ",
        " can t hear ",
        " no audio ",
        " need a moment ",
        " wrong name ",
        " isnt my name ",
        " chua san sang ",
        " khong nghe ",
        " can mot chut ",
        " sai ten ",
        " khong phai ten ",
    )
    if any(token in value for token in negative):
        return "hold"
    if stage == "language_check" and _language_choice(text) is not None:
        return "advance"
    positive = (
        " yes ",
        " ready ",
        " audio is clear ",
        " hear you ",
        " thats me ",
        " my name is ",
        " i am ",
        " all good ",
        " okay ",
        " ok ",
        " continue ",
        " no adjustment ",
        " san sang ",
        " nghe ro ",
        " dung roi ",
        " toi la ",
        " tiep tuc ",
        " vang ",
        " duoc ",
        " tot ",
    )
    return "advance" if any(token in value for token in positive) else "unclear"


async def _message_for_turn_key(
    db: AsyncSession, session_id: object, turn_key: str
) -> InterviewSessionMessage | None:
    result = await db.execute(
        select(InterviewSessionMessage).where(
            InterviewSessionMessage.session_id == session_id,
            InterviewSessionMessage.role == "user",
        )
    )
    for message in result.scalars().all():
        if (message.metadata_json or {}).get("turn_key") == turn_key:
            return message
    return None


async def _current_ai_text(db: AsyncSession, session: InterviewSession) -> str | None:
    kind: CeremonyKind = onboarding_ceremony_kind(session.onboarding_stage)
    message = await ensure_ceremony_message(
        db, session=session, kind=kind, language=session.interview_language
    )
    return message.content_text


def _guided_text(action: str | None, language: str) -> str:
    vi = language == "vi"
    values = {
        "confirm_identity": "Đúng là tôi." if vi else "Yes, that’s me.",
        "audio_clear": "Tôi nghe rõ." if vi else "The audio is clear.",
        "confirm_language": (
            "Tôi muốn tiếp tục bằng tiếng Việt." if vi else "I’d like to continue in English."
        ),
        "continue_setup": "Tôi muốn tiếp tục." if vi else "I’m ready to continue.",
        # Keep the previous combined action readable for older clients.
        "confirm_setup": "Đúng, âm thanh rõ ràng." if vi else "Yes, that’s me. The audio is clear.",
        "needs_adjustment": (
            "Tôi cần thêm một chút thời gian." if vi else "I need a moment to adjust my audio."
        ),
        "ready": "Tôi đã sẵn sàng bắt đầu." if vi else "I’m ready to begin.",
        "not_ready": "Tôi chưa sẵn sàng." if vi else "I’m not ready yet.",
        "skip_setup": "Bỏ qua phần thiết lập." if vi else "Skip the setup.",
    }
    return values.get(action or "", "")


async def respond(  # noqa: C901 - explicit persisted state machine
    db: AsyncSession,
    *,
    session_id: object,
    actor: CurrentUser,
    stage: str,
    response_text: str | None,
    action: str | None,
    language: str | None,
    turn_key: str,
) -> OnboardingResult:
    """Advance setup exactly once; onboarding turns are never question-linked."""
    session = await db.get(InterviewSession, session_id)
    if session is None:
        raise NotFoundError(f"Interview session {session_id} not found")
    if session.student_id != actor.user_id:
        raise ForbiddenError("Session does not belong to the calling user")
    if session.status != "in_progress":
        raise AppError("Interview session is not in progress")
    if await _message_for_turn_key(db, session.id, turn_key) is not None:
        return OnboardingResult(session=session, ai_text=await _current_ai_text(db, session))
    if session.onboarding_stage == "completed":
        return OnboardingResult(session=session, ai_text=None)
    if stage != session.onboarding_stage:
        raise AppError(f"Expected onboarding stage {session.onboarding_stage}")
    allowed_actions = {
        "identity_check": {
            "confirm_identity",
            "confirm_setup",
            "reject_identity",
            "set_name",
            "skip_setup",
        },
        "audio_check": {"audio_clear", "needs_adjustment", "skip_setup"},
        "language_check": {"confirm_language", "skip_setup"},
        "preparation": {"continue_setup", "needs_adjustment", "skip_setup"},
        "readiness": {"ready", "not_ready"},
    }
    if action is not None and action not in allowed_actions[stage]:
        raise AppError(f"Action {action} is not valid during {stage}")

    if language is not None and stage == "language_check":
        session.interview_language = normalize_language(language)
    lang = session.interview_language
    text = (response_text or "").strip() or _guided_text(action, lang)
    if not text:
        raise AppError("An onboarding response is required")

    selected_language = _language_choice(text) if stage == "language_check" else None
    if selected_language is not None:
        session.interview_language = selected_language
        lang = selected_language

    decision = _natural_decision(text, stage)
    if action in {
        "confirm_identity",
        "confirm_setup",
        "audio_clear",
        "confirm_language",
        "continue_setup",
        "ready",
        "skip_setup",
    }:
        decision = "advance"
    elif action in {"needs_adjustment", "not_ready", "reject_identity"}:
        decision = "hold"
    elif action == "set_name":
        decision = "advance"

    db.add(
        InterviewSessionMessage(
            session_id=session.id,
            session_question_id=None,
            role="user",
            content_text=text,
            metadata_json={
                "kind": "onboarding",
                "onboarding_stage": stage,
                "action": action,
                "decision": decision,
                "language": lang,
                "turn_key": turn_key,
            },
        )
    )
    try:
        await db.flush()
    except IntegrityError:
        await db.rollback()
        session = await db.get(InterviewSession, session_id)
        if session is None:
            raise NotFoundError(f"Interview session {session_id} not found") from None
        return OnboardingResult(session=session, ai_text=await _current_ai_text(db, session))

    if decision != "advance":
        if action == "reject_identity":
            retry = ask_preferred_name_text(language=lang)
        else:
            retry = onboarding_retry_text(stage=stage, language=lang)
        db.add(
            InterviewSessionMessage(
                session_id=session.id,
                session_question_id=None,
                role="ai",
                content_text=retry,
                metadata_json={
                    "kind": "onboarding",
                    "onboarding_stage": stage,
                    "response_kind": decision,
                    "language": lang,
                    **({"awaiting": "preferred_name"} if action == "reject_identity" else {}),
                },
            )
        )
        await db.flush()
        return OnboardingResult(session=session, ai_text=retry)

    # The candidate provided their preferred name: persist it (session-scoped,
    # never written to their profile) and acknowledge before advancing.
    ack_text: str | None = None
    if action == "set_name":
        session.preferred_name = text[:60]
        ack_text = preferred_name_ack_text(name=session.preferred_name, language=lang)

    # "skip_setup" fast-forwards past the remaining setup steps straight to the
    # readiness briefing. It stops AT readiness (never past it): the assessed
    # timer only starts when the candidate explicitly confirms "ready".
    if action == "skip_setup":
        next_stage: str | None = "readiness"
    else:
        next_stage = {
            "identity_check": "audio_check",
            "audio_check": "language_check",
            "language_check": "preparation",
            "preparation": "readiness",
        }.get(stage)
    if next_stage is not None:
        session.onboarding_stage = next_stage
        message = await ensure_ceremony_message(
            db,
            session=session,
            kind=onboarding_ceremony_kind(next_stage),
            language=lang,
        )
        # When the candidate just set their name, the acknowledgement already
        # carries the audio-check question, so surface it instead of the
        # generic audio-check ceremony text (the ceremony row still persists in
        # the transcript for a consistent canonical record).
        return OnboardingResult(session=session, ai_text=ack_text or message.content_text)

    session.onboarding_stage = "completed"
    if session.assessment_started_at is None:
        session.assessment_started_at = utcnow()
    message = await ensure_ceremony_message(
        db, session=session, kind="ready_transition", language=lang
    )
    return OnboardingResult(session=session, ai_text=message.content_text)


__all__ = ["OnboardingResult", "respond"]
