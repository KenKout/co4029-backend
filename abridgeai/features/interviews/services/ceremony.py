"""Deterministic opening and closing turns for learner interviews.

Ceremony turns are ordinary persisted AI messages.  Keeping them in the
transcript makes REST narration, LiveKit speech, retries, and audit views agree
on the exact words the candidate saw and heard.
"""

from __future__ import annotations

import re
from typing import Literal, cast

from sqlalchemy import select

from abridgeai.core.db.conflict_mapper import flush_or_conflict
from abridgeai.features.identity.models import UserProfile
from abridgeai.features.interviews.models import (
    InterviewConfig,
    InterviewSession,
    InterviewSessionMessage,
)
from abridgeai.features.interviews.orchestrator import interviewer_identity
from abridgeai.features.interviews.orchestrator.interviewer_identity import InterviewerIdentity
from abridgeai.features.interviews.orchestrator.persona import (
    OpeningStyle,
    profile_from_config,
)
from abridgeai.features.interviews.orchestrator.utterance import Persona, persona_from

CeremonyKind = Literal[
    "opening",
    "candidate_confirmation",
    "audio_check",
    "language_check",
    "preparation",
    "briefing",
    "ready_transition",
    "closing",
]
FinishReason = Literal["natural", "ended_early", "timed_out"]


def normalize_language(language: str | None) -> str:
    return "vi" if language and language.strip().lower().startswith("vi") else "en"


def _clean(value: str | None, *, limit: int) -> str:
    return re.sub(r"\s+", " ", value or "").strip()[:limit]


def candidate_first_name(profile: UserProfile | None) -> str | None:
    """Return a safe conversational first name, or ``None`` when unavailable."""
    if profile is None:
        return None
    given = _clean(profile.given_name, limit=60)
    if given:
        return given
    display = _clean(profile.display_name, limit=100)
    return display.split(" ", 1)[0] if display else None


def session_address_name(session: InterviewSession, profile: UserProfile | None) -> str | None:
    """Resolve how the interviewer should address the candidate this session.

    A session-scoped ``preferred_name`` (set when the candidate corrects the
    profile-derived name during identity_check) always wins; otherwise fall
    back to the profile's conversational first name.
    """
    preferred = _clean(getattr(session, "preferred_name", None), limit=60)
    if preferred:
        return preferred
    return candidate_first_name(profile)


def ask_preferred_name_text(*, language: str | None) -> str:
    """AI reply after the candidate says the profile name isn't theirs."""
    return (
        "Không sao. Vậy tôi nên gọi bạn là gì?"
        if normalize_language(language) == "vi"
        else "No problem. What should I call you?"
    )


def preferred_name_ack_text(*, name: str, language: str | None) -> str:
    """AI acknowledgement once the candidate provides their preferred name."""
    safe = _clean(name, limit=60)
    if normalize_language(language) == "vi":
        return f"Đã rõ, từ giờ tôi sẽ gọi bạn là {safe}. Bạn có nghe rõ tôi không?"
    return f"Got it — I’ll call you {safe} from now on. Can you hear me clearly?"


def _address(name: str | None, language: str) -> str:
    if not name:
        return ""
    return f" {name}" if language == "vi" else f", {name}"


def opening_text(
    *,
    title: str,
    name: str | None,
    persona: str | None,
    language: str | None,
    identity: InterviewerIdentity | None = None,
    opening_style: OpeningStyle | None = None,
) -> str:
    """Introduce the AI transparently, then ask only for candidate identity.

    ``identity`` gives the interviewer a name and professional title. It is
    ADDITIVE to the AI disclosure, never a replacement: the candidate is always
    told they are speaking to an AI interviewer. Presenting an AI as a human
    would be a consent problem, and the research on AI-conducted interviews
    finds that misleading candidates about the system's nature harms both
    perceived procedural fairness and the honesty of their answers.

    ``opening_style`` (a persona trait that was plumbed end-to-end but never
    consumed until now) tunes the ceremony around that introduction: ``BRIEF``
    drops the pleasantry, ``COMFORT`` adds one put-the-candidate-at-ease line.

    Both only take effect for a NAMED identity. A config that has not opted into
    an interviewer role returns exactly the pre-feature sentence, byte for byte,
    so existing interviews are untouched until a teacher chooses a role.

    ``persona`` remains unread here — tone is applied by the phrasing layer, not
    by this deterministic ceremony text.
    """
    del persona  # tone is the phrasing layer's business, not the ceremony's
    lang = normalize_language(language)
    safe_title = _clean(title, limit=180) or ("buổi phỏng vấn" if lang == "vi" else "interview")
    safe_name = _clean(name, limit=60) or None
    address = _address(safe_name, lang)
    who = identity or interviewer_identity.identity_from(None)
    style = opening_style or OpeningStyle.STANDARD

    if lang == "vi":
        named = f" bạn{address}" if address else " bạn"
        confirm = f"Trước tiên, bạn có thể xác nhận tôi đang trao đổi với{named} không?"
        if not who.is_named():
            return (
                f"Xin chào{address}. Rất vui được gặp bạn. Tôi là trợ lý phỏng vấn ảo của "
                f"aBridgeAI và sẽ hướng dẫn buổi phỏng vấn kỹ thuật “{safe_title}” hôm nay. "
                f"{confirm}"
            )
        parts = [f"Xin chào{address}."]
        if style is not OpeningStyle.BRIEF:
            parts.append("Rất vui được gặp bạn.")
        parts.append(
            f"Tôi là {who.name}, {who.title(lang)}, và hôm nay tôi là người phỏng vấn AI "
            f"của aBridgeAI cho buổi phỏng vấn kỹ thuật “{safe_title}”."
        )
        if style is OpeningStyle.COMFORT:
            parts.append(
                "Bạn cứ thoải mái, không cần vội — hãy dành thời gian cho mỗi câu trả lời."
            )
        parts.append(confirm)
        return " ".join(parts)

    named = f" {safe_name}" if safe_name else " you"
    confirm = f"First, can you confirm that I’m speaking with{named}?"
    if not who.is_named():
        return (
            f"Hi{address}. It’s nice to meet you. I’m aBridgeAI’s virtual interview "
            f"assistant, and I’ll guide your “{safe_title}” technical course interview "
            f"today. {confirm}"
        )
    parts = [f"Hi{address}."]
    if style is not OpeningStyle.BRIEF:
        parts.append("It’s nice to meet you.")
    parts.append(
        f"I’m {who.name}, a {who.title(lang)} — and today I’m aBridgeAI’s AI interviewer "
        f"for your “{safe_title}” technical course interview."
    )
    if style is OpeningStyle.COMFORT:
        parts.append("There’s no rush — take your time with each answer.")
    parts.append(confirm)
    return " ".join(parts)


def room_intro_text(
    *,
    identity: InterviewerIdentity,
    language: str | None,
) -> str | None:
    """One short line spoken as the voice channel opens, or ``None``.

    Fixes a real gap: ``get_opening_text`` returns ``None`` once onboarding is
    complete, and a realtime token is only minted once it IS complete, so the
    first thing a voice candidate ever heard was a raw question from the bank
    with no one introducing it.

    Returns ``None`` for an unnamed identity, which keeps every existing voice
    session byte-identical. This is deliberately NOT a second introduction —
    onboarding already greeted the candidate over REST narration. It is the
    "I'm here, let's start" beat a person gives when a call connects.

    Deliberately not a persisted ceremony message: it carries no state, and
    hearing it again after a reconnect is what a person would do anyway. That
    also keeps ``ensure_ceremony_message``'s contract (always returns a message)
    intact rather than bending it for an optional line.
    """
    if not identity.is_named():
        return None
    lang = normalize_language(language)
    if lang == "vi":
        return f"{identity.name} đây. Chúng ta bắt đầu nhé."
    return f"{identity.name} here. Let's get started."


def audio_check_text(*, language: str | None) -> str:
    """Ask the audio check as its own short turn."""
    return (
        "Cảm ơn bạn. Bạn có nghe rõ tôi không?"
        if normalize_language(language) == "vi"
        else "Thank you. Can you hear me clearly?"
    )


def language_check_text(*, language: str | None) -> str:
    """Ask the candidate to choose the interview language."""
    return (
        "Tốt rồi. Bạn muốn sử dụng ngôn ngữ nào cho buổi phỏng vấn: tiếng Anh hay tiếng Việt?"
        if normalize_language(language) == "vi"
        else (
            "Great. Which language would you like to use for this interview: English or Vietnamese?"
        )
    )


def preparation_text(*, language: str | None) -> str:
    """Offer one purposeful preparation pause before the formal briefing."""
    return (
        "Chúng ta sẽ tiếp tục bằng tiếng Việt. Trước khi tôi giải thích cấu trúc "
        "buổi phỏng vấn, bạn có cần một chút thời gian để chuẩn bị hoặc điều chỉnh "
        "thiết bị không?"
        if normalize_language(language) == "vi"
        else (
            "We’ll continue in English. Before I explain the interview structure, "
            "do you need a moment to prepare or adjust your setup?"
        )
    )


def briefing_text(
    *,
    title: str,
    time_limit_minutes: int | None,
    input_mode: str,
    language: str | None,
) -> str:
    """Explain only capabilities and rules the course interview supports."""
    lang = normalize_language(language)
    safe_title = _clean(title, limit=180) or ("buổi phỏng vấn" if lang == "vi" else "interview")
    if lang == "vi":
        duration = (
            f"Buổi phỏng vấn kéo dài tối đa {time_limit_minutes} phút."
            if time_limit_minutes
            else "Buổi phỏng vấn này không có giới hạn tổng thời gian cố định."
        )
        answer_mode = {
            "voice": "Bạn sẽ trả lời bằng giọng nói.",
            "hybrid": "Bạn có thể nhập hoặc nói câu trả lời.",
        }.get(input_mode, "Bạn sẽ nhập câu trả lời.")
        return (
            f"Cảm ơn bạn. {duration} Buổi “{safe_title}” sẽ đánh giá câu trả lời theo "
            "các tiêu chí của mô-đun hiện tại và tôi sẽ hỏi từng câu kỹ thuật một. "
            f"{answer_mode} Bạn có thể yêu cầu tôi nhắc lại hoặc làm rõ câu hỏi bất cứ "
            "lúc nào. Không có giới hạn riêng cho từng câu; đồng hồ tổng chỉ bắt đầu khi "
            "bạn xác nhận sẵn sàng. Bạn đã sẵn sàng bắt đầu chưa?"
        )

    duration = (
        f"This interview will take up to {time_limit_minutes} minutes."
        if time_limit_minutes
        else "This interview has no fixed overall time limit."
    )
    answer_mode = {
        "voice": "You’ll answer by speaking.",
        "hybrid": "You may type or speak your answers.",
    }.get(input_mode, "You’ll type your answers.")
    return (
        f"Thank you. {duration} The “{safe_title}” interview will assess your answers "
        "against the current module criteria, and I’ll ask one technical question at "
        f"a time. {answer_mode} You may ask me to repeat or clarify a question at any "
        "time. There is no separate per-question limit; the overall timer begins only "
        "when you confirm. Are you ready to begin?"
    )


def ready_transition_text(*, language: str | None) -> str:
    return (
        "Tuyệt vời—phần chào hỏi đã hoàn tất. Chúng ta bắt đầu nhé. Đây là câu hỏi đầu tiên."
        if normalize_language(language) == "vi"
        else "Great—the introduction is complete. Let’s begin. Here is your first question."
    )


def onboarding_retry_text(*, stage: str, language: str | None) -> str:
    lang = normalize_language(language)
    if stage == "identity_check":
        return (
            "Tôi chưa xác nhận được. Vui lòng cho tôi biết tôi có đang trao đổi đúng người không."
            if lang == "vi"
            else (
                "I couldn’t confirm that. Please let me know whether I’m speaking "
                "with the right candidate."
            )
        )
    if stage == "audio_check":
        return (
            "Không sao. Hãy điều chỉnh âm thanh nếu cần, sau đó cho tôi biết khi bạn nghe rõ."
            if lang == "vi"
            else (
                "No problem. Adjust your audio if needed, then let me know when you "
                "can hear me clearly."
            )
        )
    if stage == "language_check":
        return (
            "Vui lòng chọn tiếng Anh hoặc tiếng Việt cho buổi phỏng vấn."
            if lang == "vi"
            else "Please choose English or Vietnamese for the interview."
        )
    if stage == "preparation":
        return (
            "Không sao. Hãy dành thêm một chút thời gian và cho tôi biết khi bạn muốn tiếp tục."
            if lang == "vi"
            else "No problem. Take another moment and let me know when you’d like to continue."
        )
    return (
        "Không sao. Hãy dành thêm một chút thời gian. Hãy cho tôi biết khi bạn sẵn sàng bắt đầu."
        if lang == "vi"
        else "No problem. Take another moment, and let me know when you’re ready to begin."
    )


def onboarding_ceremony_kind(stage: str) -> CeremonyKind:
    """Return the single AI ceremony message associated with a setup stage."""
    return cast(
        CeremonyKind,
        {
            "identity_check": "candidate_confirmation",
            "audio_check": "audio_check",
            "language_check": "language_check",
            "preparation": "preparation",
            "readiness": "briefing",
            "completed": "ready_transition",
        }.get(stage, "candidate_confirmation"),
    )


def closing_text(
    *,
    title: str,
    name: str | None,
    persona: str | None,
    language: str | None,
    reason: FinishReason,
) -> str:
    """Build a final goodbye.  It deliberately never asks for another reply."""
    lang = normalize_language(language)
    tone = persona_from(persona)
    safe_title = _clean(title, limit=180) or ("buổi phỏng vấn" if lang == "vi" else "interview")
    address = _address(_clean(name, limit=60) or None, lang)

    if lang == "vi":
        if reason == "timed_out":
            middle = f"Thời gian cho buổi phỏng vấn “{safe_title}” đã kết thúc."
        elif reason == "ended_early":
            middle = f"Chúng ta sẽ kết thúc sớm buổi phỏng vấn “{safe_title}” tại đây."
        else:
            middle = f"Buổi phỏng vấn “{safe_title}” đến đây là kết thúc."
        thanks = f"Cảm ơn bạn{address}"
        if tone is Persona.SUPPORTIVE:
            thanks += " đã dành thời gian và nỗ lực"
        return (
            f"{thanks}. {middle} Các câu trả lời đã gửi của bạn "
            "đã được ghi nhận để đánh giá. Tạm biệt."
        )

    if reason == "timed_out":
        middle = f"The time for your “{safe_title}” interview has ended."
    elif reason == "ended_early":
        middle = f"We’ll end your “{safe_title}” interview here."
    else:
        middle = f"That concludes your “{safe_title}” interview."
    thanks = "Thank you for your time and effort" if tone is Persona.SUPPORTIVE else "Thank you"
    return (
        f"{thanks}{address}. {middle} Your submitted responses have been recorded "
        "for evaluation. Goodbye."
    )


async def _existing_message(
    db: object, session_id: object, ceremony_key: CeremonyKind
) -> InterviewSessionMessage | None:
    # Filter metadata in Python for compatibility with both PostgreSQL and the
    # lightweight DB doubles used by unit tests. The migration supplies the
    # race-proof database uniqueness guarantee in production.
    result = await db.execute(  # type: ignore[attr-defined]
        select(InterviewSessionMessage)
        .where(
            InterviewSessionMessage.session_id == session_id,
            InterviewSessionMessage.role == "ai",
        )
        .order_by(InterviewSessionMessage.created_at)
    )
    for message in result.scalars().all():
        metadata = message.metadata_json
        if isinstance(metadata, dict) and metadata.get("ceremony_key") == ceremony_key:
            return cast(InterviewSessionMessage, message)
    return None


async def ensure_ceremony_message(
    db: object,
    *,
    session: InterviewSession,
    kind: CeremonyKind,
    language: str | None,
    reason: FinishReason = "natural",
) -> InterviewSessionMessage:
    """Return the one persisted opening/closing message for a session."""
    existing = await _existing_message(db, session.id, kind)
    if existing is not None:
        return existing

    config = await db.get(InterviewConfig, session.interview_config_id)  # type: ignore[attr-defined]
    profile = await db.get(UserProfile, session.student_id)  # type: ignore[attr-defined]
    name = session_address_name(session, profile)
    title = config.title if config is not None else "Interview"
    persona = config.persona if config is not None else None
    persona_profile_json = getattr(config, "persona_profile_json", None) if config else None
    if kind in {"opening", "candidate_confirmation"}:
        text = opening_text(
            title=title,
            name=name,
            persona=persona,
            language=language,
            identity=interviewer_identity.identity_from_config(persona_profile_json),
            # opening_style rides on the persona profile; resolving it here keeps
            # the preset/override merge in one place (persona.profile_from_config).
            opening_style=profile_from_config(persona, persona_profile_json).opening_style,
        )
    elif kind == "audio_check":
        text = audio_check_text(language=language)
    elif kind == "language_check":
        text = language_check_text(language=language)
    elif kind == "preparation":
        text = preparation_text(language=language)
    elif kind == "briefing":
        text = briefing_text(
            title=title,
            time_limit_minutes=config.time_limit_minutes if config is not None else None,
            input_mode=session.input_mode,
            language=language,
        )
    elif kind == "ready_transition":
        text = ready_transition_text(language=language)
    else:
        text = closing_text(
            title=title,
            name=name,
            persona=persona,
            language=language,
            reason=reason,
        )
    message = InterviewSessionMessage(
        session_id=session.id,
        session_question_id=None,
        role="ai",
        content_text=text,
        metadata_json={
            "kind": kind,
            "ceremony_key": kind,
            "language": normalize_language(language),
            **({"finish_reason": reason} if kind == "closing" else {}),
        },
    )
    db.add(message)  # type: ignore[attr-defined]
    await flush_or_conflict(db)  # type: ignore[arg-type]
    return message


__all__ = [
    "FinishReason",
    "ask_preferred_name_text",
    "audio_check_text",
    "briefing_text",
    "candidate_first_name",
    "closing_text",
    "preferred_name_ack_text",
    "session_address_name",
    "ensure_ceremony_message",
    "language_check_text",
    "normalize_language",
    "onboarding_ceremony_kind",
    "onboarding_retry_text",
    "opening_text",
    "preparation_text",
    "ready_transition_text",
]
