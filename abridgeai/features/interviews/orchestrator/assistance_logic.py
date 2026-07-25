"""Safe, progressive assistance for the current interview question.

This stage deliberately receives only the learner-visible question and an
optional term from that question. It never receives model answers, outcomes,
rubrics, future questions, or author-only instructions.
"""

from __future__ import annotations

import json
import logging
import re

from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.ai.llm import LLMGateway, LLMRole
from abridgeai.ai.prompts import render_prompt
from abridgeai.features.interviews.orchestrator.security import SecurityAction

logger = logging.getLogger(__name__)

ASSISTANCE_STAGE_NAME = "interview_question_assistance"
_ASSISTANCE_ROLE = LLMRole.INTERVIEW_FOLLOWUP


def _is_vi(language: str | None) -> bool:
    return bool(language and language.lower().startswith("vi"))


def _fallback_clarification(question_text: str, language: str | None) -> str:
    """Return a useful answer-safe rephrasing without depending on the LLM."""
    question = " ".join((question_text or "").strip().split()).rstrip(" ?.!")
    vi = _is_vi(language)
    if vi:
        return (
            "Nói cách khác, hãy xác định các khái niệm chính trong câu hỏi, giải thích "
            "mối quan hệ giữa chúng và trả lời rõ từng phần được yêu cầu."
        )

    comparison = re.fullmatch(
        r"compare(?:\s+and\s+contrast)?\s+(.+?)\s+"
        r"(?:and|with|versus|vs\.?)\s+(.+?)(?:\s+in\s+(.+))?",
        question,
        flags=re.IGNORECASE,
    )
    if comparison:
        first, second, context = comparison.groups()
        context_suffix = f" within {context}" if context else ""
        return (
            f"Put another way, explain what {first} and {second} have in common, "
            f"how they differ, and how they relate{context_suffix}."
        )
    return (
        "Put another way, identify the main concepts in the question and address "
        "each part of what it is asking in a clear, structured response."
    )


def _fallback(
    action: SecurityAction,
    language: str | None,
    *,
    question_text: str = "",
    valid_term: bool = True,
    hint_level: int | None = None,
) -> str:
    vi = _is_vi(language)
    if action is SecurityAction.HINT_CURRENT_QUESTION:
        if hint_level is not None:
            # Slice 11 upgrade: escalation-aware fallback. Reuse the SHARED
            # deterministic laddered hint so a typed hint still escalates and is
            # byte-identical to the adaptive decision path's fallback when the
            # LLM is unavailable. Imported locally to avoid an import cycle.
            from abridgeai.features.interviews.orchestrator.utterance import (  # noqa: PLC0415
                laddered_hint,
            )

            return laddered_hint(hint_level, language)
        return (
            "Gợi ý nhỏ: hãy sắp xếp câu trả lời theo các khái niệm chính trong câu hỏi, "
            "mối quan hệ giữa chúng và một ví dụ thực tế nếu phù hợp."
            if vi
            else "A small hint: organize your answer around the main concepts in the "
            "question, how they relate, and a practical example if relevant."
        )
    if action is SecurityAction.EXPLAIN_CURRENT_TERM:
        if not valid_term:
            return (
                "Hãy chọn một từ hoặc cụm từ xuất hiện trong câu hỏi hiện tại."
                if vi
                else "Please choose a word or phrase that appears in the current question."
            )
        return (
            "Tôi chưa thể giải thích thuật ngữ đó ngay lúc này. Bạn có thể thử lại hoặc "
            "yêu cầu tôi diễn đạt lại toàn bộ câu hỏi."
            if vi
            else "I couldn't explain that term just now. You can try again or ask me to "
            "rephrase the whole question."
        )
    return _fallback_clarification(question_text, language)


# Verbs that make a question ASK FOR a definition/comparison of its subject.
# When the requested term IS that subject, explaining it hands over the answer
# (the "define X" → "explain X" leak). EN + VI.
_DEFINITION_LEAD = (
    r"compare(?:\s+and\s+contrast)?|contrast|define|describe|explain|discuss|"
    r"what\s+(?:is|are)|state|list|outline|"
    r"so\s+sánh|định\s+nghĩa|giải\s+thích|mô\s+tả|trình\s+bày|liệt\s+kê|nêu"
)
# Splits the subject blob off the trailing context ("... in a dimensional model",
# "... trong mô hình ..."). Only the part BEFORE this is the graded subject.
_SUBJECT_CONTEXT_SPLIT = re.compile(
    r"\s+(?:in|within|for|under|trong|theo|trong\s+ngữ\s+cảnh)\s+",
    flags=re.IGNORECASE,
)
# Splits a multi-part subject ("fact tables and factless fact tables").
_SUBJECT_CONJ_SPLIT = re.compile(
    r"\s+(?:and|versus|vs\.?|with|or|và|hoặc|với)\s+|\s*,\s*",
    flags=re.IGNORECASE,
)


def _normalize(text: str) -> str:
    return " ".join((text or "").casefold().split()).strip(" \t\r\n.,?!:;\"'“”‘’")


def question_subjects(question_text: str) -> list[str]:
    """The concept(s) a definition/comparison question asks the candidate to produce.

    For "Compare and contrast fact tables and factless fact tables in a
    dimensional model." this is ``["fact tables", "factless fact tables"]`` —
    NOT "dimensional model", which is trailing context. Returns [] when the
    question is not a define/compare/explain-style prompt, so a plain question
    never triggers the guard.
    """
    normalized = _normalize(question_text)
    if not normalized:
        return []
    match = re.match(rf"(?:{_DEFINITION_LEAD})\s+(.+)", normalized, flags=re.IGNORECASE)
    if not match:
        return []
    subject_blob = _SUBJECT_CONTEXT_SPLIT.split(match.group(1), maxsplit=1)[0]
    subjects: list[str] = []
    for part in _SUBJECT_CONJ_SPLIT.split(subject_blob):
        # Drop leading articles so "the fact table" ≈ "fact table".
        cleaned = re.sub(r"^(?:the|a|an)\s+", "", _normalize(part))
        if len(cleaned) > 1:
            subjects.append(cleaned)
    return subjects


def term_is_question_subject(term: str, question_text: str) -> bool:
    """True when explaining ``term`` would define the question's own subject.

    Conservative by design (the leak it prevents — handing the candidate the
    answer — is worse than occasionally redirecting a genuine vocabulary
    question): a term that equals, contains, or is contained by any subject
    phrase counts as the subject. Context terms that are not the subject
    (e.g. "dimensional model" above) do not match and are still explained.
    """
    normalized_term = re.sub(r"^(?:the|a|an)\s+", "", _normalize(term))
    if len(normalized_term) <= 1:
        return False
    for subject in question_subjects(question_text):
        if normalized_term == subject or normalized_term in subject or subject in normalized_term:
            return True
    return False


def _subject_redirect(language: str | None) -> str:
    """Answer-safe reply when the requested term IS the question's subject.

    Refuses to define it (that would be the answer) and points at structure
    instead — the same scaffolding a hint gives, so the candidate is still
    helped without being handed the concept.
    """
    if _is_vi(language):
        return (
            "Thuật ngữ đó chính là điều câu hỏi đang yêu cầu bạn định nghĩa, nên mình "
            "không giải thích thay được. Bạn thử sắp xếp câu trả lời quanh: nó là gì, "
            "dùng để làm gì, và nó liên hệ thế nào với khái niệm còn lại trong câu hỏi."
        )
    return (
        "That term is exactly what the question asks you to define, so I can't "
        "explain it for you — but you could organize your answer around what it "
        "is, what it's used for, and how it relates to the other concept in the "
        "question."
    )


def extract_question_term(request_text: str, question_text: str) -> str | None:
    """Return a bounded requested phrase only when it occurs in the question."""
    request = (request_text or "").strip()
    question = (question_text or "").strip()
    if not request or not question:
        return None

    candidates: list[str] = []
    # A candidate may naturally respond to "which phrase is unclear?" with
    # only that phrase. It remains bounded and is accepted only when the whole
    # phrase occurs in the learner-visible current question.
    if len(request) <= 100 and len(request.split()) <= 12:
        candidates.append(request)
    candidates.extend(
        value.strip() for value in re.findall(r"[\"'“”‘’]([^\"'“”‘’]{1,100})[\"'“”‘’]", request)
    )
    for pattern in (
        r"what\s+does\s+(.{1,100}?)\s+mean(?:\s+in\s+(?:this|the)\s+question)?[?.!]*$",
        r"(?:explain|define)\s+(?:the\s+)?(?:term\s+)?(.{1,100}?)[?.!]*$",
        r"(?:giải\s+thích|định\s+nghĩa)\s+(?:thuật\s+ngữ\s+)?(.{1,100}?)[?.!]*$",
    ):
        match = re.search(pattern, request, flags=re.IGNORECASE)
        if match:
            candidates.append(match.group(1).strip())

    normalized_question = " ".join(question.casefold().split())
    for candidate in candidates:
        cleaned = candidate.strip(" \t\r\n.,?!:;\"'“”‘’")
        normalized = " ".join(cleaned.casefold().split())
        if 1 < len(normalized) <= 100 and normalized in normalized_question:
            return cleaned
    return None


def is_term_selection_reply(
    request_text: str,
    question_text: str,
    *,
    prior_assistance_text: str,
    prior_assistance_kind: str | None,
) -> bool:
    """Recognize a bare term only after the interviewer explicitly requested one.

    This narrow contextual rule repairs in-progress sessions that contain the
    former clarification fallback. A short phrase is never converted into a
    control turn merely because it happens to occur in the question.
    """
    if prior_assistance_kind != "clarification":
        return False
    if extract_question_term(request_text, question_text) is None:
        return False
    prompt = " ".join((prior_assistance_text or "").casefold().split())
    return any(
        marker in prompt
        for marker in (
            "which word or phrase",
            "what word or phrase",
            "tell me which word",
            "tell me which phrase",
            "which term",
            "từ hoặc cụm từ nào",
            "thuật ngữ nào",
        )
    )


def _validated_text(payload: object, action: SecurityAction, question_text: str) -> str | None:
    if not isinstance(payload, dict):
        return None
    value = payload.get("response_text")
    if not isinstance(value, str):
        return None
    text = " ".join(value.strip().split())
    if not text:
        return None
    limit = 320 if action is SecurityAction.HINT_CURRENT_QUESTION else 600
    if len(text) > limit:
        return None
    if action is SecurityAction.CLARIFY_CURRENT_QUESTION:
        original = " ".join(question_text.strip().split())
        if text.casefold() == original.casefold() or len(text) > max(len(original) * 2, 500):
            return None
        if re.search(
            r"\b(?:which|what)\s+(?:word|phrase|term|part)\b|"
            r"\btell\s+me\s+(?:which|what)\b",
            text,
            flags=re.IGNORECASE,
        ):
            # Clarification must rephrase immediately. A generated question
            # here creates hidden state and lets the next bare term enter the
            # assessed-answer path accidentally.
            return None
    return text


async def generate_question_assistance(
    db: AsyncSession,
    *,
    action: SecurityAction,
    question_text: str,
    request_text: str,
    language: str,
    persona: str | None,
    gateway: LLMGateway | None = None,
    hint_level: int | None = None,
) -> str:
    """Generate one answer-safe clarification, term explanation, or scaffold.

    ``hint_level`` (Slice 11 upgrade) escalates a hint: when provided, it is fed
    to the model so it produces a question-SPECIFIC hint at that escalation rung,
    and the deterministic shared laddered hint at the same level is the answer-
    safe fallback if the model is unavailable or returns unusable text. So a
    typed hint gets question-specific phrasing when the LLM is up and still
    escalates identically to the adaptive decision path either way.
    """
    term = (
        extract_question_term(request_text, question_text)
        if action is SecurityAction.EXPLAIN_CURRENT_TERM
        else None
    )
    if action is SecurityAction.EXPLAIN_CURRENT_TERM and term is None:
        return _fallback(
            action,
            language,
            question_text=question_text,
            valid_term=False,
        )
    # Semantic-leak guard: when the requested term IS the concept the question
    # asks the candidate to define/compare, explaining it hands over the answer.
    # Refuse-and-redirect BEFORE the LLM call (the model prompt can't see the
    # whole question's intent, only the isolated term, so it can't catch this).
    if action is SecurityAction.EXPLAIN_CURRENT_TERM and term is not None:
        if term_is_question_subject(term, question_text):
            return _subject_redirect(language)

    try:
        system_prompt = render_prompt(
            "prompts/assistance_system.j2",
            language=language or "en",
        )
        payload: dict[str, object] = {
            "action": action.value,
            "language": language or "en",
            "persona": persona or "neutral",
            "current_question": question_text,
            "requested_term": term,
        }
        if hint_level is not None and action is SecurityAction.HINT_CURRENT_QUESTION:
            # Escalation rung for the model: 0 = gentle structural nudge,
            # 1 = stronger scaffold, 2+ = how-to-approach (never the answer).
            payload["hint_level"] = hint_level
        user_prompt = json.dumps(payload, ensure_ascii=False)
        llm = gateway or LLMGateway()
        async with db.begin_nested():
            result = await llm.generate_json(
                role=_ASSISTANCE_ROLE,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                db=db,
                stage_name=ASSISTANCE_STAGE_NAME,
            )
        text = _validated_text(result.content_json, action, question_text)
        if text is not None:
            return text
    except Exception:  # noqa: BLE001 -- assistance must never block the interview
        logger.exception("interview assistance generation failed; using safe fallback")
    return _fallback(action, language, question_text=question_text, hint_level=hint_level)


__all__ = [
    "ASSISTANCE_STAGE_NAME",
    "extract_question_term",
    "generate_question_assistance",
    "is_term_selection_reply",
    "question_subjects",
    "term_is_question_subject",
]
