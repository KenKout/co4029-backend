"""Synthetic candidate personas for the voice harness (Phase 5).

The harness historically drove the interview with a fixed ``--answers`` list:
the same words every run, regardless of what the interviewer asked. That
exercises the *plumbing* (STT → adaptive brain → TTS) but never the parts of
the system that only react to HOW a candidate answers — the hint ladder, affect
detection (nervous / rambling / terse), the rambling-redirect steer, or the
recovery lead-in.

This module adds an optional LLM answer generator behind ``--candidate-profile``.
A profile is an *ability × state* prose sketch (the Sotopia lesson: a plain
"it's natural to feel a bit nervous" prompt is what actually surfaces persona
differences, far more than a bare ability label). Given the interviewer's latest
question, the generator produces the next student answer in-character.

Hard boundaries
---------------
* This is a TEST HARNESS helper. It never runs in production and touches nothing
  in the live turn path.
* The generated answer is just spoken student audio — the adaptive brain and the
  rubric judge it exactly as they would a human's. The profile never reaches
  scoring; it only shapes the words the synthetic student says.
* The static ``--answers`` path stays the default, so existing Makefile targets
  (``voice-harness-en`` / ``voice-harness-vi``) run byte-for-byte unchanged.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

logger = logging.getLogger("voice_harness.candidate")


@dataclass(frozen=True)
class CandidateProfile:
    """One synthetic candidate: an ability level crossed with an affective state.

    ``ability`` and ``state`` are prose because that is what makes the answers
    read like a real person under that condition — an ability label alone
    ("weak") produces bland text that never trips affect detection.
    """

    key: str
    ability: str
    state: str
    # A short instruction appended to the generator prompt that nudges the
    # answer LENGTH/SHAPE toward what the state implies (terse, rambling, etc.).
    style_hint: str


# The four profiles from the plan. ability × state, expressed as prose.
PROFILES: dict[str, CandidateProfile] = {
    "confident_strong": CandidateProfile(
        key="confident_strong",
        ability="strong — knows the material well and reasons clearly",
        state="calm and structured; answers directly and concisely",
        style_hint=(
            "Answer confidently in 2-3 clear sentences. Get to the point; "
            "no hedging, no rambling."
        ),
    ),
    "nervous_mid": CandidateProfile(
        key="nervous_mid",
        ability="medium — roughly half-right, some gaps and a shaky spot",
        state=(
            "anxious and hesitant; hedges, second-guesses, and sometimes "
            "self-corrects mid-answer"
        ),
        style_hint=(
            "Answer nervously: start unsure, hedge with phrases like 'I think' "
            "or 'maybe', and correct yourself once. 2-4 sentences."
        ),
    ),
    "rambling_mid": CandidateProfile(
        key="rambling_mid",
        ability="medium — has relevant ideas but buried in noise",
        state="verbose and unfocused; drifts off-topic before landing the point",
        style_hint=(
            "Answer in a long, meandering way: wander into a tangent or two "
            "before (barely) addressing the question. 4-6 rambly sentences."
        ),
    ),
    "stuck_weak": CandidateProfile(
        key="stuck_weak",
        ability="weak — largely does not know the answer",
        state="freezes and struggles; gives very little, asks for help or a hint",
        style_hint=(
            "Answer as someone stuck: 1-2 short sentences, admit you're unsure, "
            "and often ask for a hint or clarification instead of answering."
        ),
    ),
}


def profile_names() -> list[str]:
    """The valid ``--candidate-profile`` choices, in a stable order."""
    return list(PROFILES.keys())


_SYSTEM_PROMPT = (
    "You are role-playing a STUDENT being interviewed orally in a course "
    "assessment. Stay fully in character as the candidate described. You are "
    "the one ANSWERING, never the interviewer. Reply with ONLY what the student "
    "would say out loud — no stage directions, no quotation marks, no labels. "
    "Answer in the requested language. It is completely fine to be wrong, "
    "partial, or stuck: authenticity matters more than correctness. Never break "
    "character and never mention that you are an AI."
)


def _lang_name(language: str) -> str:
    return "Vietnamese" if (language or "en").lower().startswith("vi") else "English"


def _user_prompt(profile: CandidateProfile, question: str, language: str) -> str:
    return json.dumps(
        {
            "your_ability": profile.ability,
            "your_state": profile.state,
            "how_to_answer": profile.style_hint,
            "language": _lang_name(language),
            "interviewer_just_said": question,
            "task": (
                "Give your spoken answer to the interviewer's latest turn, in "
                f"{_lang_name(language)}, staying in character."
            ),
        },
        ensure_ascii=False,
    )


# A per-profile static fallback answer, used when no LLM gateway is available
# (offline dry-run / missing creds) so --candidate-profile still produces a
# distinct, on-character turn instead of crashing.
_FALLBACK: dict[tuple[str, str], str] = {
    ("confident_strong", "en"): (
        "A fact table stores the quantitative measurements of a business "
        "process, with foreign keys to the surrounding dimension tables."
    ),
    ("confident_strong", "vi"): (
        "Bảng fact lưu các số đo định lượng của một quy trình nghiệp vụ, cùng "
        "các khóa ngoại trỏ tới các bảng dimension xung quanh."
    ),
    ("nervous_mid", "en"): (
        "Um, I think a fact table holds the numbers? Like measurements... "
        "wait, no — I mean it links to the dimension tables, I'm not totally sure."
    ),
    ("nervous_mid", "vi"): (
        "Ừm, em nghĩ bảng fact chứa các con số? Kiểu số đo... khoan, ý em là "
        "nó liên kết tới các bảng dimension, em không chắc lắm ạ."
    ),
    ("rambling_mid", "en"): (
        "So there are a lot of tables in a warehouse, right, and dimensions "
        "describe things like customers and dates, which reminds me of star "
        "schemas we saw — anyway I guess a fact table is sort of the middle one "
        "with the numbers in it."
    ),
    ("rambling_mid", "vi"): (
        "Thì trong kho dữ liệu có nhiều bảng lắm, dimension mô tả mấy thứ như "
        "khách hàng với ngày tháng, làm em nhớ tới star schema hôm trước — nói "
        "chung chắc bảng fact là cái ở giữa chứa mấy con số ạ."
    ),
    ("stuck_weak", "en"): "I'm not really sure. Could you give me a hint?",
    ("stuck_weak", "vi"): "Em không chắc lắm ạ. Thầy cho em một gợi ý được không?",
}


def _fallback_answer(profile: CandidateProfile, language: str) -> str:
    lang = "vi" if (language or "en").lower().startswith("vi") else "en"
    return _FALLBACK.get(
        (profile.key, lang),
        _FALLBACK[("confident_strong", lang)],
    )


async def generate_answer(
    profile: CandidateProfile,
    *,
    question: str,
    language: str,
    settings: object,
) -> str:
    """Produce one in-character student answer to ``question``.

    Best-effort: on any gateway problem (missing creds, network, malformed
    output) it falls back to the per-profile static answer so a harness run is
    never blocked by the answer generator.
    """
    try:
        from abridgeai.ai.llm import LLMGateway, LLMRole
        from abridgeai.core.db import get_sessionmaker

        gateway = LLMGateway()
        # generate_json books the call against ai_model_calls, so it needs a DB
        # session. The harness runs outside a request, so open a short-lived one.
        async with get_sessionmaker()() as db:
            result = await gateway.generate_json(
                role=LLMRole.INTERVIEW_FOLLOWUP,  # small/fast tier; this is a helper
                system_prompt=_SYSTEM_PROMPT
                + ' Return JSON: {"answer": "<what the student says>"}.',
                user_prompt=_user_prompt(profile, question, language),
                db=db,
                stage_name="voice_harness_candidate",
            )
            await db.commit()
        payload = result.content_json if isinstance(result.content_json, dict) else None
        if isinstance(payload, dict):
            answer = payload.get("answer")
            if isinstance(answer, str) and answer.strip():
                return answer.strip()
    except Exception:  # noqa: BLE001 — harness helper must never hard-fail
        logger.warning(
            "candidate answer generation failed for profile=%s; using fallback",
            profile.key,
            exc_info=True,
        )
    return _fallback_answer(profile, language)


__all__ = [
    "CandidateProfile",
    "PROFILES",
    "generate_answer",
    "profile_names",
]
