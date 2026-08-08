"""LLM phrasing layer for interviewer utterances (Phase 9).

Wraps the deterministic :func:`build_fallback_utterance`. The LLM may rewrite
the *phrasing* for naturalness, but it can NEVER change the decision (action,
reason code, selected question, advance/end/record flags) — those are fixed by
the caller before this runs. On any failure (LLM down, malformed output,
answer-leak guard trips) we return the deterministic fallback verbatim.

Guardrails enforced here (requirement #8):
* The LLM only reshapes acknowledgement + transition + the provided question/
  probe text. It is told the question text verbatim and instructed to include
  it unchanged; if it drops or mutates it we discard the rewrite.
* Answer-leak heuristic: the rewrite must not add content beyond what we sent.
  We keep it conservative — if the rewrite is suspiciously longer, fall back.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from abridgeai.ai.llm import LLMGateway, LLMRole
from abridgeai.ai.prompts import render_prompt
from abridgeai.features.interviews.orchestrator.interviewer_identity import (
    InterviewerIdentity,
    as_prompt_identity,
)
from abridgeai.features.interviews.orchestrator.persona import PersonaProfile, profile_from
from abridgeai.features.interviews.orchestrator.utterance import (
    Persona,
    Utterance,
    build_fallback_utterance,
)

if TYPE_CHECKING:
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession

    from abridgeai.features.interviews.orchestrator.decision import InterviewerDecision

logger = logging.getLogger(__name__)

UTTERANCE_STAGE_NAME = "interview_utterance"

# Reuse the small, latency-sensitive follow-up tier — phrasing is cheap and the
# student is waiting. (No new role needed; INTERVIEW_FOLLOWUP is small-tier.)
_UTTERANCE_ROLE = LLMRole.INTERVIEW_FOLLOWUP


async def generate_utterance(
    db: AsyncSession,
    decision: InterviewerDecision,
    *,
    persona: Persona,
    language: str | None,
    question_text: str | None = None,
    grounding_question: str | None = None,
    persona_profile: object | None = None,
    identity: object | None = None,
    use_llm: bool = True,
    affect: object | None = None,
    hint_level: int = 0,
    reframe_count: int = 0,
    time_pressure: bool = False,
    recovery: bool = False,
    pipeline_run_id: UUID | None = None,
    gateway: LLMGateway | None = None,
) -> tuple[Utterance, str]:
    """Return ``(utterance, status)`` for a decision.

    ``status`` is one of ``"llm"`` (LLM phrasing used), ``"fallback"``
    (deterministic template — LLM disabled/failed/rejected). Never raises: the
    deterministic fallback always yields a valid utterance so the interview is
    never blocked (requirement #9).
    """
    fallback = build_fallback_utterance(
        decision,
        persona=persona,
        language=language,
        question_text=question_text,
        affect=affect,
        hint_level=hint_level,
        reframe_count=reframe_count,
        time_pressure=time_pressure,
        recovery=recovery,
    )
    if not use_llm:
        return fallback, "fallback"

    try:
        # An assistance turn carries a placeholder scaffold, not exam content, so
        # the prompt lets it be reworded for the current question. Keyed off the
        # SAME condition as `require_verbatim` below (inverted) so the prompt and
        # the validator can never disagree about which mode this turn is in.
        assistance_turn = not _requires_verbatim(fallback, question_text)
        system_prompt = render_prompt(
            "prompts/utterance_system.j2",
            language=(language or "en"),
            assistance=assistance_turn,
        )
        # Resolve the persona label to its trait profile and hand the phrasing
        # model explicit tone numbers (warmth / directness / verbosity /
        # formality / ack_frequency + opening_style) instead of a bare word it
        # has to interpret. TONE ONLY — as_prompt_traits() carries no
        # decision-bearing data (see persona.py), and the system prompt still
        # forbids the model from changing difficulty, fairness, or the verbatim
        # question. A fourth/custom persona flows through here with no code
        # change: it is just a different set of trait numbers.
        # A resolved override profile (teacher per-trait tuning, Phase 3) takes
        # precedence; otherwise resolve the preset from the persona label. Both
        # are clamped so an override can never push a trait out of range.
        if isinstance(persona_profile, PersonaProfile):
            profile = persona_profile.clamped()
        else:
            profile = profile_from(persona.value).clamped()
        # Interviewer IDENTITY (who is speaking) rides alongside the tone traits.
        # Separate axis from persona: a supportive tech lead and a strict tech
        # lead are both coherent. as_prompt_identity() carries only name / title
        # / register — presentational strings, nothing decision-bearing (guarded
        # by tests/unit/test_interviewer_identity.py). Absent identity → the key
        # is omitted entirely so an unnamed config's prompt is byte-identical to
        # what it was before identity existed.
        prompt_payload = build_prompt_payload(
            decision,
            fallback=fallback,
            persona_value=persona.value,
            persona_traits=profile.as_prompt_traits(),
            language=language,
            question_text=question_text,
            grounding_question=grounding_question,
            hint_level=hint_level,
        )
        if isinstance(identity, InterviewerIdentity) and identity.is_named():
            prompt_payload["interviewer"] = as_prompt_identity(identity, language)
        user_prompt = json.dumps(prompt_payload, ensure_ascii=False)
        gateway = gateway or LLMGateway()
        async with db.begin_nested():
            llm_result = await gateway.generate_json(
                role=_UTTERANCE_ROLE,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                db=db,
                stage_name=UTTERANCE_STAGE_NAME,
                pipeline_run_id=pipeline_run_id,
            )
        payload = llm_result.content_json if isinstance(llm_result.content_json, dict) else None
        # Preserve the text verbatim ONLY when it is the authoritative bank
        # question. On an assistance turn `question_or_probe` is a canned template
        # and rewriting it is the whole point of the call.
        rewritten = validated_rewrite(payload, fallback, require_verbatim=not assistance_turn)
        if rewritten is not None:
            return rewritten, "llm"
    except Exception:  # noqa: BLE001 — phrasing is best-effort; never bubble
        logger.exception("interview utterance phrasing failed; using deterministic fallback")

    return fallback, "fallback"


def _requires_verbatim(fallback: Utterance, question_text: str | None) -> bool:
    """Whether ``question_or_probe`` is authoritative exam content.

    True only when it IS the text the caller declared authoritative — the selected
    bank question, or a re-spoken question on REPEAT. Anything else is a generated
    scaffold (hint / clarify / reframe), which the model may reword.

    Single source of truth for BOTH the prompt mode and the validator: if they
    disagreed, one of the two failure modes below would ship silently.
      * prompt says "reword", validator demands verbatim → every rewrite rejected
      * prompt says "keep", validator allows changes → generic boilerplate stays
    """
    if not question_text:
        return False
    return fallback.question_or_probe.strip() == question_text.strip()


def build_prompt_payload(
    decision: InterviewerDecision,
    *,
    fallback: Utterance,
    persona_value: str,
    persona_traits: object,
    language: str | None,
    question_text: str | None,
    grounding_question: str | None = None,
    hint_level: int,
) -> dict[str, object]:
    """Assemble the phrasing prompt body.

    Two different questions are in play and conflating them is what made hints
    generic:

    * ``question_text`` — the AUTHORITATIVE text that must survive verbatim (the
      selected bank question, or a re-spoken question on REPEAT). It is None on a
      hint / clarify / reframe turn, because ``probe_seed_text`` only yields a
      value for REPEAT_QUESTION.
    * ``grounding_question`` — the question the candidate is currently working on,
      supplied so the model can write a hint ABOUT something. Always pass it when
      a current question exists.

    ``current_question`` in the payload takes the grounding question, falling back
    to the authoritative text (identical on an advance). It is the question PROMPT
    only — never expected answers or rubric content — so it cannot leak an answer.
    """
    payload: dict[str, object] = {
        "persona": persona_value,
        "persona_traits": persona_traits,
        "language": language or "en",
        "action": decision.action.value,
        "approved_parts": {
            "acknowledgement": fallback.acknowledgement,
            "transition": fallback.transition,
            "question_or_probe": fallback.question_or_probe,
        },
    }
    grounding = grounding_question or question_text
    if grounding:
        payload["current_question"] = grounding
    if hint_level > 0:
        payload["hint_level"] = hint_level
    return payload


def validated_rewrite(
    payload: object,
    fallback: Utterance,
    *,
    require_verbatim: bool,
) -> Utterance | None:
    """Accept an LLM rewrite only if it is answer-safe.

    Returns None to signal "reject → use fallback".

    ``require_verbatim`` must be True when ``fallback.question_or_probe`` is an
    authoritative question from the bank: the selector chose it, and a model that
    rewords it changes difficulty and fairness. It must be False when that field
    holds a generated template (hint / clarify / reframe), because replacing that
    wording with something grounded in the question is the purpose of the call.

    The length-based answer-leak guard applies in BOTH modes — it is the defence
    against the model volunteering the expected answer, and exempting the
    verbatim check must never exempt that.
    """
    if not isinstance(payload, dict):
        return None
    ack = _clean(payload.get("acknowledgement"), limit=200)
    transition = _clean(payload.get("transition"), limit=200)
    combined = _clean(payload.get("ai_turn_text"), limit=1200)
    if not combined:
        return None

    qp = fallback.question_or_probe.strip()
    if require_verbatim and qp and qp not in combined:
        return None

    # Answer-leak guard: reject a rewrite materially longer than the source
    # parts (model likely added expected-answer content).
    source_len = len(fallback.ai_turn_text)
    if len(combined) > max(source_len * 2, source_len + 120):
        return None

    return Utterance(
        acknowledgement=ack,
        transition=transition,
        question_or_probe=qp,
        ai_turn_text=combined,
    )


def _clean(value: object, *, limit: int) -> str:
    if isinstance(value, str):
        return value.strip()[:limit]
    return ""


__all__ = [
    "UTTERANCE_STAGE_NAME",
    "build_prompt_payload",
    "generate_utterance",
    "validated_rewrite",
]
