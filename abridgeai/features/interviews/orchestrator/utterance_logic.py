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
        system_prompt = render_prompt("prompts/utterance_system.j2", language=(language or "en"))
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
        prompt_payload: dict[str, object] = {
            "persona": persona.value,
            "persona_traits": profile.as_prompt_traits(),
            "language": language or "en",
            "action": decision.action.value,
            "approved_parts": {
                "acknowledgement": fallback.acknowledgement,
                "transition": fallback.transition,
                "question_or_probe": fallback.question_or_probe,
            },
        }
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
        rewritten = _validated_rewrite(payload, fallback)
        if rewritten is not None:
            return rewritten, "llm"
    except Exception:  # noqa: BLE001 — phrasing is best-effort; never bubble
        logger.exception("interview utterance phrasing failed; using deterministic fallback")

    return fallback, "fallback"


def _validated_rewrite(payload: object, fallback: Utterance) -> Utterance | None:
    """Accept an LLM rewrite only if it preserves the question and adds no bulk.

    Returns None to signal "reject → use fallback". The question/probe text must
    survive verbatim (the decision is authoritative); acknowledgement/transition
    may be rephrased. An answer-leak guard rejects a rewrite that balloons in
    length (a proxy for the model injecting expected-answer content).
    """
    if not isinstance(payload, dict):
        return None
    ack = _clean(payload.get("acknowledgement"), limit=200)
    transition = _clean(payload.get("transition"), limit=200)
    combined = _clean(payload.get("ai_turn_text"), limit=1200)
    if not combined:
        return None

    # The authoritative question/probe MUST appear unchanged in the rewrite.
    qp = fallback.question_or_probe.strip()
    if qp and qp not in combined:
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


__all__ = ["UTTERANCE_STAGE_NAME", "generate_utterance"]
