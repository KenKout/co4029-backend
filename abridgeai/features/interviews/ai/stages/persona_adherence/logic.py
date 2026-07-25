"""Persona-adherence stage: audit whether the AI interviewer held its persona.

This stage runs OFFLINE, after a session completes, over the STORED transcript
(``interview_session_messages``) — it never re-runs a session and no student is
waiting. It exists because the product promises three interviewer personas but,
before this, nothing measured whether the live interviewer actually held the
declared tone. The judge here is a tone-only auditor (see ``prompts/system.j2``).

Boundary
--------
This is a DIAGNOSTIC. It must never influence ``pass_verdict``, rubric scores,
or any control flow. It reads a persona (as a :class:`PersonaProfile`) and the
transcript, and returns a :class:`PersonaAdherence`. The caller decides whether
to persist it; the stage is side-effect-free apart from the one LLM call it
books against ``ai_model_calls`` for cost attribution.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from abridgeai.ai.llm import LLMGateway, LLMRole
from abridgeai.ai.prompts import render_prompt
from abridgeai.features.interviews.ai.stages.persona_adherence.parsers import (
    PersonaAdherence,
    parse_persona_adherence,
    unavailable,
)

if TYPE_CHECKING:
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession

    from abridgeai.features.interviews.models import InterviewSessionMessage
    from abridgeai.features.interviews.orchestrator.persona import PersonaProfile

PERSONA_ADHERENCE_STAGE_NAME = "persona_adherence"

# Cap how many interviewer turns we send. A tone audit does not need every turn
# of a very long session, and an unbounded transcript could blow the context /
# cost budget of what is only a diagnostic.
_MAX_TURNS = 60


async def audit_persona_adherence(
    db: AsyncSession,
    *,
    persona: PersonaProfile,
    messages: Sequence[InterviewSessionMessage],
    pipeline_run_id: UUID | None = None,
    gateway: LLMGateway | None = None,
) -> PersonaAdherence:
    """Judge how well the interviewer's turns matched the declared persona.

    Returns ``unavailable()`` (never raises) when there are no interviewer turns
    to judge or the LLM produces nothing usable — the caller can then tell
    "audited, clean" apart from "no audit". Tone-only: the persona traits are the
    sole yardstick; correctness, difficulty, and question quality are ignored.
    """
    turns = _interviewer_turns(messages)
    if not turns:
        return unavailable()

    gateway = gateway or LLMGateway()
    system_prompt = render_prompt("prompts/system.j2")
    user_prompt = json.dumps(
        {
            "declared_persona": persona.clamped().as_prompt_traits(),
            "interviewer_turns": turns,
        },
        ensure_ascii=False,
    )

    try:
        llm_result = await gateway.generate_json(
            role=LLMRole.INTERVIEW_PERSONA_ADHERENCE,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            db=db,
            stage_name=PERSONA_ADHERENCE_STAGE_NAME,
            pipeline_run_id=pipeline_run_id,
            parent_run_id=pipeline_run_id,
        )
    except Exception:  # noqa: BLE001 — diagnostic must never break the caller
        return unavailable()

    payload = llm_result.content_json if isinstance(llm_result.content_json, dict) else None
    return parse_persona_adherence(payload)


def _interviewer_turns(
    messages: Sequence[InterviewSessionMessage],
) -> list[dict[str, Any]]:
    """Extract the AI interviewer's spoken turns, numbered, in order.

    Only ``role='ai'`` turns with visible text are audited — the persona is a
    property of what the INTERVIEWER says, not the candidate. Turn numbers are
    1-based positional indices over the returned turns so ``drift_turns`` in the
    judge output can point back at specific lines.
    """
    turns: list[dict[str, Any]] = []
    for message in messages:
        if getattr(message, "role", None) != "ai":
            continue
        text = (getattr(message, "content_text", None) or "").strip()
        if not text:
            continue
        turns.append({"turn": len(turns) + 1, "text": text})
        if len(turns) >= _MAX_TURNS:
            break
    return turns


__all__ = [
    "PERSONA_ADHERENCE_STAGE_NAME",
    "audit_persona_adherence",
]
