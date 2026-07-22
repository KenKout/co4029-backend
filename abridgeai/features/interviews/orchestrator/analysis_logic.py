"""Answer-analysis LLM call (Phase 3).

Wires :mod:`orchestrator.analysis` to the LLM gateway. Pure types + parser
live in that module; this one owns the I/O and mirrors the follow-up / intent
conventions:

* SAVEPOINT (``begin_nested``) around the gateway call so a failed
  ``ai_model_calls`` audit flush can't poison the request session.
* Malformed output is retried ONCE (per the brief's Phase 3 requirement),
  then falls back to a safe not-assessable analysis — the student's turn is
  still recorded and the session still advances.
* Never raises: analysis is guidance for the next interviewer action, never a
  hard dependency of recording the answer.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from abridgeai.ai.llm import LLMGateway, LLMRole
from abridgeai.ai.prompts import render_prompt
from abridgeai.features.interviews.orchestrator.analysis import (
    AnswerAnalysis,
    fallback_analysis,
    parse_analysis_response,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

ANALYSIS_STAGE_NAME = "interview_analysis"


async def analyze_answer(
    db: AsyncSession,
    *,
    question_text: str,
    student_answer: str,
    turn_id: str,
    outcome_id: str | None = None,
    outcome_text: str | None = None,
    expected_evidence: Sequence[str] | None = None,
    common_misconceptions: Sequence[str] | None = None,
    supplementary_instructions: str | None = None,
    prior_claims: Sequence[str] | None = None,
    pipeline_run_id: UUID | None = None,
    gateway: LLMGateway | None = None,
) -> AnswerAnalysis:
    """Analyze a student answer into a structured :class:`AnswerAnalysis`.

    Never raises. On malformed model output, retries once; if still unusable
    (or the call fails), returns :func:`fallback_analysis` so the orchestrator
    advances neutrally without inventing a score.
    """
    answer = (student_answer or "").strip()
    if not answer:
        return fallback_analysis(turn_id)

    try:
        system_prompt = render_prompt("prompts/analysis_system.j2")
        user_prompt = json.dumps(
            {
                "current_question": question_text,
                "student_answer": answer,
                "turn_id": turn_id,
                "outcome": {"id": outcome_id or "", "text": outcome_text or ""},
                "expected_evidence": list(expected_evidence or []),
                "common_misconceptions": list(common_misconceptions or []),
                "supplementary_instructions": supplementary_instructions or "",
                "prior_claims": list(prior_claims or []),
            },
            ensure_ascii=False,
        )
        gateway = gateway or LLMGateway()

        for attempt in (1, 2):  # one retry on malformed output (Phase 3 spec)
            async with db.begin_nested():
                llm_result = await gateway.generate_json(
                    role=LLMRole.INTERVIEW_ANALYSIS,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    db=db,
                    stage_name=ANALYSIS_STAGE_NAME,
                    pipeline_run_id=pipeline_run_id,
                )
            payload = llm_result.content_json if isinstance(llm_result.content_json, dict) else None
            parsed = parse_analysis_response(payload, default_turn_id=turn_id)
            if parsed is not None:
                return parsed
            logger.warning("interview analysis: malformed payload (attempt %d/2)", attempt)
    except Exception:  # noqa: BLE001 — analysis is best-effort; never bubble to /respond
        logger.exception("interview analysis stage failed; using fallback")

    return fallback_analysis(turn_id)


__all__ = ["ANALYSIS_STAGE_NAME", "analyze_answer"]
