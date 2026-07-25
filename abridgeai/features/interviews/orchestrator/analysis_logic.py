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
from collections.abc import Mapping
from typing import TYPE_CHECKING

from abridgeai.ai.llm import LLMGateway, LLMRole
from abridgeai.ai.prompts import render_prompt
from abridgeai.features.interviews.orchestrator.analysis import (
    AnswerAnalysis,
    OutcomeEvidence,
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
    other_outcomes: Sequence[Mapping[str, str]] | None = None,
    pipeline_run_id: UUID | None = None,
    gateway: LLMGateway | None = None,
) -> AnswerAnalysis:
    """Analyze a student answer into a structured :class:`AnswerAnalysis`.

    Never raises. On malformed model output, retries once; if still unusable
    (or the call fails), returns :func:`fallback_analysis` so the orchestrator
    advances neutrally without inventing a score.

    ``other_outcomes`` enables *emergent* evidence: a list of
    ``{"id": ..., "text": ...}`` for the config's OTHER outcomes, letting the
    analyzer credit an outcome the question did not target when the candidate
    demonstrably covered it in passing. Omit/empty → v1 behaviour (only the
    linked outcome is visible). Attributions are sanitized against this
    whitelist by :func:`_sanitize_evidence`, so a hallucinated id can never
    create a phantom outcome.
    """
    answer = (student_answer or "").strip()
    if not answer:
        return fallback_analysis(turn_id)

    allowed_other = {
        str(o["id"]) for o in (other_outcomes or []) if isinstance(o, Mapping) and o.get("id")
    }
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
                "other_outcomes": [
                    {"id": str(o["id"]), "text": str(o.get("text", ""))}
                    for o in (other_outcomes or [])
                    if isinstance(o, Mapping) and o.get("id")
                ],
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
                _sanitize_evidence(
                    parsed, target_outcome_id=outcome_id, allowed_other=allowed_other
                )
                return parsed
            logger.warning("interview analysis: malformed payload (attempt %d/2)", attempt)
    except Exception:  # noqa: BLE001 — analysis is best-effort; never bubble to /respond
        logger.exception("interview analysis stage failed; using fallback")

    return fallback_analysis(turn_id)


def _sanitize_evidence(
    analysis: AnswerAnalysis,
    *,
    target_outcome_id: str | None,
    allowed_other: set[str],
) -> None:
    """Drop untrustworthy evidence attributions in place (defence in depth).

    The model is asked to only cite ids it was given, but a wrong id would
    silently create a phantom outcome in the coverage map, so we enforce it here
    rather than trusting the prompt:

    * evidence for the linked outcome → kept, ``secondary=False``;
    * evidence for an id in ``allowed_other`` → kept, ``secondary=True``
      (forced, regardless of what the model claimed);
    * anything else (unknown/invented/empty id) → dropped.

    When ``allowed_other`` is empty (emergent evidence disabled) this reduces to
    v1 behaviour: only target-outcome evidence survives.
    """
    target = str(target_outcome_id) if target_outcome_id else None
    kept: list[OutcomeEvidence] = []
    for ev in analysis.evidence:
        oid = (ev.outcome_id or "").strip()
        if target is not None and oid == target:
            ev.secondary = False
            kept.append(ev)
        elif oid and oid in allowed_other:
            ev.secondary = True
            kept.append(ev)
        elif not oid and target is not None:
            # Model omitted the id on a target-outcome item — repair it rather
            # than discarding real evidence for the question actually asked.
            ev.outcome_id = target
            ev.secondary = False
            kept.append(ev)
    analysis.evidence = kept


__all__ = ["ANALYSIS_STAGE_NAME", "analyze_answer"]
