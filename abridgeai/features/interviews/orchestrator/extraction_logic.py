"""Quarantined extraction LLM call.

Wires :mod:`orchestrator.extraction` to the gateway. This is the ONLY runtime
call that receives a student's raw answer, and it is deliberately starved: the
prompt carries the current question and the utterance, nothing else. Anything
an injection could ask this call to leak, it does not have.

Conventions match :mod:`orchestrator.intent_logic` and
:mod:`orchestrator.analysis_logic`:

* Deterministic rules run FIRST (via ``classify_by_rules``); a confident
  non-academic verdict short-circuits the LLM entirely, exactly as the intent
  stage did before the split — so "repeat the question" still costs nothing.
* SAVEPOINT (``begin_nested``) around the gateway call so a failed
  ``ai_model_calls`` audit flush cannot poison the request session.
* Never raises. Any failure degrades to :func:`fallback_claims`, which carries
  no claims — the matcher then produces a not-assessable analysis rather than
  inventing evidence.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from abridgeai.ai.llm import LLMGateway, LLMRole
from abridgeai.ai.prompts import render_prompt
from abridgeai.features.interviews.orchestrator.claim_filter import filter_claims
from abridgeai.features.interviews.orchestrator.extraction import (
    AnswerClaims,
    fallback_claims,
    parse_answer_claims,
)
from abridgeai.features.interviews.orchestrator.intent import (
    NON_ACADEMIC_INTENTS,
    classify_by_rules,
)

if TYPE_CHECKING:
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

EXTRACTION_STAGE_NAME = "interview_extraction"


async def extract_answer_claims(
    db: AsyncSession,
    *,
    question_text: str,
    student_answer: str,
    pipeline_run_id: UUID | None = None,
    gateway: LLMGateway | None = None,
) -> AnswerClaims:
    """Reduce one raw student turn to bounded, screened claims.

    The returned claims have already passed :func:`filter_claims`, so callers
    may hand them to the rubric-bearing matcher without further screening.
    """
    # 1) Deterministic short-circuit. A confident non-academic verdict means no
    # analysis will run downstream, so there is nothing to extract and no
    # reason to spend a call.
    rule_verdict = classify_by_rules(student_answer)
    if (
        rule_verdict is not None
        and rule_verdict.confidence >= 0.9
        and rule_verdict.intent in NON_ACADEMIC_INTENTS
    ):
        return AnswerClaims(intent=rule_verdict, claims=[], source="rules")

    parsed: AnswerClaims | None = None
    try:
        system_prompt = render_prompt("prompts/extraction_system.j2")
        user_prompt = json.dumps(
            {
                "current_question": question_text,
                "student_utterance": student_answer,
            },
            ensure_ascii=False,
        )
        gateway = gateway or LLMGateway()
        async with db.begin_nested():
            llm_result = await gateway.generate_json(
                role=LLMRole.INTERVIEW_EXTRACTION,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                db=db,
                stage_name=EXTRACTION_STAGE_NAME,
                pipeline_run_id=pipeline_run_id,
            )
        payload = llm_result.content_json if isinstance(llm_result.content_json, dict) else None
        parsed = parse_answer_claims(payload)
    except Exception:  # noqa: BLE001 -- extraction is best-effort; never bubble to /respond
        logger.exception("interview extraction stage failed; using fallback")
        parsed = None

    if parsed is None:
        return fallback_claims(rule_verdict)

    # 2) The boundary. Nothing reaches the matcher without passing this.
    return filter_claims(parsed)


__all__ = ["EXTRACTION_STAGE_NAME", "extract_answer_claims"]
