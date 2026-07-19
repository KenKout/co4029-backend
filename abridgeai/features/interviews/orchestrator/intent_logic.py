"""Intent classification LLM call (Phase 2).

Wires :mod:`orchestrator.intent` to the LLM gateway. The pure types, rules, and
parser live in that module; this one owns the I/O so it mirrors the follow-up
stage's proven conventions:

* Deterministic rules run FIRST — an unambiguous "repeat the question" never
  costs an LLM call and never mis-scores.
* The LLM call is wrapped in a SAVEPOINT (``begin_nested``) so a failed
  ``ai_model_calls`` audit flush can't poison the request session.
* ANY failure falls back to a deterministic verdict (ANSWER when no rule fired)
  so session progression NEVER blocks on the classifier.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from abridgeai.ai.llm import LLMGateway, LLMRole
from abridgeai.ai.prompts import render_prompt
from abridgeai.features.interviews.orchestrator.intent import (
    IntentClassification,
    StudentIntent,
    classify_by_rules,
    parse_intent_response,
)

if TYPE_CHECKING:
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

INTENT_STAGE_NAME = "interview_intent"


async def classify_intent(
    db: AsyncSession,
    *,
    question_text: str,
    student_utterance: str,
    pipeline_run_id: UUID | None = None,
    gateway: LLMGateway | None = None,
) -> IntentClassification:
    """Classify a student utterance's intent (rules first, LLM second).

    Never raises: on any LLM failure it falls back to the deterministic rule
    verdict, or to ANSWER when no rule matched — so the interview always
    progresses. High-confidence rule matches short-circuit the LLM entirely.
    """
    # 1) Deterministic high-precision rules. A confident match short-circuits.
    rule_verdict = classify_by_rules(student_utterance)
    if rule_verdict is not None and rule_verdict.confidence >= 0.9:
        return rule_verdict

    # 2) LLM classification (best-effort).
    try:
        system_prompt = render_prompt("prompts/intent_system.j2")
        user_prompt = json.dumps(
            {
                "current_question": question_text,
                "student_utterance": student_utterance,
            },
            ensure_ascii=False,
        )
        gateway = gateway or LLMGateway()
        async with db.begin_nested():
            llm_result = await gateway.generate_json(
                role=LLMRole.INTERVIEW_INTENT,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                db=db,
                stage_name=INTENT_STAGE_NAME,
                pipeline_run_id=pipeline_run_id,
            )
        payload = llm_result.content_json if isinstance(llm_result.content_json, dict) else None
        parsed = parse_intent_response(payload)
    except Exception:  # noqa: BLE001 — intent is best-effort; never bubble to /respond
        logger.exception("interview intent stage failed; using fallback")
        parsed = None

    if parsed is not None:
        return parsed

    # 3) Fallback: the rule verdict if any fired, else assume a genuine answer
    # (the safe default — treating a real answer as an answer never harms the
    # student, and progression continues).
    if rule_verdict is not None:
        return rule_verdict
    return IntentClassification(
        intent=StudentIntent.ANSWER,
        confidence=0.3,
        rationale="Classifier unavailable and no rule matched; defaulting to answer.",
        source="fallback",
    )


__all__ = ["INTENT_STAGE_NAME", "classify_intent"]
