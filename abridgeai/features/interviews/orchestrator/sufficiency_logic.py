"""Fast sufficiency probe LLM call — the only analysis left on the turn path.

Wires :mod:`orchestrator.sufficiency` to the gateway. Conventions match
:mod:`orchestrator.extraction_logic` and :mod:`orchestrator.analysis_logic`:

* SAVEPOINT (``begin_nested``) around the gateway call so a failed
  ``ai_model_calls`` audit flush cannot poison the request session.
* Never raises. Any failure degrades to :func:`fallback_verdict`, which touches
  no outcome and therefore moves no coverage — a probe failure loses a turn's
  runtime guidance, it does not invent coverage and it does not break the turn.
* ONE call, no retry. The retry in ``analysis_logic._call_and_parse`` exists to
  rescue a rich schema from a malformed response; here the parser accepts any
  mapping, so a retry could only fire on a total non-mapping and would double
  the latency this stage exists to remove. The full extraction lands later and
  is authoritative anyway.

What this call is NOT given: the expected evidence, the misconceptions list, the
teacher's supplementary instructions, or any criterion text. The full analysis
DOES receive all of those; the probe holds only the question, the answer, and the
outcome statements, because a coverage bit does not need the rubric to be judged
and every field omitted here is a field an injection cannot reach.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING

from abridgeai.ai.llm import LLMGateway, LLMRole
from abridgeai.ai.prompts import render_prompt
from abridgeai.features.interviews.orchestrator.sufficiency import (
    SufficiencyVerdict,
    fallback_verdict,
    parse_sufficiency_response,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

SUFFICIENCY_STAGE_NAME = "interview_sufficiency"


async def probe_sufficiency(
    db: AsyncSession,
    *,
    question_text: str,
    student_answer: str,
    outcome_id: str | None = None,
    outcome_text: str | None = None,
    other_outcomes: Sequence[Mapping[str, str]] | None = None,
    pipeline_run_id: UUID | None = None,
    gateway: LLMGateway | None = None,
) -> SufficiencyVerdict:
    """Ask whether this answer moved the current outcome toward sufficiency.

    Returns a minimal verdict the caller projects onto coverage via
    :func:`orchestrator.sufficiency.verdict_to_evidence`. An empty answer
    short-circuits without spending a call.
    """
    answer = (student_answer or "").strip()
    if not answer:
        return fallback_verdict()

    try:
        system_prompt = render_prompt("prompts/sufficiency_system.j2")
        user_prompt = json.dumps(
            {
                "current_question": question_text,
                "student_answer": answer,
                "outcome": {"id": outcome_id or "", "text": outcome_text or ""},
                "other_outcomes": [
                    {"id": str(o["id"]), "text": str(o.get("text", ""))}
                    for o in (other_outcomes or [])
                    if isinstance(o, Mapping) and o.get("id")
                ],
            },
            ensure_ascii=False,
        )
        gateway = gateway or LLMGateway()
        async with db.begin_nested():
            llm_result = await gateway.generate_json(
                role=LLMRole.INTERVIEW_SUFFICIENCY,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                db=db,
                stage_name=SUFFICIENCY_STAGE_NAME,
                pipeline_run_id=pipeline_run_id,
            )
        payload = llm_result.content_json if isinstance(llm_result.content_json, dict) else None
        parsed = parse_sufficiency_response(payload)
    except Exception:  # noqa: BLE001 -- the probe is best-effort; never bubble to the turn
        logger.exception("interview sufficiency probe failed; using fallback")
        return fallback_verdict()

    if parsed is None:
        logger.warning("%s: malformed payload; using fallback", SUFFICIENCY_STAGE_NAME)
        return fallback_verdict()
    return parsed


__all__ = ["SUFFICIENCY_STAGE_NAME", "probe_sufficiency"]
