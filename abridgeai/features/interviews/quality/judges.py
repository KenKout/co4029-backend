"""LLM-as-judge calls for the post-hoc quality metrics.

Offline only — invoked from the quality CLI over a stored transcript, never from
a live turn. Both judges are per-utterance (one call per interviewer question),
which is the expensive-but-honest shape: a single whole-transcript verdict cannot
tell you WHICH question leaked the answer, and an actionable list is the entire
point.

Cost control is the caller's job: :func:`judge_contingency` and
:func:`judge_leading` both take ``limit`` so the CLI can cap how many utterances
are judged per session, and the CLI requires an explicit budget.

Failures never raise. A judge call that errors or returns garbage is recorded in
the report's ``errors`` list and the turn is EXCLUDED from the mean (see
``scoring`` for why we do not score it 1).
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from abridgeai.ai.llm import LLMGateway, LLMRole
from abridgeai.ai.prompts import render_prompt
from abridgeai.features.interviews.quality.scoring import (
    ContingencyReport,
    JudgedTurn,
    LeadingReport,
    parse_verdict,
)
from abridgeai.features.interviews.quality.transcript import (
    build_qa_pairs,
    followup_pairs,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from sqlalchemy.ext.asyncio import AsyncSession

    from abridgeai.features.interviews.quality.transcript import QAPair, TranscriptTurn

logger = logging.getLogger(__name__)

CONTINGENCY_STAGE_NAME = "interview_quality_contingency"
LEADING_STAGE_NAME = "interview_quality_leading"


async def judge_contingency(
    db: AsyncSession,
    *,
    session_id: str,
    turns: list[TranscriptTurn],
    limit: int | None = None,
    gateway: LLMGateway | None = None,
) -> ContingencyReport:
    """Score how well each follow-up builds on the answer before it.

    Only follow-ups are judged (same ``session_question_id`` as the preceding
    answer); a deliberate move to the next question is not a contingency failure.
    An interview with NO follow-ups sets ``no_followups=True`` — that is the
    scripted-checklist signal, not missing data.
    """
    pairs = followup_pairs(turns)
    if not pairs:
        return ContingencyReport(session_id=session_id, no_followups=True)

    system_prompt = render_prompt("prompts/contingency_system.j2")
    gateway = gateway or LLMGateway()
    judged, errors = await _judge_pairs(
        db,
        pairs=pairs[:limit] if limit else pairs,
        system_prompt=system_prompt,
        stage_name=CONTINGENCY_STAGE_NAME,
        gateway=gateway,
        payload_builder=lambda p: {
            "student_previous_answer": p.preceding_student_text or "",
            "interviewer_followup": p.interviewer_text,
        },
    )
    return ContingencyReport(session_id=session_id, turns=judged, errors=errors)


async def judge_leading(
    db: AsyncSession,
    *,
    session_id: str,
    turns: list[TranscriptTurn],
    limit: int | None = None,
    gateway: LLMGateway | None = None,
) -> LeadingReport:
    """Score every interviewer question for handing the student the answer.

    Unlike contingency this judges ALL interviewer utterances, not just
    follow-ups: a scripted question can leak the answer just as easily as a
    probe. The student's reply is supplied so the judge can tell a genuine
    answer from an echo of the question.
    """
    pairs = build_qa_pairs(turns)
    if not pairs:
        return LeadingReport(session_id=session_id)

    system_prompt = render_prompt("prompts/leading_system.j2")
    gateway = gateway or LLMGateway()
    judged, errors = await _judge_pairs(
        db,
        pairs=pairs[:limit] if limit else pairs,
        system_prompt=system_prompt,
        stage_name=LEADING_STAGE_NAME,
        gateway=gateway,
        payload_builder=lambda p: {
            "interviewer_question": p.interviewer_text,
            "student_answer": p.student_text,
            "student_previous_answer": p.preceding_student_text or "",
        },
    )
    return LeadingReport(session_id=session_id, turns=judged, errors=errors)


async def _judge_pairs(
    db: AsyncSession,
    *,
    pairs: list[QAPair],
    system_prompt: str,
    stage_name: str,
    gateway: LLMGateway,
    payload_builder: Callable[[QAPair], dict[str, str]],
) -> tuple[list[JudgedTurn], list[str]]:
    """Judge each pair with one LLM call; collect verdicts and errors.

    Sequential on purpose: this runs offline where wall-clock does not matter,
    and it keeps spend predictable and interruptible (Ctrl-C stops after the
    current call rather than after N in-flight ones).
    """
    judged: list[JudgedTurn] = []
    errors: list[str] = []
    for pair in pairs:
        try:
            user_prompt = json.dumps(payload_builder(pair), ensure_ascii=False)
            result = await gateway.generate_json(
                role=LLMRole.INTERVIEW_QUALITY_JUDGE,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                db=db,
                stage_name=stage_name,
            )
            payload = result.content_json if isinstance(result.content_json, dict) else None
            verdict = parse_verdict(payload, message_id=pair.interviewer_message_id)
            if verdict is None:
                errors.append(f"{pair.interviewer_message_id}: unparseable judge verdict")
                continue
            judged.append(verdict)
        except Exception as exc:  # noqa: BLE001 — one bad call must not abort the run
            logger.exception("quality judge failed for %s", pair.interviewer_message_id)
            errors.append(f"{pair.interviewer_message_id}: {type(exc).__name__}: {exc}")
    return judged, errors


__all__ = [
    "CONTINGENCY_STAGE_NAME",
    "LEADING_STAGE_NAME",
    "judge_contingency",
    "judge_leading",
]
