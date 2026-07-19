"""Interview follow-up stage orchestrator (T6.7).

Runs at SESSION runtime (called from ``take_session_step`` in T6.13), NOT in
the generation pipeline. Decides whether the candidate's answer to the
current question is shallow enough to warrant a single probing follow-up.

Design constraints (plan §6414-6453):

* Latency-sensitive: target < 3s P95 — student is waiting. Uses
  :class:`LLMRole.INTERVIEW_FOLLOWUP` which is bound to the *small* tier
  per ``ai/llm/roles.py`` (commit bf574d9).
* Single follow-up cap: this function returns ``None`` when the current
  question already has an assistant follow-up message persisted in
  ``interview_session_messages`` for the same session/session_question.
  Without the cap a student giving repeatedly shallow answers could chain
  follow-ups indefinitely.
* Pure return contract: the caller (services/take_session_step) handles
  persistence and audio generation. This stage only returns the follow-up
  text or ``None``.
* Best-effort: ANY exception inside the LLM call is swallowed and the
  stage returns ``None``. A student typing an answer must NEVER get a
  500 because the optional follow-up couldn't be generated — the answer
  itself must still be recorded and the session must still advance.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any
from uuid import UUID

from sqlalchemy import select

from abridgeai.ai.llm import LLMGateway, LLMRole
from abridgeai.ai.prompts import render_prompt
from abridgeai.features.interviews.ai.stages.followup.parsers import (
    FollowupVerdict,
    parse_followup_response,
)
from abridgeai.features.interviews.models import (
    InterviewSessionMessage,
    InterviewSessionQuestion,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from sqlalchemy.ext.asyncio import AsyncSession

    from abridgeai.features.interviews.models import (
        InterviewQuestion,
        InterviewSession,
    )


logger = logging.getLogger(__name__)

FOLLOWUP_STAGE_NAME = "interview_followup"


async def maybe_generate_followup(
    db: AsyncSession,
    *,
    session: InterviewSession,
    current_question: InterviewQuestion,
    student_answer: str,
    related_chunks: Sequence[Any] | None = None,
    pipeline_run_id: UUID | None = None,
    gateway: LLMGateway | None = None,
) -> str | None:
    """Return a follow-up question text, or ``None`` when no follow-up is needed.

    Returns ``None`` immediately (without an LLM call) when:

    * ``student_answer`` is empty after stripping whitespace.
    * The session/question already has at least one ``role='ai'`` message
      whose ``metadata_json.kind == 'followup'`` (single-cap rule).

    Otherwise calls the gateway with ``LLMRole.INTERVIEW_FOLLOWUP`` and
    returns the parsed follow-up text — or ``None`` when the LLM judges
    the answer sufficient, returns a malformed payload, or the call
    itself fails (missing credentials, provider down, etc.). The follow-
    up is a nice-to-have, never a hard dependency of the answer-record
    path.
    """

    answer = (student_answer or "").strip()
    if not answer:
        return None

    if await _has_existing_followup(db, session_id=session.id, question_id=current_question.id):
        return None

    chunk_views = [_chunk_for_prompt(chunk) for chunk in related_chunks or []]

    try:
        system_prompt = render_prompt("prompts/system.j2")
        user_prompt = json.dumps(
            {
                "current_question": current_question.prompt_text,
                "student_answer": answer,
                "related_chunks": chunk_views,
            },
            ensure_ascii=False,
        )

        gateway = gateway or LLMGateway()
        # SAVEPOINT around the gateway call: it flushes an ``ai_model_calls``
        # audit row, and any DB-level failure there (constraint violation,
        # provider-error row, ...) must roll back ONLY this savepoint. Without
        # it a failed flush poisons the request session (PendingRollbackError),
        # and the next flush in ``take_session_step`` 500s the whole /respond
        # request even though the follow-up is best-effort.
        async with db.begin_nested():
            llm_result = await gateway.generate_json(
                role=LLMRole.INTERVIEW_FOLLOWUP,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                db=db,
                stage_name=FOLLOWUP_STAGE_NAME,
                pipeline_run_id=pipeline_run_id,
            )

        payload = llm_result.content_json if isinstance(llm_result.content_json, dict) else None
        verdict: FollowupVerdict = parse_followup_response(payload)
    except Exception:  # noqa: BLE001 — follow-up is best-effort; never bubble to /respond
        logger.exception(
            "interview followup stage failed; advancing without follow-up (session=%s)",
            session.id,
        )
        return None

    if verdict.is_sufficient:
        return None
    return verdict.followup


async def _has_existing_followup(
    db: AsyncSession,
    *,
    session_id: UUID,
    question_id: UUID,
) -> bool:
    """Return True when this question already has a persisted follow-up.

    The single-cap rule (plan §6427) lives here so every caller — production
    and test — gets the same behaviour. We look for an ``ai`` role message
    that ties back to a ``session_question`` row pointing at
    ``question_id``, tagged ``metadata_json.kind = 'followup'``. Question
    bodies themselves carry ``kind = 'question'`` and are filtered out by
    the metadata predicate.
    """

    stmt = (
        select(InterviewSessionMessage.id)
        .join(
            InterviewSessionQuestion,
            InterviewSessionMessage.session_question_id == InterviewSessionQuestion.id,
        )
        .where(
            InterviewSessionMessage.session_id == session_id,
            InterviewSessionMessage.role == "ai",
            InterviewSessionQuestion.interview_question_id == question_id,
            InterviewSessionMessage.metadata_json["kind"].astext == "followup",
        )
        .limit(1)
    )
    result = await db.execute(stmt)
    return result.scalar_one_or_none() is not None


def _chunk_for_prompt(chunk: object) -> dict[str, Any]:
    """Coerce a chunk into the dict shape the user prompt expects."""

    chunk_id = getattr(chunk, "id", None) or getattr(chunk, "chunk_id", None)
    content = getattr(chunk, "content", "") or ""
    metadata = getattr(chunk, "metadata", None) or {}
    section_title = ""
    if isinstance(metadata, dict):
        section_title = str(metadata.get("section_title") or "")
    return {
        "id": str(chunk_id) if chunk_id is not None else "",
        "content": str(content),
        "section_title": section_title,
    }


__all__ = ["FOLLOWUP_STAGE_NAME", "maybe_generate_followup"]
