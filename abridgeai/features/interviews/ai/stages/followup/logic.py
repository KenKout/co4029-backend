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

from sqlalchemy import func, select

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
from abridgeai.features.interviews.orchestrator.decision import (
    DEFAULT_MAX_FOLLOWUPS_PER_QUESTION,
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

# How many recent interviewer turns the follow-up stage is shown so it can avoid
# re-asking, in different words, something already covered. 5 matches the window
# SparkMe uses and keeps the prompt bounded regardless of session length: long
# enough to catch the realistic repeat (the last question or two), short enough
# that it cannot crowd out the current answer.
RECENT_QUESTION_WINDOW = 5


async def maybe_generate_followup(
    db: AsyncSession,
    *,
    session: InterviewSession,
    current_question: InterviewQuestion,
    student_answer: str,
    related_chunks: Sequence[Any] | None = None,
    pipeline_run_id: UUID | None = None,
    gateway: LLMGateway | None = None,
    max_follow_ups_per_question: int = DEFAULT_MAX_FOLLOWUPS_PER_QUESTION,
) -> str | None:
    """Return a follow-up question text, or ``None`` when no follow-up is needed.

    Returns ``None`` immediately (without an LLM call) when:

    * ``student_answer`` is empty after stripping whitespace.
    * The session/question already has the configured number of ``role='ai'``
      messages whose ``metadata_json.kind == 'followup'``.

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

    followup_count = await _existing_followup_count(
        db,
        session_id=session.id,
        question_id=current_question.id,
    )
    if followup_count >= max(0, max_follow_ups_per_question):
        return None

    chunk_views = [_chunk_for_prompt(chunk) for chunk in related_chunks or []]

    # Semantic repetition guard: give the model the recent interviewer turns so
    # it can avoid re-asking something already covered in different words.
    # Exact-id de-dup elsewhere cannot catch this (a generated follow-up has no
    # question id). Bounded so the prompt cannot grow with session length.
    recent_questions = await _recent_interviewer_questions(
        db, session_id=session.id, limit=RECENT_QUESTION_WINDOW
    )

    try:
        system_prompt = render_prompt("prompts/system.j2")
        user_prompt = json.dumps(
            {
                "current_question": current_question.prompt_text,
                "student_answer": answer,
                "related_chunks": chunk_views,
                "recent_interviewer_questions": recent_questions,
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


async def _recent_interviewer_questions(
    db: AsyncSession,
    *,
    session_id: UUID,
    limit: int,
) -> list[str]:
    """The last ``limit`` things the interviewer actually asked, newest last.

    Without this the follow-up stage sees ONE question and one answer, so it can
    re-ask — in different words — something already covered earlier in the
    session. Exact-id de-duplication in ``selection.py`` cannot catch that,
    because a generated follow-up has no question id at all.

    Cheap by construction: one indexed SELECT on ``session_id``, no LLM call.
    Only ``role='ai'`` messages are considered (both authored questions and
    earlier follow-ups — both are things the candidate has already been asked).
    """
    if limit <= 0:
        return []
    stmt = (
        select(InterviewSessionMessage.content_text)
        .where(
            InterviewSessionMessage.session_id == session_id,
            InterviewSessionMessage.role == "ai",
            InterviewSessionMessage.content_text.isnot(None),
        )
        .order_by(
            InterviewSessionMessage.created_at.desc(),
            InterviewSessionMessage.id.desc(),
        )
        .limit(limit)
    )
    rows = (await db.execute(stmt)).scalars().all()
    # Query is newest-first for the LIMIT; hand back chronological order so the
    # model reads them the way the conversation happened.
    return [t.strip() for t in reversed(rows) if t and t.strip()]


async def _existing_followup_count(
    db: AsyncSession,
    *,
    session_id: UUID,
    question_id: UUID,
) -> int:
    """Count persisted follow-ups for one question in a session.

    The transcript is legacy mode's durable per-question counter, so retries
    and fallback paths cannot reset the configured budget.
    """

    stmt = (
        select(func.count(InterviewSessionMessage.id))
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
    )
    result = await db.execute(stmt)
    return int(result.scalar_one() or 0)


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
