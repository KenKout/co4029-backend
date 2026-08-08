"""Persistence for the native agent's conversation.

The routed path recorded every turn as a side effect of ``take_session_step``. The
native agent never calls it — the model holds the conversation in its own
``chat_ctx`` — so nothing was writing ``interview_session_messages`` on this path.
The consequence was silent and severe: a finished interview's stored transcript
held only the REST onboarding turns and the closing ceremony, and the evaluation
and gap report, which read that table, had no answers to judge.

The source is the SDK's ``conversation_item_added`` event, which fires for every
chat item the session commits — the interviewer's questions and follow-ups in the
words it actually used, and the candidate's answers, typed or spoken. That makes
it the one hook that cannot drift from what the model saw.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from abridgeai.core.db import get_sessionmaker
from abridgeai.features.interviews.realtime import observability as obs

if TYPE_CHECKING:
    from uuid import UUID

logger = logging.getLogger(__name__)

# `role` is CHECK-constrained to these on `interview_session_messages`.
_ROLE_BY_SDK: dict[str, str] = {
    "assistant": "ai",
    "user": "user",
    "system": "system",
}


async def _resolve_session_question(
    db: object, session_id: UUID, bank_question_id: UUID | None
) -> UUID | None:
    """Map a BANK question id onto this session's own question row.

    ``state.current_question_id`` holds an ``interview_questions`` id, but
    ``interview_session_messages.session_question_id`` is a foreign key to
    ``interview_session_questions`` — a different table. Writing the bank id
    straight through violated that constraint, so EVERY turn insert was rolled
    back and the stored transcript stayed empty.

    Created on demand: the native path never materialised these rows (the routed
    path did it inside ``take_session_step``), so the first time a question is
    recorded there is nothing to link to.
    """
    from sqlalchemy import func, select  # noqa: PLC0415

    from abridgeai.features.interviews.models import (  # noqa: PLC0415
        InterviewSessionQuestion,
    )

    if bank_question_id is None:
        return None
    found = await db.scalar(  # type: ignore[attr-defined]
        select(InterviewSessionQuestion.id).where(
            InterviewSessionQuestion.session_id == session_id,
            InterviewSessionQuestion.interview_question_id == bank_question_id,
        )
    )
    if found is not None:
        return found
    next_sequence = (
        await db.scalar(  # type: ignore[attr-defined]
            select(func.coalesce(func.max(InterviewSessionQuestion.sequence_no), 0)).where(
                InterviewSessionQuestion.session_id == session_id
            )
        )
        or 0
    ) + 1
    row = InterviewSessionQuestion(
        session_id=session_id,
        interview_question_id=bank_question_id,
        sequence_no=next_sequence,
    )
    db.add(row)  # type: ignore[attr-defined]
    await db.flush()  # type: ignore[attr-defined]
    return row.id


async def record_turn(
    session_id: UUID,
    *,
    role: str,
    text: str,
    session_question_id: UUID | None,
    kind: str,
) -> None:
    """Append one conversation turn. Never raises.

    A transcript row must never cost the candidate their turn, so every failure is
    logged and swallowed — but it IS logged, because a silent gap here is exactly
    what made a whole interview unevaluable.
    """
    from abridgeai.features.interviews.models import (  # noqa: PLC0415
        InterviewSessionMessage,
    )

    mapped = _ROLE_BY_SDK.get(role)
    content = text.strip()
    if mapped is None or not content:
        return
    try:
        async with get_sessionmaker()() as db:
            linked = await _resolve_session_question(db, session_id, session_question_id)
            db.add(
                InterviewSessionMessage(
                    session_id=session_id,
                    session_question_id=linked,
                    role=mapped,
                    content_text=content,
                    metadata_json={"kind": kind, "source": "native_agent"},
                )
            )
            await db.commit()
    except Exception as exc:  # noqa: BLE001 -- a transcript row is never worth a turn
        # Emitted, not merely logged: losing these rows leaves the evaluation with
        # nothing to grade, and the first time it happened the ONLY symptom was a
        # log line nobody was reading.
        obs.emit(
            obs.EV_TRANSCRIPT_WRITE_FAILED,
            session_id=session_id,
            role=role,
            error_class=type(exc).__name__,
        )
        logger.exception("failed to record native turn (session=%s)", session_id)


__all__ = ["record_turn"]
