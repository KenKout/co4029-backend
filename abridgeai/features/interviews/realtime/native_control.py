"""Outbound control channel for the NATIVE interview agent.

The routed path publishes control events from ``InterviewAgent`` itself
(``session_runtime._publish_control``). The native agent cannot reuse that: it
does not inherit from ``InterviewAgent``, and the two paths deliberately keep
their LiveKit-SDK usage separate so the flag-off behaviour cannot drift. This
module is the native path's own publisher.

What it sends, and why the split matters:

* :meth:`ControlPublisher.ack` / :meth:`reject` are TURN-scoped, correlated by
  ``turn_key``. An ack means "your text arrived and is being worked on" — not
  "your answer has been graded". A streaming agent has no single instant where a
  turn's result becomes true, so a client that blocks its composer on a graded
  result waits for a message that will never come.
* :meth:`snapshot` is SESSION-scoped and absolute. It is the only thing that
  tells a client the question changed or the interview ended.

Ordering is the whole contract. ``seq`` is assigned and the bytes are sent under
one lock, so wire order always equals ``seq`` order and a client can discard
anything not newer than what it has already applied. Without the lock two
concurrent tool calls can interleave between assigning ``seq`` and awaiting the
send, and the client would roll its own state backwards.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from abridgeai.features.interviews.orchestrator.tools import build_progress_report
from abridgeai.features.interviews.realtime import text_protocol as tp

if TYPE_CHECKING:
    from uuid import UUID

    from abridgeai.features.interviews.realtime.agent_userdata import InterviewUserdata

logger = logging.getLogger(__name__)


def build_snapshot(userdata: InterviewUserdata) -> tp.StateSnapshot:
    """Project the session's runtime state onto the client-facing snapshot.

    Coverage counts come from :func:`build_progress_report` rather than being
    recomputed here, so the client's progress and the model's own
    ``interview_get_progress`` can never disagree about what is ticked.
    """
    state = userdata.state
    if state is None:
        return tp.StateSnapshot(
            current_question_id=None,
            current_question_text=userdata.current_question_text,
            question_number=0,
            questions_remaining=userdata.questions_remaining,
            questions_total=userdata.questions_total,
            outcomes_covered=0,
            outcomes_required=len(userdata.required_outcome_ids),
            is_finished=False,
            has_time_limit=userdata.time_remaining_seconds is not None,
            time_remaining_seconds=userdata.time_remaining_seconds,
        )
    report = build_progress_report(
        state,
        required_outcome_ids=list(userdata.required_outcome_ids),
        questions_remaining=userdata.questions_remaining,
        outcome_titles=dict(userdata.outcome_titles),
    )
    required = len(userdata.required_outcome_ids)
    return tp.StateSnapshot(
        current_question_id=(str(state.current_question_id) if state.current_question_id else None),
        current_question_text=userdata.current_question_text,
        # Both sides of the counter come from the pool, so "n of total" is
        # consistent by construction — counting `asked_question_ids` instead let
        # the two diverge.
        question_number=max(0, userdata.questions_total - userdata.questions_remaining),
        questions_remaining=userdata.questions_remaining,
        questions_total=userdata.questions_total,
        outcomes_covered=max(0, required - len(report.required_unticked)),
        outcomes_required=required,
        is_finished=userdata.finished,
        has_time_limit=userdata.time_remaining_seconds is not None,
        time_remaining_seconds=userdata.time_remaining_seconds,
    )


class ControlPublisher:
    """Serialised, sequence-numbered control egress for one interview session."""

    def __init__(self, session: object, *, interview_session_id: UUID) -> None:
        self._session = session
        self._interview_session_id = interview_session_id
        self._seq = 0
        self._lock = asyncio.Lock()

    async def ack(self, *, turn_key: str | None, turn_action: str) -> None:
        await self._publish(
            tp.ControlEvent(
                status=tp.ControlStatus.ACCEPTED,
                turn_key=turn_key,
                seq=0,
                turn_action=turn_action,
            )
        )

    async def reject(
        self, *, turn_key: str | None, turn_action: str, rejection: tp.TurnRejection
    ) -> None:
        await self._publish(
            tp.ControlEvent(
                status=tp.ControlStatus.REJECTED,
                turn_key=turn_key,
                seq=0,
                turn_action=turn_action,
                rejection=rejection,
            )
        )

    async def snapshot(self, state: tp.StateSnapshot) -> None:
        await self._publish(
            tp.ControlEvent(
                status=tp.ControlStatus.SNAPSHOT,
                turn_key=None,
                seq=0,
                snapshot=state,
            )
        )

    async def agent_action(self, *, kind: str) -> None:
        await self._publish(
            tp.ControlEvent(
                status=tp.ControlStatus.AGENT_ACTION,
                turn_key=None,
                seq=0,
                turn_action=kind,
            )
        )

    async def _publish(self, event: tp.ControlEvent) -> None:
        """Number and send one event. Never raises.

        ``seq`` is stamped here rather than by the caller so there is exactly one
        counter and one place that can get the ordering wrong.

        Control is a convenience channel: a failed publish must not abort a turn,
        and the next snapshot carries the full state anyway.
        """
        local = self._local_participant()
        if local is None:
            logger.debug(
                "no local participant; dropping control event (session=%s)",
                self._interview_session_id,
            )
            return
        async with self._lock:
            self._seq += 1
            event.seq = self._seq
            try:
                await local.send_text(event.to_json(), topic=tp.TOPIC_CONTROL)
            except Exception:  # noqa: BLE001 - client convenience channel; never fail a turn
                logger.warning(
                    "failed to publish control event (session=%s, status=%s)",
                    self._interview_session_id,
                    event.status.value,
                )

    def _local_participant(self) -> object | None:
        """Reach the room's local participant.

        ``AgentSession`` exposes NO ``.room``; the room hangs off ``room_io``,
        which only exists after ``session.start``. A previous version of the
        routed path looked for ``session.room`` and silently dropped every
        control event, and its unit tests missed it because the fake session had
        the attribute the real one lacks.
        """
        room_io = getattr(self._session, "room_io", None)
        room = getattr(room_io, "room", None)
        return getattr(room, "local_participant", None)


__all__ = ["ControlPublisher", "build_snapshot"]
