"""Server-authoritative question advance for the native interview agent.

``interview_next_question`` is the tool the model is supposed to call before it
moves on. It kept not calling it: the interviewer said "Thanks, Duy. Let's look at
the next scenario…" and asked a new question in its own words, so server-side the
interview stayed on question one. The candidate's card, the "n of 3" counter and
the FOLLOW-UP labels all described a question nobody was answering, and the new
answer was graded against the previous question's outcome.

Prompt hardening was tried first (:mod:`agent_instructions` forbids narrated
advances, and the state note names the live question verbatim) and was not enough:
no ``voice.tool_refused`` appears in the logs for those sessions, because the gate
never got asked.

So the server stops asking. After each graded answer, if the live question is
RESOLVED — its outcome is ticked, the hint ladder is spent, or the follow-up budget
is spent — this module advances, publishes the snapshot the client renders from,
and rewrites the state note to "you have already been moved; ask this". The model's
transition then coincides with the server's by construction, whether or not it
calls the tool.

The tool stays: the model may still ask to move on at a moment the server would not
have chosen, and it is the only path when the question carries no linked outcome.
:attr:`InterviewUserdata.pending_new_question` is what stops the two from
double-spending one transition.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from abridgeai.features.interviews.orchestrator.tools import (
    current_question_resolved,
    reset_for_new_question,
)
from abridgeai.features.interviews.realtime import observability as obs

if TYPE_CHECKING:
    from abridgeai.features.interviews.realtime.agent_userdata import InterviewUserdata

logger = logging.getLogger(__name__)


class PoolSizer(Protocol):
    """The one thing this module needs from ``BankSelector``."""

    def remaining(self) -> int: ...


# A second advance closer together than this is the tail of ONE spoken answer,
# not a new one: the recognizer emits a final per pause, so a hesitant candidate
# produces several end-of-turn commits seconds apart (df269681: fragments at
# 0.8s, 1.6s and 6.9s gaps; both "advances" fired while the candidate was
# mid-sentence). The window is measured from the LAST advance and must cover
# the model's acknowledge-and-ask reply too — anything shorter re-arms inside
# the interviewer's own transition.
ADVANCE_COALESCE_WINDOW_S = 45.0


@dataclass(frozen=True)
class AdvanceOutcome:
    """What the server did with the turn that just finished.

    ``exhausted`` is distinct from a plain no-advance: the bank is empty, so the
    caller must not keep probing a question the interview has moved past — the
    model should close instead.
    """

    advanced: bool
    question_text: str | None = None
    exhausted: bool = False


async def advance_if_resolved(userdata: InterviewUserdata, selector: PoolSizer) -> AdvanceOutcome:
    """Move to the next bank question when the live one has nothing left to give.

    Deliberately narrower than the gate the tool consults. ``resolve_next_question``
    also stands open when there is no live question and when time is nearly up;
    both are right for a model ASKING permission and wrong as a trigger for the
    server acting on its own — the second would push a fresh question at the buzzer
    instead of closing.

    Mutates ``userdata`` and its runtime state in place, then publishes. Publishing
    is what persists the mutation (``publish_state`` saves, then snapshots), so a
    rejoin lands on the question the candidate was actually moved to.
    """
    state = userdata.state
    if state is None or userdata.finished or userdata.below_closing_threshold:
        return AdvanceOutcome(advanced=False)
    # Fragment guard: a spoken answer commits in pieces, and right after an
    # advance every piece looks "resolved" (the outcome it targeted was ticked
    # by the piece that legitimately advanced). Refuse a second advance inside
    # the coalesce window; the fragments still count as follow-up probes below.
    now = time.monotonic()
    if (
        userdata.last_advance_monotonic is not None
        and now - userdata.last_advance_monotonic < ADVANCE_COALESCE_WINDOW_S
    ):
        return AdvanceOutcome(advanced=False)
    if not current_question_resolved(
        state,
        current_outcome_id=state.current_outcome_id,
        max_follow_ups_per_question=userdata.max_follow_ups_per_question,
        max_hints=userdata.max_hints_per_question,
    ):
        return AdvanceOutcome(advanced=False)
    if userdata.questions_remaining <= 0:
        return AdvanceOutcome(advanced=False, exhausted=True)

    selected = userdata.select_next()
    if selected is None:
        return AdvanceOutcome(advanced=False, exhausted=True)

    state.current_outcome_id = selected.outcome_id
    reset_for_new_question(state)
    userdata.current_question_text = selected.prompt_text
    userdata.questions_remaining = selector.remaining()
    userdata.pending_new_question = True
    userdata.last_advance_monotonic = time.monotonic()
    # Any assistance marker still pending belongs to the question just left —
    # the next utterance is the NEW question's reading, and a stale "hint"
    # marker mislabeled it in the persisted transcript.
    userdata.pending_assistant_kind = None
    obs.emit(
        obs.EV_SERVER_ADVANCED,
        session_id=userdata.interview_session_id,
        outcome_id=selected.outcome_id,
        questions_remaining=userdata.questions_remaining,
    )
    try:
        await userdata.publish_state()
    except Exception:  # noqa: BLE001 -- the advance already happened; never fail the turn
        logger.exception(
            "publishing the advanced question failed (session=%s)",
            userdata.interview_session_id,
        )
    # Tell the client the agent's NEXT utterance is the new question, so the
    # live labeler badges it QUESTION instead of guessing from turn timing —
    # the committed question turn is stamped when the client APPLIES the
    # snapshot, which can sit after the speech began, and that race filed the
    # reading one question back as a FOLLOW-UP.
    try:
        await userdata.publish_agent_action(kind="question")  # type: ignore[call-arg]
    except Exception:  # noqa: BLE001 -- convenience channel; never fail the turn
        logger.warning(
            "publishing the question agent_action failed (session=%s)",
            userdata.interview_session_id,
        )
    return AdvanceOutcome(advanced=True, question_text=selected.prompt_text)


def count_follow_up(userdata: InterviewUserdata) -> None:
    """Charge one follow-up against the live question's budget.

    Nothing on the native path was charging it. ``current_question_follow_up_count``
    is incremented only by ``turn_state.apply_state_updates``, which belongs to the
    routed path, so on the native path the note reported "0/2" for the whole session
    and the budget's escape hatch never fired. A candidate whose outcome never ticks
    and who is never offered a hint therefore stayed on question one indefinitely —
    the tool's own refusal bound cannot save that, because it only counts calls the
    model actually makes.

    Charged when the turn did NOT advance, so the count measures probes the
    interviewer has already spent on this question. ``total_follow_up_count`` goes
    up with it, matching the routed path's ``turn_state`` accounting: the
    session-wide budget is what stops a long interview spending its whole clock
    probing question one.
    """
    if userdata.state is None:
        return
    userdata.state.current_question_follow_up_count += 1
    userdata.state.total_follow_up_count += 1


__all__ = ["AdvanceOutcome", "advance_if_resolved", "count_follow_up"]
