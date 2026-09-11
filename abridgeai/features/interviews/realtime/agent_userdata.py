"""Per-session state carried by ``AgentSession`` for the interview agent.

Mirrors ``Userdata`` in ``reference/agents/examples/hotel_receptionist/common.py``:
the single session-scoped handle every ``@function_tool`` reaches through
``ctx.userdata``.

Separated from ``agent_tools`` so the tool module stays free of construction
concerns and can be imported by tests without a live session.

The two callables are the seam that keeps the gate logic pure: ``select_next``
and ``finalize_session`` are injected by the runtime, so ``agent_tools`` performs
no DB access and ``orchestrator/tools.py`` stays property-testable with plain
objects.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol
from uuid import UUID

if TYPE_CHECKING:
    from abridgeai.features.interviews.orchestrator.state import InterviewRuntimeStateData


class SelectedQuestion(Protocol):
    """What ``select_next`` must return for the advance tool to use it."""

    @property
    def outcome_id(self) -> str: ...

    @property
    def prompt_text(self) -> str: ...


async def _no_finalize() -> bool:
    return False
    """Default finalizer: do nothing.

    A no-op rather than a raise so a partially-wired session (a test, or a
    diagnostic harness) can still exercise the tools without ending a real
    interview by accident.
    """
    return None


async def _no_publish() -> None:
    return


@dataclass
class InterviewUserdata:
    interview_session_id: UUID
    student_id: UUID
    language: str = "en"

    # Runtime state loaded from the DB. The tools MUTATE this (hint_level,
    # follow-up counters, refusal counters); the runtime persists it after the
    # turn, so the object here must be the same one the runtime saves — not a copy.
    state: InterviewRuntimeStateData | None = None

    required_outcome_ids: list[str] = field(default_factory=list)
    outcome_titles: dict[str, str] = field(default_factory=dict)
    questions_remaining: int = 0
    # The question pool's size, fixed for the session. The counter's denominator
    # comes from here so it cannot drift against `questions_remaining`.
    questions_total: int = 0
    max_follow_ups_per_question: int = 2
    max_hints_per_question: int = 3
    # DEPRECATED field kept for one release: the value was snapshotted ONCE at
    # join, so a session that crossed the closing threshold mid-interview never
    # saw the agent's urgency change. All readers must call
    # :meth:`below_closing_threshold_now` instead. New code: do not read this.
    below_closing_threshold: bool = False
    # Total configured duration in SECONDS (config.time_limit_minutes * 60),
    # injected at setup so the live fraction can be derived anywhere. 0/None =
    # untimed.
    total_duration_seconds: int | None = None
    current_question_text: str | None = None
    # Seconds left on the session clock AS OF ``clock_read_monotonic``. None means
    # the session is UNTIMED — distinct from 0, and the reminder must not report it
    # as a deadline or the agent rushes a session that has no limit.
    #
    # Read this through :meth:`remaining_seconds_now`, never directly, unless you
    # genuinely want the value at read time. It is a SNAPSHOT: nothing refreshes it
    # per turn, so every client snapshot re-sent the countdown the session had at
    # join. Ten minutes into a thirty-minute interview a rejoining candidate saw
    # nearly thirty minutes again while the backend still finished on the real
    # deadline — a timer that lies in the candidate's favour, then stops them.
    time_remaining_seconds: int | None = None
    # ``time.monotonic()`` when ``time_remaining_seconds`` was read. Monotonic, not
    # wall clock: a container NTP step must not move a candidate's deadline.
    clock_read_monotonic: float | None = None
    # True from the moment the SERVER advanced the question until the model has had
    # its turn to ask it. Two readers: the state note flips from "call the tool to
    # move on" to "you have already been moved", and `interview_next_question`
    # returns the question already selected instead of consuming another — the model
    # calling the tool right after a server advance must not skip a question.
    pending_new_question: bool = False

    # When the server LAST advanced (monotonic seconds). A spoken answer
    # arrives as several end-of-turn commits (the recognizer emits a final per
    # pause), and a freshly-advanced question makes every one of them look
    # "resolved" — the tail of the candidate's own sentence then advanced the
    # interview AGAIN within seconds (production df269681: two advances in 24s,
    # both while the candidate was mid-sentence). `fold_turn` reads this to
    # refuse a second advance inside the window.
    last_advance_monotonic: float | None = None

    # The bank question the candidate's CURRENT answer is answering, snapshotted at
    # fold time — BEFORE the server advances. The transcript handler reads this for
    # user items because `state.current_question_id` has already moved on by then;
    # using the live value filed every answer one question ahead, which is how a
    # session that visibly answered everything scored "1/3 answered".
    answered_question_id: str | None = None

    # True once the session has been submitted for evaluation, so a snapshot can
    # tell the client the interview is over. Not derived from `state.phase`: the
    # hard stop and the end tool both finalize through `finalize_session`, and
    # only that one flag marks the point of no return for both.
    finished: bool = False

    # The assistance kind the agent's NEXT assistant utterance delivers
    # ("hint" / "clarification"), set when the server grants assistance and
    # consumed by the transcript recorder. Persisted rows otherwise all carry
    # kind="question", so after a reload every hint in the history rendered as a
    # FOLLOW-UP. None means the next utterance is ordinary interview speech.
    pending_assistant_kind: str | None = None

    # Injected by the runtime. `select_next` runs the deterministic scorer
    # (selection.py) and returns the chosen question, or None when the bank is
    # exhausted — the model never picks.
    select_next: Callable[[], SelectedQuestion | None] = lambda: None
    finalize_session: Callable[[], Awaitable[bool]] = _no_finalize
    # Persist runtime state, then publish a snapshot. Injected by the runtime
    # after `session.start`, because the room the snapshot rides on does not
    # exist before that. A no-op default keeps the tools usable in tests and in
    # a partially-wired diagnostic harness.
    publish_state: Callable[[], Awaitable[None]] = _no_publish
    # Persist runtime state WITHOUT publishing a snapshot. For the mutations the
    # client cannot see: the hint ladder and the follow-up budgets. Those live only
    # in runtime state, and a tool that changed one and did not save it lost it on
    # a restart — handing the candidate back a rung or a probe they had spent.
    # `publish_state` would work but tells the client nothing changed, on every
    # probe of every question.
    save_state: Callable[[], Awaitable[None]] = _no_publish
    # Tell the client the agent's next utterance is assistance of this kind, so
    # the live transcript can badge it. Same injection/no-op pattern as
    # `publish_state`.
    publish_agent_action: Callable[[str], Awaitable[None]] = lambda kind: _no_publish()
    # The turn_key of the typed answer whose durable receipt was JUST persisted
    # and whose SDK echo (conversation_item_added) has not arrived yet. The
    # transcript handler consumes it to suppress that one echo — the receipt
    # already is the transcript row. See `TurnIntake.arm_echo`.
    echo_turn_key: str | None = None

    def remaining_seconds_now(self) -> int | None:
        """The countdown as of NOW, derived from the snapshot and elapsed time.

        None for an untimed session (nothing to count down), and never negative:
        past the deadline the honest answer is 0 — the hard stop owns what happens
        next, and a negative number renders as a nonsense timer.
        """
        if self.time_remaining_seconds is None:
            return None
        if self.clock_read_monotonic is None:
            return self.time_remaining_seconds
        elapsed = time.monotonic() - self.clock_read_monotonic
        return max(0, int(self.time_remaining_seconds - elapsed))

    def below_closing_threshold_now(self) -> bool:
        """Whether the session is in its closing window AS OF NOW.

        Derived live: remaining / total against the SHARED orchestrator closing
        fraction (``DecisionInputs.closing_time_fraction``) — no duplicate
        literal. Untimed sessions (or a nonsensical total) are never "closing".
        This replaces the join-time ``below_closing_threshold`` bool, which was
        frozen at setup and never crossed mid-interview.
        """
        remaining = self.remaining_seconds_now()
        if remaining is None or not self.total_duration_seconds or self.total_duration_seconds <= 0:
            return False
        from abridgeai.features.interviews.orchestrator.decision import (  # noqa: PLC0415
            DecisionInputs,
        )

        return (remaining / self.total_duration_seconds) <= DecisionInputs.closing_time_fraction


__all__ = ["InterviewUserdata", "SelectedQuestion"]
