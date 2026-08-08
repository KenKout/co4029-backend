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


async def _no_finalize() -> None:
    """Default finalizer: do nothing.

    A no-op rather than a raise so a partially-wired session (a test, or a
    diagnostic harness) can still exercise the tools without ending a real
    interview by accident.
    """
    return


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
    below_closing_threshold: bool = False
    current_question_text: str | None = None
    # Seconds left on the session clock, refreshed each turn. None means the
    # session is UNTIMED — distinct from 0, and the reminder must not report it
    # as a deadline or the agent rushes a session that has no limit.
    time_remaining_seconds: int | None = None

    # True from the moment the SERVER advanced the question until the model has had
    # its turn to ask it. Two readers: the state note flips from "call the tool to
    # move on" to "you have already been moved", and `interview_next_question`
    # returns the question already selected instead of consuming another — the model
    # calling the tool right after a server advance must not skip a question.
    pending_new_question: bool = False

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

    # Injected by the runtime. `select_next` runs the deterministic scorer
    # (selection.py) and returns the chosen question, or None when the bank is
    # exhausted — the model never picks.
    select_next: Callable[[], SelectedQuestion | None] = lambda: None
    finalize_session: Callable[[], Awaitable[None]] = _no_finalize
    # Persist runtime state, then publish a snapshot. Injected by the runtime
    # after `session.start`, because the room the snapshot rides on does not
    # exist before that. A no-op default keeps the tools usable in tests and in a
    # partially-wired diagnostic harness.
    publish_state: Callable[[], Awaitable[None]] = _no_publish


__all__ = ["InterviewUserdata", "SelectedQuestion"]
