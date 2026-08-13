"""LiveKit runtime for the NATIVE (multiturn) interview agent.

Sibling of :mod:`session_runtime`, not a replacement: that module holds the
ROUTED agent (no LLM, every turn routed out to the text brain, ``StopResponse``
suppressing the default reply) and stays the behaviour behind
``interview_native_agent_enabled=False``. This module holds the conversational
agent that runs when the flag is ON.

Three differences carry the whole design:

* The session is built by :func:`agent_session.build_native_session`, which
  passes ``llm=``. The agent therefore holds ONE ``chat_ctx`` and reads its own
  conversation, instead of routing each turn to a stateless per-stage call.
* :meth:`NativeInterviewAgent.on_user_turn_completed` does NOT raise
  ``StopResponse``. The routed agent must (it has no LLM, so the default reply
  would produce silence); here the LLM generating the reply IS the feature.
* Typed turns use the SDK's DEFAULT text callback. The routed path must override
  it because the default calls ``generate_reply()`` against an absent LLM; this
  path has one, so the default is the correct behaviour and
  :func:`agent_session.room_options_for_mode` deliberately leaves
  ``text_input_cb`` unset.

Per the version-isolation decision at the top of :mod:`session_runtime`, all
LiveKit-SDK usage for this path lives here.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from livekit import (
    rtc,  # type: ignore[attr-defined]  # rtc is a lazy submodule; livekit ships no stubs
)
from livekit.agents import Agent, AgentSession, ChatContext, get_job_context

from abridgeai.features.interviews.orchestrator.intent import (
    IntentClassification,
    StudentIntent,
)
from abridgeai.features.interviews.orchestrator.shadow import shadow_check_turn
from abridgeai.features.interviews.realtime import observability as obs
from abridgeai.features.interviews.realtime.agent_context import (
    end_on_user_turn,
    seed_onboarding_history,
)
from abridgeai.features.interviews.realtime.agent_instructions import build_instructions
from abridgeai.features.interviews.realtime.agent_session import (
    build_native_session,
    build_state_reminder,
    room_options_for_mode,
)
from abridgeai.features.interviews.realtime.agent_tools import InterviewToolsMixin
from abridgeai.features.interviews.realtime.native_advance import (
    advance_if_resolved,
    count_follow_up,
)
from abridgeai.features.interviews.realtime.native_control import ControlPublisher, build_snapshot
from abridgeai.features.interviews.realtime.native_text_input import make_text_input_cb
from abridgeai.features.interviews.realtime.native_transcript import record_turn

if TYPE_CHECKING:
    from livekit.agents import ChatMessage, JobContext

    from abridgeai.core.config import Settings
    from abridgeai.features.interviews.realtime.agent_userdata import InterviewUserdata
    from abridgeai.features.interviews.realtime.native_bridge import NativeSetup

logger = logging.getLogger(__name__)

# Wall-clock budget the hard stop allows per remaining question when the config
# sets no time limit. Generous on purpose: this is a backstop against a model
# that never ends, not a pacing mechanism, and cutting a productive interview
# short is a worse failure than running a few minutes long.
_SECONDS_PER_REMAINING_QUESTION = 240.0
# Never arm the stop closer than this, even with no questions left: a session
# joined at the very end of its window still deserves a closing exchange rather
# than an immediate teardown.
_MIN_HARD_STOP_SECONDS = 120.0
# Fired past the deadline, not at it: `submit_session` refuses a `timed_out`
# reason until the limit has actually elapsed (with a 2s scheduling-tolerance
# buffer of its own), and a hard stop that fires early therefore cannot submit.
_DEADLINE_GRACE_SECONDS = 5.0
# Used only when neither a time limit nor a question budget is known. Matches
# ``interview_voice_idle_timeout_minutes``' order of magnitude: long enough that
# no real interview trips it, short enough that an abandoned room still closes.
_DEFAULT_HARD_STOP_SECONDS = 45.0 * 60.0
# Bound on waiting for the closing to finish playing out before teardown, so a
# stuck SpeechHandle cannot hang the job. Mirrors ``session_runtime``.
_CLOSING_PLAYOUT_TIMEOUT_S = 30.0


def hard_stop_deadline_seconds(
    *, time_remaining_seconds: int | None, questions_remaining: int
) -> float:
    """Seconds until the server-side hard stop fires.

    The config's time limit is the authoritative deadline when one exists; the
    question budget only applies to an UNTIMED session. Using ``min`` of the two
    let a small question pool end a 30-minute interview after 8 — the reported
    session was killed mid-conversation, and the timed-out submission was then
    REJECTED because the real limit had not elapsed. Pure so the arithmetic is
    testable without a room.
    """
    if time_remaining_seconds is not None:
        return max(_MIN_HARD_STOP_SECONDS, float(time_remaining_seconds) + _DEADLINE_GRACE_SECONDS)
    budget = max(0.0, questions_remaining) * _SECONDS_PER_REMAINING_QUESTION
    if budget <= 0.0:
        return _DEFAULT_HARD_STOP_SECONDS
    return max(_MIN_HARD_STOP_SECONDS, budget)


@dataclass
class HardStopPlan:
    """What the hard stop needs to close a session the model never ends.

    ``close`` submits the session and returns the canonical ceremony closing to
    speak — the same ``orchestration_bridge.finalize_session`` the
    ``interview_end_interview`` tool reaches, so both routes submit once, the
    same way.
    """

    deadline_seconds: float
    close: Callable[[], Awaitable[str | None]]
    interview_session_id: UUID


class HardStopTimer:
    """The third anti-deadlock layer: a wall clock the model cannot argue with.

    The other two are the bounded refusal counters in ``orchestrator/tools.py``
    (``MAX_END_REFUSALS`` / ``MAX_ADVANCE_REFUSALS``), and both only help a model
    that TRIES to advance or end. A model that simply keeps talking defeats them,
    which is why this layer exists outside the model's reach entirely.

    :meth:`finalize_once` is the shared entry for both routes, so a session
    cannot be submitted twice when the model ends just as the timer fires.
    """

    def __init__(self, plan: HardStopPlan, *, session: AgentSession) -> None:
        self._plan = plan
        self._session = session
        self._task: asyncio.Task[None] | None = None
        self._finalized = False
        # Set by the runtime once the publisher exists. Runs after EITHER route
        # submits, so "the interview is over" reaches the client whether the model
        # ended it or the wall clock did.
        self.on_finalized: Callable[[], Awaitable[None]] | None = None

    @property
    def fired(self) -> bool:
        return self._finalized

    def start(self) -> None:
        self._task = asyncio.create_task(self._run())

    def cancel(self) -> None:
        if self._task is not None:
            self._task.cancel()
            self._task = None

    async def finalize_once(self, inner: Callable[[], Awaitable[None]]) -> None:
        """Run ``inner`` at most once per session, and disarm the timer.

        Wrapping the tool's finalizer rather than guarding inside it keeps
        ``agent_tools`` unaware that a timer exists.
        """
        if self._finalized:
            return
        self._finalized = True
        self.cancel()
        await inner()
        await self._announce_finished()

    async def _announce_finished(self) -> None:
        """Tell the client the interview is over. Never raises."""
        if self.on_finalized is None:
            return
        try:
            await self.on_finalized()
        except Exception:  # noqa: BLE001 -- a submitted session must not fail on a notification
            logger.exception(
                "failed to announce finish (session=%s)", self._plan.interview_session_id
            )

    async def _run(self) -> None:
        try:
            await asyncio.sleep(self._plan.deadline_seconds)
        except asyncio.CancelledError:
            return
        await self._stop()

    async def _stop(self) -> None:
        if self._finalized:
            return
        self._finalized = True
        # Submit BEFORE speaking. The candidate's answers are already graded
        # evidence at this point, and a transport failure during the goodbye must
        # not be the reason a completed interview was never submitted.
        closing: str | None = None
        try:
            closing = await self._plan.close()
        except Exception:
            logger.exception(
                "hard stop failed to finalize (session=%s)",
                self._plan.interview_session_id,
            )
        obs.emit(
            obs.EV_CLOSING_EMITTED,
            session_id=self._plan.interview_session_id,
            reason="hard_stop_deadline",
            closing_chars=len(closing or ""),
        )
        await self._announce_finished()
        handle = None
        if closing:
            handle = self._session.say(closing, allow_interruptions=False)
            await handle
        await _await_playout(handle)
        job = get_job_context(required=False)
        if job is not None:
            job.shutdown(reason="interview_hard_stop")


async def _await_playout(handle: object) -> None:
    """Wait for a closing utterance to reach the candidate, bounded.

    Awaiting ``say()`` only means the speech was scheduled and generated, not
    that the audio arrived — tearing the room down here would cut the closing.
    Never raises: a stuck handle must not block shutdown.
    """
    waiter = getattr(handle, "wait_for_playout", None)
    if waiter is None:
        return
    try:
        await asyncio.wait_for(waiter(), timeout=_CLOSING_PLAYOUT_TIMEOUT_S)
    except Exception:  # noqa: BLE001 - playout is best-effort; shutdown proceeds
        logger.warning("closing playout wait failed; shutting down anyway")


def _rejoin_question_text(question: str, language: str) -> str:
    """A short lead-in plus the verbatim question, for a mid-interview rejoin.

    The lead-in keeps the re-read out of the client's verbatim-dedup path (the
    pinned card already shows the bare question) and reads as a person re-stating
    the question, not a form being read aloud.
    """
    if (language or "en").lower().startswith("vi"):
        return f"Để tôi nhắc lại câu hỏi: {question}"
    return f"Let me repeat the question: {question}"


async def _re_read_question(session: AgentSession, question: str, language: str) -> None:
    """Re-speak the current question after a rejoin. Best-effort.

    Fire-and-forget from the participant-connect handler; a failed TTS must not
    take the session down.
    """
    try:
        handle = session.say(
            _rejoin_question_text(question, language), allow_interruptions=False
        )
        await handle
    except Exception:  # noqa: BLE001 -- a failed re-read must not cost the session
        logger.exception("re-read question on rejoin failed")


class NativeInterviewAgent(InterviewToolsMixin, Agent):
    """The conversational interviewer: its own words, the server's authority.

    ``InterviewToolsMixin`` comes first in the MRO so its four
    ``@function_tool`` methods are the ones the SDK discovers on the instance.
    """

    def __init__(self, *, instructions: str, chat_ctx: ChatContext, setup: NativeSetup) -> None:
        super().__init__(instructions=instructions, chat_ctx=chat_ctx)
        self._setup = setup
        self._base_instructions = instructions

    @property
    def userdata(self) -> InterviewUserdata:
        return self._setup.userdata

    async def refresh_state_note(self, *, opening: bool = False) -> None:
        """Rewrite the SYSTEM instructions to carry the current state note.

        The note has to live at ``messages[0]``. Appended to ``chat_ctx`` as a
        mid-conversation system message — which is what this used to do — Gemini
        effectively discards it: probed against the live gateway, the same note at
        ``messages[0]`` produced the question it names while mid-conversation it
        produced a generic greeting. Everything the note enforces (which question
        is live, the budgets, whether advancing is permitted) was being thrown away.

        Never allowed to fail the turn: a stale note is recoverable, a dropped
        reply is not.
        """
        note = build_state_reminder(self._setup.userdata, opening=opening)
        if not note:
            return
        try:
            await self.update_instructions(f"{self._base_instructions}\n\n{note}")
        except Exception:  # noqa: BLE001 -- a stale note must not cost the reply
            logger.exception(
                "refreshing the state note failed (session=%s)",
                self._setup.userdata.interview_session_id,
            )

    async def on_enter(self) -> None:
        """Open the interview in the interviewer's OWN words.

        This used to ``say()`` the bank text verbatim, on the grounds that the REST
        ceremony had just announced "here is your first question" and the model had
        no conversation yet to ground a rewording in. That ceremony line is gone —
        the agent owns its opening — and reading the bank text back was the one
        place the interviewer sounded like a form being read aloud. It also made
        question one the ONLY question missing from the candidate's transcript: a
        verbatim reading is indistinguishable from the card pinned above it, so the
        client drops it as a duplicate, while every paraphrased later question shows.

        The question travels in the SYSTEM instructions (``opening=True``), not as a
        per-call instruction: this gateway takes system content only at
        ``messages[0]``, and a trailing system message is rejected outright.

        ``tool_choice="none"`` — an opening that called ``next_question`` would skip
        the very question it was about to ask.
        """
        if not self._setup.userdata.current_question_text:
            return
        await self.refresh_state_note(opening=True)
        self.session.generate_reply(tool_choice="none", allow_interruptions=False)

    async def on_user_turn_completed(self, turn_ctx: ChatContext, new_message: ChatMessage) -> None:
        """Refresh the server's state note, then let the LLM reply.

        Deliberately does NOT raise ``StopResponse``. The routed agent does,
        because it has no LLM and the default generation step would produce
        silence; here that step is the entire point, and raising would leave the
        candidate listening to nothing after every answer.

        SPOKEN turns only. The SDK reaches this hook from ``on_end_of_turn``,
        whose sole caller is the STT path (``audio_recognition.py``), so a TYPED
        turn never arrives here — :mod:`native_text_input` calls
        :meth:`fold_turn` for those. Any work added here that must apply to every
        turn belongs in ``fold_turn``, not in this method.
        """
        await self.fold_turn(answer_text=(new_message.text_content or ""))

    async def fold_turn(self, *, answer_text: str) -> None:
        """Grade one candidate answer, then refresh the state note.

        The single graded path, shared by the spoken and typed doors. Takes no
        chat context: the note lives in the SYSTEM instructions now, so neither
        door has to mutate a per-turn copy and write it back.
        """
        userdata = self._setup.userdata
        # Belongs to the PREVIOUS turn: the model has had its chance to ask the
        # question the server handed it, so from here on the note must describe the
        # live question normally again.
        userdata.pending_new_question = False
        # Snapshot BEFORE the advance below: this turn's answer belongs to the
        # question it was folded against, and the transcript handler runs after
        # `current_question_id` has moved on.
        answered_question_id = (
            str(userdata.state.current_question_id)
            if userdata.state is not None and userdata.state.current_question_id
            else None
        )
        # Grade FIRST. The note is built from `outcome_coverage`, so building it
        # before folding this answer in would describe the previous turn's state
        # and the model would probe an outcome it has just satisfied.
        if self._setup.grade_turn is not None:
            try:
                await self._setup.grade_turn(
                    state=userdata.state,
                    answer_text=answer_text,
                    question_text=(userdata.current_question_text or ""),
                    turn_id=str(uuid4()),
                )
            except Exception:  # noqa: BLE001 -- grading must never cost the reply
                logger.exception(
                    "native turn grading failed (session=%s)", userdata.interview_session_id
                )
        # The count is a plain int on the userdata dataclass, so it does not
        # follow the pool on its own. Left stale, the note keeps promising
        # questions the bank no longer has and the model plans around them.
        userdata.questions_remaining = self._setup.selector.remaining()
        # The server, not the model, owns the transition — see `native_advance`.
        # Must run BEFORE the note is built: the note is the only thing that tells
        # the model which question it is now on.
        outcome = await advance_if_resolved(userdata, self._setup.selector)
        if not outcome.advanced:
            count_follow_up(userdata)
        # Publish after the fold so the answer's transcript row can be attributed
        # to the question it actually answered.
        userdata.answered_question_id = answered_question_id
        self._record_shadow(userdata, advanced=outcome.advanced)
        await self.refresh_state_note()

    def _record_shadow(self, userdata: InterviewUserdata, *, advanced: bool) -> None:
        """Run the audited policy beside the model and log both.

        The native path produces no ``ReasonCode`` of its own, so this is the only
        source of an auditable justification per turn — and the only measure of how
        far the conversational agent drifts from the policy that was signed off.

        Never allowed to fail the turn: an audit record is not worth an interview.
        """
        if userdata.state is None:
            return
        try:
            verdict = shadow_check_turn(
                state=userdata.state,
                intent=IntentClassification(
                    intent=StudentIntent.ANSWER, confidence=0.0, rationale="native-path shadow"
                ),
                model_advanced=advanced,
                questions_remaining=userdata.questions_remaining,
                time_fraction_remaining=None,
                required_outcome_ids=userdata.required_outcome_ids,
            )
            obs.emit(
                obs.EV_SHADOW,
                session_id=userdata.interview_session_id,
                **verdict.to_dict(),
            )
        except Exception:  # noqa: BLE001 -- audit must never cost a graded turn
            logger.exception("shadow check failed (session=%s)", userdata.interview_session_id)


async def run_native_interview(
    ctx: JobContext, *, settings: Settings, setup: NativeSetup
) -> HardStopTimer:
    """Start the native agent for one job. Returns the armed hard stop.

    The timer is returned so the caller (and the tests) can assert it was armed.
    It stays alive after this returns without the caller holding it: the
    ``finalize_session`` closure below captures it, ``userdata`` holds that
    closure, and the SDK holds ``userdata`` for the life of the session.
    """
    userdata = setup.userdata
    chat_ctx = ChatContext.empty()
    seed_onboarding_history(chat_ctx, setup.onboarding_turns)
    # The opening is GENERATED, so the join request must satisfy the gateway's
    # "must end with a user turn" rule — verified: an assistant OR a system
    # message last is a 400, and the candidate heard nothing at all.
    end_on_user_turn(chat_ctx)

    agent = NativeInterviewAgent(
        instructions=build_instructions(
            language=setup.language, interviewer_name=setup.interviewer_name
        ),
        chat_ctx=chat_ctx,
        setup=setup,
    )
    session = build_native_session(
        settings, userdata, language=setup.language, voice=setup.tts_voice
    )
    hard_stop = HardStopTimer(
        HardStopPlan(
            deadline_seconds=hard_stop_deadline_seconds(
                time_remaining_seconds=setup.time_remaining_seconds,
                questions_remaining=userdata.questions_remaining,
            ),
            close=setup.close_session,
            interview_session_id=userdata.interview_session_id,
        ),
        session=session,
    )
    # Built before `session.start` because the room options are an ARGUMENT to it.
    # Safe because the publisher resolves the room lazily on each publish.
    publisher = ControlPublisher(session, interview_session_id=userdata.interview_session_id)
    userdata.publish_state = _make_state_publisher(userdata, publisher, save=setup.save_state)
    hard_stop.on_finalized = _make_finish_marker(userdata)

    # Route the tool's finalizer through the same once-only guard, so the model
    # ending the interview and the timer firing cannot both submit the session.
    tool_finalize = userdata.finalize_session
    userdata.finalize_session = lambda: hard_stop.finalize_once(tool_finalize)

    room_input, room_output = room_options_for_mode(
        setup.input_mode, text_input_cb=make_text_input_cb(publisher)
    )
    await session.start(
        agent,
        room=ctx.room,
        room_input_options=room_input,
        room_output_options=room_output,
    )
    _record_conversation(session, userdata)
    hard_stop.start()
    obs.emit(
        obs.EV_AGENT_DISPATCH,
        session_id=userdata.interview_session_id,
        language=setup.language,
        native=True,
    )
    # Snapshot on join, so a client that reloaded or rejoined mid-interview learns
    # which question it is on without waiting for the next state change.
    await userdata.publish_state()

    # Re-read the current question when the candidate rejoins mid-interview.
    # `participant_connected` fires for the candidate's FIRST join too (the agent
    # subscribes to a room the candidate is already in), so only re-read after a
    # DISCONNECT has been observed — that is the definition of a rejoin.
    student_identity = f"student-{userdata.student_id}"
    student_disconnected = False

    def _on_disconnected(participant: rtc.RemoteParticipant) -> None:
        nonlocal student_disconnected
        if participant.identity == student_identity:
            student_disconnected = True

    def _on_connected(participant: rtc.RemoteParticipant) -> None:
        nonlocal student_disconnected
        if participant.identity != student_identity:
            return
        if not student_disconnected:
            return
        student_disconnected = False
        question = userdata.current_question_text
        if not question or userdata.finished:
            return
        logger.info(
            "re-reading current question after rejoin (session=%s)",
            userdata.interview_session_id,
        )
        asyncio.create_task(  # noqa: RUF006 - fire-and-forget best-effort
            _re_read_question(session, question, userdata.language)
        )

    ctx.room.on("participant_connected", _on_connected)
    ctx.room.on("participant_disconnected", _on_disconnected)

    return hard_stop


def _make_finish_marker(userdata: InterviewUserdata) -> Callable[[], Awaitable[None]]:
    async def _mark() -> None:
        userdata.finished = True
        await userdata.publish_state()

    return _mark


def _make_state_publisher(
    userdata: InterviewUserdata,
    publisher: ControlPublisher,
    *,
    save: Callable[[], Awaitable[None]] | None,
) -> Callable[[], Awaitable[None]]:
    """Persist state, then tell the client. Never raises.

    Persist BEFORE publishing: a client told the question advanced, on state that
    was never written, would re-render the old question after a rejoin and look
    like it lost the candidate's progress.
    """

    async def _publish() -> None:
        if save is not None:
            try:
                await save()
            except Exception:  # noqa: BLE001 -- a failed save must not cost the notification
                logger.exception(
                    "runtime state save failed (session=%s)", userdata.interview_session_id
                )
        await publisher.snapshot(build_snapshot(userdata))

    return _publish


__all__ = [
    "HardStopPlan",
    "HardStopTimer",
    "NativeInterviewAgent",
    "hard_stop_deadline_seconds",
    "run_native_interview",
]


def _record_conversation(session: AgentSession, userdata: InterviewUserdata) -> None:
    """Persist every committed chat item to ``interview_session_messages``.

    The evaluation and the gap report read that table, so without this a native
    interview is graded against onboarding and a goodbye. Subscribed AFTER
    ``session.start`` so no item can arrive before the handler exists.

    The question id is read at emit time rather than captured: the tools move
    ``current_question_id`` mid-session, and a captured value would file every
    answer under question one. User items are the one deliberate exception — see
    ``answered_question_id``.
    """

    def _on_item(event: object) -> None:
        item = getattr(event, "item", None)
        role = getattr(item, "role", None)
        if role is None:
            return
        text = getattr(item, "text_content", None) or ""
        state = userdata.state
        # A user item is the answer to the question it was FOLDED against. The
        # live `current_question_id` has already advanced past it by the time this
        # handler runs, so reading it here filed every answer one question ahead —
        # Q1's answer under Q2, and so on — which made the gap report pair answers
        # with the wrong prompts and count them as unanswered. An assistant item is
        # the agent speaking, which tracks the LIVE question (or the one the server
        # just moved it to).
        # The agent's closing line is spoken AFTER the session is submitted, so
        # `current_question_id` still points at the last question — without this
        # guard the goodbye would be filed as a "question" under it, and the
        # transcript showed the closing twice (once from the ceremony, once from
        # this mislabelled copy).
        is_closing = bool(userdata.finished)
        raw_question_id = (
            None if is_closing else (userdata.answered_question_id if str(role) == "user" else None)
        )
        if raw_question_id is None:
            raw_question_id = getattr(state, "current_question_id", None) if state else None
        question_id: UUID | None = None
        if raw_question_id:
            try:
                question_id = UUID(str(raw_question_id))
            except (ValueError, AttributeError, TypeError):
                question_id = None
        asyncio.create_task(  # noqa: RUF006 - fire-and-forget; record_turn never raises
            record_turn(
                userdata.interview_session_id,
                role=str(role),
                text=text,
                session_question_id=question_id,
                kind="closing" if is_closing else ("answer" if str(role) == "user" else "question"),
            )
        )

    session.on("conversation_item_added", _on_item)
