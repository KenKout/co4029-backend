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
from collections.abc import Awaitable, Callable, Coroutine
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from livekit import (
    rtc,  # rtc is a lazy submodule; livekit ships no stubs
)
from livekit.agents import Agent, AgentSession, ChatContext, ChatMessage, get_job_context

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
from abridgeai.features.interviews.realtime.native_rejoin import (
    _re_read_question,  # noqa: F401  -- re-exported: room-rejoin path calls it here
    _rejoin_question_text,  # noqa: F401  -- re-exported: room-rejoin path calls it here
)
from abridgeai.features.interviews.realtime.native_text_input import make_text_input_cb
from abridgeai.features.interviews.realtime.native_transcript import record_turn

if TYPE_CHECKING:
    from livekit.agents import JobContext

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
# Used only when neither a time limit nor a question budget is known. Long
# enough that no real interview trips it, short enough that an abandoned room
# still closes.
_DEFAULT_HARD_STOP_SECONDS = 45.0 * 60.0
# Bound on waiting for the closing to finish playing out before teardown, so a
# stuck SpeechHandle cannot hang the job. Mirrors ``session_runtime``.
_CLOSING_PLAYOUT_TIMEOUT_S = 30.0
_TRANSCRIPT_FLUSH_TIMEOUT_S = 3.0


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


async def _no_flush() -> None:
    return None


class TranscriptWriteBarrier:
    """Track native transcript writes until session finalization.

    Conversation callbacks cannot await ``record_turn`` directly. Before either
    native finalization path submits and enqueues evaluation, this barrier waits
    for every write task already scheduled, so the evaluator sees the final
    candidate answer.
    """

    def __init__(self) -> None:
        self._pending: set[asyncio.Task[None]] = set()

    def create(self, coro: Coroutine[Any, Any, None]) -> None:
        task = asyncio.create_task(coro)
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)

    async def flush(self) -> None:
        """Wait briefly for writes in flight without cancelling them.

        Loop with a single deadline so a write task created WHILE the flush is
        already running (an SDK callback landing exactly as finalization
        begins) is also drained before submit. Writers still pending at the
        deadline are left to finish on their own — never cancelled — so a
        wedged DB session cannot hang the finalizer, and a merely slow write
        is not lost, only unawaited.
        """
        deadline = asyncio.get_running_loop().time() + _TRANSCRIPT_FLUSH_TIMEOUT_S
        while self._pending:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                logger.error(
                    "transcript flush timed out (pending=%d, timeout_seconds=%s)",
                    len(self._pending),
                    _TRANSCRIPT_FLUSH_TIMEOUT_S,
                )
                return
            done, _still = await asyncio.wait(
                tuple(self._pending),
                timeout=remaining,
            )
            for task in done:
                try:
                    task.result()
                except Exception:  # noqa: BLE001 -- record_turn normally swallows failures
                    logger.exception("unexpected transcript task failure")


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
    flush_transcript: Callable[[], Awaitable[None]] = _no_flush


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
        await self._plan.flush_transcript()
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
            await self._plan.flush_transcript()
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


# Markers identifying the server's injected control notes in the chat context.
# Everything the server pins mid-conversation (the advance directive, the
# end-refusal checklist) carries one, so a new note can REPLACE the stale ones
# instead of stacking contradictory instructions for the model.
_CONTROL_NOTE_MARKERS: tuple[str, ...] = (
    "[interview control — not spoken by the candidate]",
    "[interview checklist — not spoken by the candidate]",
)


def _control_notes_removed(chat_ctx: ChatContext) -> list[ChatMessage]:
    """The context's items minus every injected control note (read-only view).

    ``getattr`` on every field: the context carries more than messages —
    ``update_instructions`` appends an ``AgentConfigUpdate`` item, which has no
    ``text_content`` and must pass through untouched. A direct attribute read
    raised ``AttributeError`` there and killed EVERY directive injection after
    the first ``update_instructions`` (session 564334c0: the advance selected
    Q3, the directive never landed, and the interviewer stopped responding).
    """
    kept: list[ChatMessage] = []
    for item in chat_ctx.items:
        text = getattr(item, "text_content", None)
        if isinstance(text, str) and any(
            text.startswith(marker) for marker in _CONTROL_NOTE_MARKERS
        ):
            continue
        kept.append(item)
    return kept


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
        if outcome.advanced:
            await self._inject_advance_directive()
        else:
            count_follow_up(userdata)
        # Publish after the fold so the answer's transcript row can be attributed
        # to the question it actually answered.
        userdata.answered_question_id = answered_question_id
        self._record_shadow(userdata, advanced=outcome.advanced)
        await self.refresh_state_note()

    async def _inject_advance_directive(self) -> None:
        """Pin "ask the NEW question now" into the chat context on an advance.

        The state note already says the server has moved — and the model kept
        debating the OLD question anyway (production session 1d629118: the card
        advanced to Q2 at 1:27 while the interviewer probed Q1 for three more
        exchanges and Q2 was never asked). System instructions are only merged at
        the head, so a user-role message is the one mid-conversation channel this
        gateway honours — the same pattern the end-interview refusal note uses.

        REPLACES the previous control notes rather than appending: directives
        accumulate one per advance, and after two advances the context held
        "ask Q2, do not ask anything else" next to "ask Q3, do not ask anything
        else" — contradictory instructions the model resolved by calling
        end_interview instead (session e972fade: the bank ran dry mid-race and
        the gate, seeing remaining=0, allowed it — Q3 was never asked).

        Never raises: the advance already happened; a directive that fails to
        land must not cost the model its reply.
        """
        question = self._setup.userdata.current_question_text
        if not question:
            return
        try:
            note = ChatMessage(
                role="user",
                content=[
                    "[interview control — not spoken by the candidate] The "
                    "previous question is finished and the server has ALREADY "
                    f'moved the interview to a NEW question: "{question}". In '
                    "your next reply: acknowledge the candidate's last answer "
                    "in ONE short sentence, then ask this question in your own "
                    "words. Do NOT ask anything else and do NOT continue the "
                    "previous question."
                ],
            )
            await self.update_chat_ctx(
                ChatContext(items=[*_control_notes_removed(self.chat_ctx), note])
            )
        except Exception:  # noqa: BLE001 -- reinforcement, never the gate
            logger.warning(
                "advance directive could not be injected (session=%s)",
                self._setup.userdata.interview_session_id,
                exc_info=True,
            )

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
    transcript_writes = TranscriptWriteBarrier()
    hard_stop = HardStopTimer(
        HardStopPlan(
            deadline_seconds=hard_stop_deadline_seconds(
                time_remaining_seconds=setup.time_remaining_seconds,
                questions_remaining=userdata.questions_remaining,
            ),
            close=setup.close_session,
            interview_session_id=userdata.interview_session_id,
            flush_transcript=transcript_writes.flush,
        ),
        session=session,
    )
    # Built before `session.start` because the room options are an ARGUMENT to it.
    # Safe because the publisher resolves the room lazily on each publish.
    publisher = ControlPublisher(session, interview_session_id=userdata.interview_session_id)
    userdata.publish_state = _make_state_publisher(userdata, publisher, save=setup.save_state)
    userdata.publish_agent_action = partial(publisher.agent_action)
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
    _record_conversation(session, userdata, transcript_writes)
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
        if userdata.finished:
            return
        # A reloaded client resubscribes to the control topic from scratch, so
        # its first render is the REST history — which may already lag the
        # server's latest advance. A fresh snapshot re-syncs the card, the
        # "n of m" counter and the deadline before the re-read finishes.
        asyncio.create_task(  # noqa: RUF006 - fire-and-forget; publish_state never raises
            userdata.publish_state()
        )
        if not question:
            return
        logger.info(
            "re-reading current question after rejoin (session=%s)",
            userdata.interview_session_id,
        )

        async def _announce_re_read(text: str) -> None:
            await userdata.publish_agent_action(kind="repeat", text=text)  # type: ignore[call-arg]

        asyncio.create_task(  # noqa: RUF006 - fire-and-forget best-effort
            _re_read_question(session, question, userdata.language, _announce_re_read)
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
    "TranscriptWriteBarrier",
    "hard_stop_deadline_seconds",
    "run_native_interview",
]


def _record_conversation(
    session: AgentSession,
    userdata: InterviewUserdata,
    transcript_writes: TranscriptWriteBarrier,
) -> None:
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
        # An assistant item consumes the pending assistance kind set when the
        # server granted a hint / clarification: this utterance IS that
        # assistance. Everything else keeps the ordinary "question" kind, and
        # the user turn before it stays an "answer" — the pending kind never
        # applies to the candidate's own words.
        if str(role) == "assistant":
            kind = userdata.pending_assistant_kind or "question"
            userdata.pending_assistant_kind = None
        else:
            kind = "answer"
        transcript_writes.create(
            record_turn(
                userdata.interview_session_id,
                role=str(role),
                text=text,
                session_question_id=question_id,
                kind="closing" if is_closing else kind,
            )
        )

    session.on("conversation_item_added", _on_item)
