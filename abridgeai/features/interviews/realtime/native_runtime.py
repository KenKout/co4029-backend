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
from functools import partial
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from livekit import (
    rtc,  # rtc is a lazy submodule; livekit ships no stubs
)
from livekit.agents import Agent, AgentSession, ChatContext, ChatMessage

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

# Re-exported below: the finish machinery moved to its own module for the LOC gate,
# but it is part of THIS module's published surface and its callers (and tests)
# reach it as `native_runtime.HardStopTimer`.
from abridgeai.features.interviews.realtime.native_hard_stop import (
    HardStopPlan,
    HardStopTimer,
    TranscriptWriteBarrier,
    hard_stop_deadline_seconds,
)
from abridgeai.features.interviews.realtime.native_rejoin import (
    _re_read_question,  # noqa: F401  -- re-exported: room-rejoin path calls it here
    _rejoin_question_text,  # noqa: F401  -- re-exported: room-rejoin path calls it here
)
from abridgeai.features.interviews.realtime.native_text_input import make_text_input_cb
from abridgeai.features.interviews.realtime.native_transcript import record_turn
from abridgeai.features.interviews.realtime.native_turn_intake import TurnIntake

if TYPE_CHECKING:
    from livekit.agents import JobContext

    from abridgeai.core.config import Settings
    from abridgeai.features.interviews.realtime.agent_userdata import InterviewUserdata
    from abridgeai.features.interviews.realtime.native_bridge import NativeSetup

logger = logging.getLogger(__name__)

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

    async def fold_turn(self, *, answer_text: str, turn_key: str | None = None) -> None:
        """Grade one candidate answer, then refresh the state note.

        The single graded path, shared by the spoken and typed doors. Takes no
        chat context: the note lives in the SYSTEM instructions now, so neither
        door has to mutate a per-turn copy and write it back.

        ``turn_key`` is the TYPED door's client idempotency key, persisted with the
        runtime state as ``last_turn_idempotency_key``. It is what lets a resend
        that lands on a DIFFERENT agent process — the in-memory ledger is empty
        there — still be recognised as the same turn. A spoken turn has no key: the
        SDK's end-of-turn commits are not client retries.
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
                    turn_key=turn_key,
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
            # Persist the charge. `grade_turn` already saved — but it saved BEFORE
            # this increment, and the advance branch is the only one that writes
            # again (via `publish_state` inside `advance_if_resolved`). So a
            # non-advancing turn's follow-up was held in memory only: a worker
            # restart resumed on the previous count and handed the candidate's
            # spent probe back, which is what let a question be probed past its
            # budget and never auto-advance.
            await self._persist_turn_counters()
        # Publish after the fold so the answer's transcript row can be attributed
        # to the question it actually answered.
        userdata.answered_question_id = answered_question_id
        self._record_shadow(userdata, advanced=outcome.advanced)
        await self.refresh_state_note()

    async def _persist_turn_counters(self) -> None:
        """Write the counters this turn charged, without publishing a snapshot.

        A plain save rather than ``publish_state``: nothing the client renders
        changed on a non-advancing turn (same question, same "n of m"), and
        re-publishing an identical snapshot on every probe is noise the client has
        to diff. The BUDGETS did change, and those live only in runtime state.

        Never raises: the charge is already applied in memory, and losing the write
        costs a re-drive's accuracy, not the candidate's reply.
        """
        save = self._setup.save_state
        if save is None:
            return
        try:
            await save()
        except Exception:  # noqa: BLE001 -- a failed save must not cost the reply
            logger.exception(
                "persisting the turn counters failed (session=%s)",
                self._setup.userdata.interview_session_id,
            )

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
    # One per session: the turn_key ledger that makes a resend idempotent, and the
    # in-flight count a finish waits on. Seeded from the key persisted by any
    # earlier run of this session, so a restart does not forget what it graded.
    turn_intake = TurnIntake()
    turn_intake.seed(setup.last_turn_key)
    hard_stop = HardStopTimer(
        HardStopPlan(
            deadline_seconds=hard_stop_deadline_seconds(
                time_remaining_seconds=setup.time_remaining_seconds,
                questions_remaining=userdata.questions_remaining,
            ),
            close=setup.close_session,
            interview_session_id=userdata.interview_session_id,
            flush_transcript=transcript_writes.flush,
            drain_turns=turn_intake.drain,
            close_fallback=setup.close_session_fallback,
        ),
        session=session,
    )
    # Built before `session.start` because the room options are an ARGUMENT to it.
    # Safe because the publisher resolves the room lazily on each publish.
    publisher = ControlPublisher(session, interview_session_id=userdata.interview_session_id)
    userdata.publish_state = _make_state_publisher(userdata, publisher, save=setup.save_state)
    # The tools reach persistence through this for the state a snapshot does not
    # carry (the hint ladder, the follow-up budgets). Without it a hint turn — which
    # is not on the graded path — mutated the ladder in memory only.
    if setup.save_state is not None:
        userdata.save_state = setup.save_state
    userdata.publish_agent_action = partial(publisher.agent_action)
    hard_stop.on_finalized = _make_finish_marker(userdata)

    # Route the tool's finalizer through the same once-only guard, so the model
    # ending the interview and the timer firing cannot both submit the session.
    tool_finalize = userdata.finalize_session
    userdata.finalize_session = lambda: hard_stop.finalize_once(tool_finalize)

    room_input, room_output = room_options_for_mode(
        setup.input_mode,
        text_input_cb=make_text_input_cb(publisher, intake=turn_intake),
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
