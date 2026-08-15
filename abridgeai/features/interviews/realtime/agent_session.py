"""Native (multiturn) AgentSession construction for the interview.

The difference from ``session_runtime.build_agent_session`` is one argument and it
is the whole point: this session is built WITH an ``llm``, so ``AgentSession``
holds a single ``chat_ctx`` for the interview and the agent reads the entire
conversation. The routed architecture had no LLM in the session, which forced
every turn out to stateless per-stage calls that could not see each other — the
cause of the interviewer repeating itself and asking the candidate to diagnose
their own confusion.

Kept in a separate module from ``session_runtime`` so the two constructions can
exist side by side during the migration, and because that file is already at its
size budget.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from livekit.agents import (
    NOT_GIVEN,
    AgentSession,
    RoomInputOptions,
    RoomOutputOptions,
    TurnHandlingOptions,
    inference,
)
from livekit.agents.voice.room_io.types import TextInputCallback
from livekit.plugins import deepgram, openai, silero

from abridgeai.features.interviews.orchestrator import tools as gate
from abridgeai.features.interviews.realtime.agent_userdata import InterviewUserdata
from abridgeai.features.interviews.realtime.session_runtime import turn_handling_options
from abridgeai.features.interviews.services import narration

if TYPE_CHECKING:
    from abridgeai.core.config import Settings


def build_state_reminder(data: InterviewUserdata, *, opening: bool = False) -> str:
    """The state note folded into the agent's SYSTEM instructions each turn.

    It used to be appended to ``chat_ctx`` as a mid-conversation system message.
    Gemini — the model behind this gateway — effectively ignores those: probed
    with the same note at ``messages[0]`` the model asked the question it names,
    and with the note mid-conversation it produced a generic greeting instead. So
    the note that pins the live question, the budgets and the permitted next move
    was being silently discarded, which is the likeliest reason the model kept
    announcing its own questions rather than calling ``next_question``.

    Returns "" when runtime state has not been loaded: a session that cannot
    compute the note must say nothing rather than assert something plausible, or
    the agent will confidently act on a fabricated budget.
    """
    if data.state is None:
        return ""
    return gate.build_turn_reminder(
        data.state,
        current_outcome_id=data.state.current_outcome_id,
        required_outcome_ids=data.required_outcome_ids,
        questions_remaining=data.questions_remaining,
        max_follow_ups_per_question=data.max_follow_ups_per_question,
        max_hints=data.max_hints_per_question,
        below_closing_threshold=data.below_closing_threshold,
        outcome_titles=data.outcome_titles,
        time_remaining_seconds=data.time_remaining_seconds,
        current_question_text=data.current_question_text,
        server_advanced=data.pending_new_question,
        opening=opening,
    )


def room_options_for_mode(
    input_mode: str,
    *,
    text_input_cb: TextInputCallback | None = None,
) -> tuple[RoomInputOptions, RoomOutputOptions]:
    """Room I/O for a voice, text or hybrid session.

    Text mode disables audio at the ROOM boundary rather than muting it later:
    with ``audio_enabled=False`` the candidate's microphone is never captured and
    no speech is synthesized, so a typing candidate cannot be recorded by
    accident. ``transcription_enabled`` stays on in every mode — it is how the
    agent's words reach the screen.

    ``text_input_cb`` MUST be supplied for any mode that accepts typed turns.
    Left unset the SDK default runs, which drops the ``turn_action`` /
    ``turn_key`` stream attributes and never reaches the graded path — see
    :mod:`native_text_input`. It is optional here only so the pure room-shape
    logic stays testable without a session.
    """
    audio = input_mode != "text"
    text_input = _sdk_default_text_input_cb() if text_input_cb is None else text_input_cb
    return (
        RoomInputOptions(
            text_enabled=True,
            text_input_cb=text_input,
            audio_enabled=audio,
            video_enabled=False,
            # A refresh disconnects the student for a few seconds. The SDK
            # default tears the AgentSession down on that, but the JOB stays in
            # the room: the rejoin then lands on a dead session whose text
            # callback is detached ("no callback attached") and whose say()
            # raises — every typed turn after the reload hangs forever, and no
            # new dispatch happens because an agent is still in the room.
            # Keeping the session open makes the rejoin resume the SAME job:
            # state, note, re-read and lk.chat all alive. The hard-stop timer
            # still bounds the job if the student never returns.
            close_on_disconnect=False,
        ),
        RoomOutputOptions(transcription_enabled=True, audio_enabled=audio),
    )


def _sdk_default_text_input_cb() -> TextInputCallback:
    """The SDK's own default, named explicitly rather than left implicit.

    Only reached when no callback is supplied, which in production is never — see
    the warning on :func:`room_options_for_mode`.
    """
    return RoomInputOptions().text_input_cb


def _native_turn_handling(language: str) -> TurnHandlingOptions:
    """Turn-taking for the native session, language-aware.

    English (and any locale the detector serves): the semantic turn detector.
    VAD endpointing cannot tell a finished answer from a mid-thought pause, so
    a candidate who thinks before speaking gets cut off — or holds the floor
    through dead air while the agent waits out the max delay. The detector
    classifies the transcript tail (end / unlikely end / backchannel) and hands
    over only on a real finish. Cloud credentials are already in the worker
    env, so the SDK resolves to the hosted model; without them it falls back to
    the local v1-mini in-process.

    Vietnamese: the detector's thresholds are English-tuned, and a wrong
    "unlikely end" verdict would truncate a Vietnamese answer mid-sentence —
    worse than the VAD default it replaced. Vietnamese sessions keep VAD
    endpointing with the same dynamic bounds the routed path uses.

    Endpointing bounds are shared with the routed path either way so the two
    runtimes cannot drift on pacing while the turn detector rolls out.
    """
    options = turn_handling_options()
    if not language.startswith("vi"):
        options["turn_detection"] = inference.TurnDetector()
    return options


def build_native_session(
    settings: Settings,
    userdata: InterviewUserdata,
    *,
    language: str = "en",
    voice: str | None = None,
) -> AgentSession[InterviewUserdata]:
    """Build the STT → LLM → TTS pipeline for a conversational interview.

    The ``llm`` is what makes this multiturn. Do not remove it to "simplify": the
    routed path would silently take over and the agent would stop being able to
    read its own conversation.
    """
    api_key = settings.llm_api_key or NOT_GIVEN
    base_url = settings.llm_base_url or NOT_GIVEN
    llm = openai.LLM(
        model=settings.llm_model_small,
        api_key=api_key,
        base_url=base_url,
    )

    if language.startswith("vi"):
        # Deepgram serves no Vietnamese locale; the OpenAI-compatible gateway does
        # both STT and TTS for it. Same asymmetry the routed path already handles.
        return AgentSession[InterviewUserdata](
            userdata=userdata,
            stt=openai.STT(model=settings.whisper_model, api_key=api_key, base_url=base_url),
            llm=llm,
            tts=openai.TTS(
                api_key=api_key,
                base_url=base_url,
                voice=narration.voice_for_persona(None),
            ),
            vad=silero.VAD.load(),
            turn_handling=_native_turn_handling(language),
        )

    dg_key = settings.deepgram_api_key.get_secret_value() if settings.deepgram_api_key else ""
    return AgentSession[InterviewUserdata](
        userdata=userdata,
        stt=deepgram.STT(
            model=settings.deepgram_stt_model,
            language="en-US",
            api_key=dg_key,
            base_url=f"{settings.deepgram_stt_base_url.rstrip('/')}/listen",
        ),
        llm=llm,
        tts=deepgram.TTS(
            model=narration.resolve_tts_voice(voice, settings=settings),
            api_key=dg_key,
            base_url=f"{settings.deepgram_tts_base_url.rstrip('/')}/speak",
        ),
        vad=silero.VAD.load(),
        turn_handling=_native_turn_handling(language),
    )


__all__ = ["build_native_session", "build_state_reminder", "room_options_for_mode"]
