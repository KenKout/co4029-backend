"""LiveKit agent worker entrypoint for the voice interview (Phase 3).

Run as a standalone process (see ``ecosystem.config.cjs`` pm2 entry):

    python -m abridgeai.features.interviews.realtime.agent start

The worker registers under ``settings.livekit_agent_name`` and is dispatched
per-session by the join token's room-config (Phase 2). The dispatch metadata
carries ``{session_id, student_id}`` — parsed here, then handed to the
:class:`InterviewAgent` runtime.
"""

from __future__ import annotations

import asyncio
import json
import logging
from uuid import UUID

from livekit import (
    rtc,  # type: ignore[attr-defined]  # rtc is a lazy submodule; livekit ships no stubs
)
from livekit.agents import JobContext, RoomOutputOptions, WorkerOptions, cli
from livekit.agents.voice.background_audio import (
    AudioConfig,
    BackgroundAudioPlayer,
    BuiltinAudioClip,
)

from abridgeai.core.config import get_settings
from abridgeai.features.interviews.realtime import observability as obs
from abridgeai.features.interviews.realtime import orchestration_bridge as bridge
from abridgeai.features.interviews.realtime.session_runtime import (
    InterviewAgent,
    build_agent_session,
)
from abridgeai.features.interviews.services.real_time import (
    META_LANGUAGE,
    META_SESSION_ID,
    META_STUDENT_ID,
)

logger = logging.getLogger(__name__)


async def entrypoint(ctx: JobContext) -> None:
    """Per-session job: parse dispatch metadata, connect, run the interview."""
    settings = get_settings()

    raw_metadata = ctx.job.metadata or ""
    if not raw_metadata:
        logger.error("interview agent dispatched without metadata; aborting job")
        return
    meta = json.loads(raw_metadata)
    interview_session_id = UUID(meta[META_SESSION_ID])
    student_id = UUID(meta[META_STUDENT_ID])
    # Phase 18: language for adaptive utterances (default 'en' for older tokens
    # minted before this field existed — backward compatible).
    language = meta.get(META_LANGUAGE, "en")
    logger.info("interview agent starting (session=%s)", interview_session_id)
    obs.emit(
        obs.EV_AGENT_DISPATCH,
        session_id=interview_session_id,
        language=language,
    )

    try:
        await ctx.connect()
    except Exception as exc:
        obs.emit(
            obs.EV_ROOM_JOIN,
            session_id=interview_session_id,
            ok=False,
            error_class=type(exc).__name__,
        )
        raise
    obs.emit(obs.EV_ROOM_JOIN, session_id=interview_session_id, ok=True)

    # Record student disconnects for post-session review. Non-terminal: the
    # session stays in_progress so the student can rejoin; the time-limit
    # sweep (Phase 4) is the backstop for sessions that never resume.
    def _on_participant_disconnected(participant: rtc.RemoteParticipant) -> None:
        if participant.identity == f"student-{student_id}":
            obs.emit(
                obs.EV_DISCONNECT,
                session_id=interview_session_id,
                reason="participant_disconnected",
            )
            asyncio.create_task(  # noqa: RUF006 - fire-and-forget best-effort
                bridge.record_integrity_event(
                    interview_session_id, student_id, "disconnect", severity="warning"
                )
            )

    ctx.room.on("participant_disconnected", _on_participant_disconnected)

    first_question_text = await bridge.get_current_question_text(
        interview_session_id, language=language
    )
    # get_opening_text only yields a greeting while onboarding is still pending,
    # which a realtime token forbids — so in practice it is always None and the
    # room fell straight into question one. The room intro fills that beat for
    # configs with a named interviewer; both are None otherwise, preserving the
    # existing behaviour exactly.
    opening_text = await bridge.get_opening_text(
        interview_session_id, language=language
    ) or await bridge.get_room_intro_text(interview_session_id, language=language)
    tts_voice = await bridge.get_tts_voice(interview_session_id)
    agent = InterviewAgent(
        interview_session_id=interview_session_id,
        student_id=student_id,
        first_question_text=first_question_text,
        opening_text=opening_text,
        language=language,
    )
    voice_persona, voice_verbosity = await bridge.get_voice_persona(interview_session_id)
    session = build_agent_session(
        settings,
        language=language,
        voice=tts_voice,
        persona=voice_persona,
        verbosity=voice_verbosity,
    )
    # Make transcript↔audio sync EXPLICIT (do not rely on SDK defaults). With
    # transcription_enabled + sync_transcription True, livekit-agents attaches a
    # TranscriptSynchronizer that paces the published transcript to the ACTUAL
    # TTS audio playout (RMS-based speaking-rate detection) — so the on-screen
    # text advances at the same speed as the spoken voice. transcription_speed_factor
    # 1.0 keeps text exactly in step with audio (no lead/lag).
    await session.start(
        agent,
        room=ctx.room,
        room_output_options=RoomOutputOptions(
            transcription_enabled=True,
            sync_transcription=True,
            transcription_speed_factor=1.0,
        ),
    )

    # Quiet room tone under the interview. Perfect silence between turns reads
    # as a dropped call, and candidates report AI interviews feeling like
    # "structured software" rather than a conversation; a low room floor is the
    # cheapest cue that someone is on the other end.
    #
    # Ambient only. The SDK also offers a `thinking_sound` (keyboard typing —
    # the interviewer taking notes) but it fires on the agent entering the
    # "thinking" state, which only the SDK's own LLM reply pipeline sets. This
    # agent deliberately has no LLM plugin and raises StopResponse, so that
    # state is never entered and the sound would never play. Driving it would
    # mean calling a private setter; the spoken filler in session_runtime covers
    # the same gap through a supported path.
    #
    # Volume is low on purpose: for an anxious candidate, added noise is a
    # stressor, not atmosphere.
    background = BackgroundAudioPlayer(
        ambient_sound=AudioConfig(BuiltinAudioClip.OFFICE_AMBIENCE, volume=0.3),
    )
    try:
        await background.start(room=ctx.room, agent_session=session)
    except Exception:  # noqa: BLE001 -- ambience is cosmetic; never fail a session for it
        logger.warning("background ambience failed to start (session=%s)", interview_session_id)


def run() -> None:
    """CLI launcher. Credentials come from Settings (LiveKit Cloud)."""
    settings = get_settings()
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name=settings.livekit_agent_name,
            ws_url=settings.livekit_ws_url or "",
            api_key=(
                settings.livekit_api_key.get_secret_value() if settings.livekit_api_key else ""
            ),
            api_secret=(
                settings.livekit_api_secret.get_secret_value()
                if settings.livekit_api_secret
                else ""
            ),
        )
    )


if __name__ == "__main__":
    run()
