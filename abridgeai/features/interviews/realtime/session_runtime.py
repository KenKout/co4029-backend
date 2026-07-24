"""LiveKit AgentSession wiring for the voice interview.

ALL LiveKit Agents API usage is isolated in this module (per the plan's
version-isolation decision). Targets ``livekit-agents 1.5.x``: if that surface
shifts, this is the only file to touch.

Design: STT + TTS + VAD pipeline with **no LLM plugin**. The agent never
generates replies on its own — every student turn is routed through
:mod:`orchestration_bridge` (the existing text brain) and the returned text is
spoken via ``session.say``. ``StopResponse`` suppresses the default generation
step so the (absent) LLM never fires.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any
from uuid import UUID, uuid4

from livekit.agents import (
    NOT_GIVEN,
    Agent,
    AgentSession,
    ChatContext,
    ChatMessage,
    StopResponse,
    get_job_context,
)
from livekit.plugins import deepgram, openai, silero

from abridgeai.core.config import Settings
from abridgeai.features.interviews.realtime import observability as obs
from abridgeai.features.interviews.realtime import orchestration_bridge as bridge

logger = logging.getLogger(__name__)

# The agent is non-generative (no LLM); instructions are required by the Agent
# base class but never sent to a model. Kept descriptive for clarity/audit.
_INSTRUCTIONS = (
    "You are an AI mock interviewer. You ask the provided interview questions "
    "and listen to the candidate's spoken answers. You never invent questions."
)
# Hard ceiling on how long we wait for the closing utterance to finish playing
# out before shutting the room down. Playback-aware shutdown (below) must not
# hang the job forever if a SpeechHandle never completes (e.g. transport
# hiccup); this bounds the wait, then shuts down regardless.
_CLOSING_PLAYOUT_TIMEOUT_S = 30.0


class InterviewAgent(Agent):
    """Speaks bank questions / follow-ups; routes answers to the text brain."""

    def __init__(
        self,
        *,
        interview_session_id: UUID,
        student_id: UUID,
        first_question_text: str | None,
        opening_text: str | None = None,
        language: str = "en",
    ) -> None:
        super().__init__(instructions=_INSTRUCTIONS)
        self._interview_session_id = interview_session_id
        self._student_id = student_id
        self._first_question_text = first_question_text
        self._opening_text = opening_text
        # Language for adaptive interviewer utterances ("vi"/"en"). Voice parity
        # with the REST path, which reads Accept-Language. Default "en".
        self._language = language

    async def on_enter(self) -> None:
        """Speak the greeting completely, then begin with question one."""
        if self._opening_text:
            opening_handle = self.session.say(self._opening_text, allow_interruptions=False)
            await opening_handle
            if hasattr(opening_handle, "wait_for_playout"):
                try:
                    await asyncio.wait_for(
                        opening_handle.wait_for_playout(),
                        timeout=_CLOSING_PLAYOUT_TIMEOUT_S,
                    )
                except Exception:  # noqa: BLE001
                    logger.warning(
                        "opening playout wait failed; continuing (session=%s)",
                        self._interview_session_id,
                    )
        if self._first_question_text:
            await self.session.say(self._first_question_text, allow_interruptions=False)

    async def on_user_turn_completed(self, turn_ctx: ChatContext, new_message: ChatMessage) -> None:
        """Route the transcribed answer to the text brain; speak the result.

        ``StopResponse`` is raised at the end so the pipeline does not attempt
        a default (LLM) reply — we have already said everything we intend to.
        """
        del turn_ctx
        transcript = (new_message.text_content or "").strip()
        if not transcript:
            raise StopResponse()

        # One id correlates the I/O events emitted here with the decision-level
        # events the bridge emits for this same turn.
        turn_id = uuid4().hex
        turn_started = obs.monotonic()
        # STT produced a final transcript (content NOT logged — length only).
        obs.emit(
            obs.EV_TURN_STARTED,
            session_id=self._interview_session_id,
            turn_id=turn_id,
            transcript_chars=len(transcript),
            language=self._language,
        )

        try:
            result = await bridge.handle_student_turn(
                self._interview_session_id,
                self._student_id,
                transcript,
                language=self._language,
                turn_id=turn_id,
            )
        except Exception as exc:
            logger.exception("interview turn failed (session=%s)", self._interview_session_id)
            obs.emit(
                obs.EV_TURN_ERROR,
                session_id=self._interview_session_id,
                turn_id=turn_id,
                error_class=type(exc).__name__,
                latency_ms=obs.latency_ms(turn_started),
            )
            raise StopResponse() from None

        # Latency from STT-final to the decision being ready to speak — the
        # honest "brain" turn latency (STT already done; excludes TTS playout).
        obs.emit(
            obs.EV_TURN_COMPLETED,
            session_id=self._interview_session_id,
            turn_id=turn_id,
            decision_latency_ms=obs.latency_ms(turn_started),
            will_speak=bool(result.speak_text),
            finished=result.is_finished,
        )

        # Track the LAST utterance's handle so, on a finished turn, we can wait
        # for the closing to finish PLAYING OUT before we shut the room down.
        last_handle = None

        if result.speak_text:
            obs.emit(
                obs.EV_TTS_STARTED,
                session_id=self._interview_session_id,
                turn_id=turn_id,
                speak_chars=len(result.speak_text),
            )
            tts_started = obs.monotonic()
            # On a finished turn ``speak_text`` IS the (adaptive) closing —
            # force it non-interruptible so it can't be cut short. On a normal
            # turn keep the session default (NOT_GIVEN) — unchanged behaviour.
            if result.is_finished:
                last_handle = self.session.say(result.speak_text, allow_interruptions=False)
            else:
                last_handle = self.session.say(result.speak_text)
            await last_handle
            obs.emit(
                obs.EV_TTS_COMPLETED,
                session_id=self._interview_session_id,
                turn_id=turn_id,
                tts_ms=obs.latency_ms(tts_started),
            )

        if result.is_finished:
            # Playback-aware shutdown: ``say()`` returning / awaiting only means
            # the turn was scheduled and generated — NOT that the audio finished
            # reaching the student. Wait for the closing's full playout (bounded
            # by a timeout so a stuck handle can't hang the job) BEFORE shutdown,
            # so the room is never torn down mid-closing.
            await self._await_closing_playout(last_handle, turn_id=turn_id)
            ctx = get_job_context(required=False)
            if ctx is not None:
                ctx.shutdown(reason="interview_complete")

        raise StopResponse()

    async def _await_closing_playout(
        self,
        handle: Any,  # noqa: ANN401 - livekit SpeechHandle | None; duck-typed
        *,
        turn_id: str,
    ) -> None:
        """Wait for the closing utterance to finish playing out, then emit a
        playout event. Bounded by ``_CLOSING_PLAYOUT_TIMEOUT_S`` and never
        raises — a stuck/again handle must not block shutdown or crash the job.
        """
        completed = False
        timed_out = False
        if handle is not None and hasattr(handle, "wait_for_playout"):
            playout_started = obs.monotonic()
            try:
                await asyncio.wait_for(
                    handle.wait_for_playout(),
                    timeout=_CLOSING_PLAYOUT_TIMEOUT_S,
                )
                completed = True
            except TimeoutError:
                timed_out = True
                logger.warning(
                    "closing playout exceeded %ss; shutting down anyway (session=%s)",
                    _CLOSING_PLAYOUT_TIMEOUT_S,
                    self._interview_session_id,
                )
            except Exception:  # noqa: BLE001 — never let playout wait crash shutdown
                logger.exception(
                    "closing playout wait failed (session=%s)",
                    self._interview_session_id,
                )
            playout_ms = obs.latency_ms(playout_started)
        else:
            playout_ms = None
        obs.emit(
            obs.EV_CLOSING_PLAYOUT,
            session_id=self._interview_session_id,
            turn_id=turn_id,
            completed=completed,
            timed_out=timed_out,
            playout_ms=playout_ms,
        )


def _is_english(language: str) -> bool:
    """True when the session language is English (the only Deepgram voice locale).

    Deepgram Aura TTS is English-only and Deepgram STT (nova-2/3) does not list
    Vietnamese, so VI sessions must fall back to the OpenAI-compatible gateway.
    This mirrors the REST narration policy in ``services/narration.py``.
    """
    return not language.lower().startswith("vi")


def build_agent_session(settings: Settings, *, language: str = "en") -> AgentSession[None]:
    """Construct the STT→(brain)→TTS pipeline, language-aware.

    Voice provider is chosen by session language:

    * **English** → Deepgram (STT ``nova-2``/configured model + Aura-2 TTS).
      Deepgram gives lower-latency streaming STT and natural Aura voices.
    * **Vietnamese** (or any non-English) → the OpenAI-compatible gateway,
      because Deepgram Aura TTS is English-only and Deepgram STT does not
      support Vietnamese. This matches the REST narration fallback policy.

    silero VAD handles turn detection for both (KISS — no turn-detector model
    until manual E2E shows it is needed).
    """
    if _is_english(language) and settings.deepgram_api_key is not None:
        dg_key = settings.deepgram_api_key.get_secret_value()
        # Deepgram plugin base URLs are the /listen and /speak endpoints; the
        # configured settings store the /v1 root, so append the operation path.
        stt_base = f"{settings.deepgram_stt_base_url.rstrip('/')}/listen"
        tts_base = f"{settings.deepgram_tts_base_url.rstrip('/')}/speak"
        return AgentSession[None](
            stt=deepgram.STT(
                model=settings.deepgram_stt_model,
                language="en-US",
                api_key=dg_key,
                base_url=stt_base,
            ),
            tts=deepgram.TTS(
                model=settings.deepgram_tts_model_en,
                api_key=dg_key,
                base_url=tts_base,
            ),
            vad=silero.VAD.load(),
        )

    # Non-English (Vietnamese) — Deepgram cannot serve this locale; use the
    # OpenAI-compatible gateway for both STT and TTS (same key/base_url).
    api_key = settings.llm_api_key or NOT_GIVEN
    base_url = settings.llm_base_url or NOT_GIVEN
    return AgentSession[None](
        stt=openai.STT(model=settings.whisper_model, api_key=api_key, base_url=base_url),
        tts=openai.TTS(api_key=api_key, base_url=base_url),
        vad=silero.VAD.load(),
    )


__all__ = ["InterviewAgent", "build_agent_session"]
