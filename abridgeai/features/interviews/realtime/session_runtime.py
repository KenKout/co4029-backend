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
    TurnHandlingOptions,
    get_job_context,
)
from livekit.plugins import deepgram, openai, silero

from abridgeai.core.config import Settings
from abridgeai.features.interviews.realtime import observability as obs
from abridgeai.features.interviews.realtime import orchestration_bridge as bridge
from abridgeai.features.interviews.services import narration

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

# How long the brain may think before the candidate hears an acknowledgement.
# Below roughly this, silence still reads as a normal conversational beat and a
# filler would only delay the real answer; above it, the pause starts to read as
# the system having stalled.
_THINKING_FILLER_DELAY_S = 0.8
_THINKING_FILLER_EN = "Mm-hm."
_THINKING_FILLER_VI = "Ừm."

# Endpointing — how long the candidate may pause before we call their turn over.
#
# The SDK default is a FIXED 0.5s, tuned for assistants where a request is one
# short sentence. An interview answer is not: candidates stop mid-thought to
# recall a term or plan the next clause, and at 0.5s the interviewer talks over
# them. Two independent studies of AI-run oral assessment name exactly this —
# pacing and question stacking — as the dominant complaint.
#
# ``dynamic`` mode keeps an exponential moving average of THIS speaker's pause
# lengths and moves the threshold inside [min, max], so a deliberate speaker is
# given room without making a fast one wait. The floor is raised above the
# default because the cost is asymmetric: half a second of extra silence is
# barely noticed, while being cut off mid-answer during a graded assessment is
# both stressful and unfair.
#
# That fairness point is not incidental. ASR endpointing is documented to cut
# off disfluent and stuttered speech before it finishes, so this same knob is
# the cheapest accessibility improvement available on a voice-first assessment.
_ENDPOINTING_MIN_DELAY_S = 0.7
_ENDPOINTING_MAX_DELAY_S = 4.0


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

    async def _speak_thinking_filler(self, turn_id: str) -> None:
        """Say one short acknowledgement if the decision is taking a while.

        Cancelled by the caller as soon as the brain returns, so a fast turn
        never hears it. The text is a standalone token rather than a disfluency
        spliced into the answer: filled pauses raise perceived naturalness, but
        disfluent text handed to a TTS engine measurably degrades its prosody,
        so the two must not be conflated.
        """
        try:
            await asyncio.sleep(_THINKING_FILLER_DELAY_S)
            text = _THINKING_FILLER_VI if self._language.startswith("vi") else _THINKING_FILLER_EN
            self.session.say(text, add_to_chat_ctx=False)
            obs.emit(
                obs.EV_THINKING_FILLER,
                session_id=self._interview_session_id,
                turn_id=turn_id,
                language=self._language,
            )
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 -- cosmetic; must never break a turn
            logger.debug("thinking filler failed (session=%s)", self._interview_session_id)

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

        # Cover the brain's thinking time with a short acknowledgement rather
        # than dead air. Scheduled, not spoken immediately: a fast turn should
        # not pay for a filler it does not need, so the task only fires if the
        # decision is still pending after _THINKING_FILLER_DELAY_S and is
        # cancelled otherwise. Kept OUT of the chat context — it is a social
        # signal, not interview content, and must not reach the transcript the
        # evaluator reads.
        filler_task = asyncio.create_task(self._speak_thinking_filler(turn_id))
        try:
            result = await bridge.handle_student_turn(
                self._interview_session_id,
                self._student_id,
                transcript,
                language=self._language,
                turn_id=turn_id,
            )
        except Exception as exc:
            filler_task.cancel()
            logger.exception("interview turn failed (session=%s)", self._interview_session_id)
            obs.emit(
                obs.EV_TURN_ERROR,
                session_id=self._interview_session_id,
                turn_id=turn_id,
                error_class=type(exc).__name__,
                latency_ms=obs.latency_ms(turn_started),
            )
            raise StopResponse() from None

        filler_task.cancel()

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


def speech_rate_from_verbosity(verbosity: int) -> float:
    """Speaking-rate multiplier for a 0-4 verbosity dial.

    Mirrors the frontend's ``wordsPerMinuteFromTraits`` (124 + 13·verbosity WPM)
    expressed as a ratio against the mid dial, so the browser-narration voice
    and the LiveKit voice pace the same persona alike. Clamped because the
    provider treats speed as a hard multiplier and an extreme value is worse
    than a flat one.
    """
    clamped = max(0, min(4, int(verbosity)))
    return round((124 + clamped * 13) / (124 + 2 * 13), 2)


def build_agent_session(
    settings: Settings,
    *,
    language: str = "en",
    voice: str | None = None,
    persona: str | None = None,
    verbosity: int | None = None,
) -> AgentSession[None]:
    """Construct the STT→(brain)→TTS pipeline, language-aware.

    Voice provider is chosen by session language:

    * **English** → Deepgram (STT ``nova-2``/configured model + Aura-2 TTS).
      Deepgram gives lower-latency streaming STT and natural Aura voices.
    * **Vietnamese** (or any non-English) → the OpenAI-compatible gateway,
      because Deepgram Aura TTS is English-only and Deepgram STT does not
      support Vietnamese. This matches the REST narration fallback policy.

    ``voice`` is the config's chosen Deepgram Aura voice (English only),
    validated against the narration allow-list; an unknown/absent value
    degrades to the deployment default. No effect on the Vietnamese gateway.

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
                model=narration.resolve_tts_voice(voice, settings=settings),
                api_key=dg_key,
                base_url=tts_base,
            ),
            vad=silero.VAD.load(),
            turn_handling=_turn_handling(),
        )

    # Non-English (Vietnamese) — Deepgram cannot serve this locale; use the
    # OpenAI-compatible gateway for both STT and TTS (same key/base_url).
    #
    # Persona is applied HERE and not on the Deepgram branch above by necessity,
    # not by preference: the Deepgram plugin exposes no voice-independent knobs
    # (its voice IS the model id, already chosen by the teacher), whereas the
    # OpenAI-compatible TTS takes voice + speed. Before this, the Vietnamese
    # path passed neither, so a strict and a supportive interviewer sounded
    # identical in VI while differing in EN — a silent asymmetry between the two
    # languages the platform actually serves.
    api_key = settings.llm_api_key or NOT_GIVEN
    base_url = settings.llm_base_url or NOT_GIVEN
    return AgentSession[None](
        stt=openai.STT(model=settings.whisper_model, api_key=api_key, base_url=base_url),
        tts=openai.TTS(
            api_key=api_key,
            base_url=base_url,
            voice=narration.voice_for_persona(persona),
            speed=speech_rate_from_verbosity(verbosity if verbosity is not None else 2),
        ),
        vad=silero.VAD.load(),
        turn_handling=_turn_handling(),
    )


def _turn_handling() -> TurnHandlingOptions:
    """Turn-taking configuration shared by both language paths.

    Only endpointing is set. Interruption handling is deliberately left to the
    SDK's auto-detection: on the English path Deepgram reports word-aligned
    transcripts, so it selects the ML interruption detector that distinguishes a
    real interruption from a cough or an "mm-hm"; the Vietnamese path's STT does
    not report alignment, so the same setting resolves to VAD. Naming
    ``mode="adaptive"`` explicitly would not improve English and would make the
    Vietnamese session log a warning and disable the feature.

    ``resume_false_interruption`` is likewise untouched — it already defaults to
    True with a 2s timeout, which is what makes a nervous candidate's cough stop
    truncating the question they were being asked.
    """
    return {
        "endpointing": {
            "mode": "dynamic",
            "min_delay": _ENDPOINTING_MIN_DELAY_S,
            "max_delay": _ENDPOINTING_MAX_DELAY_S,
        },
    }


__all__ = ["InterviewAgent", "build_agent_session"]
