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

import logging
from uuid import UUID

from livekit.agents import (
    NOT_GIVEN,
    Agent,
    AgentSession,
    ChatContext,
    ChatMessage,
    StopResponse,
    get_job_context,
)
from livekit.plugins import openai, silero

from abridgeai.core.config import Settings
from abridgeai.features.interviews.realtime import orchestration_bridge as bridge

logger = logging.getLogger(__name__)

# The agent is non-generative (no LLM); instructions are required by the Agent
# base class but never sent to a model. Kept descriptive for clarity/audit.
_INSTRUCTIONS = (
    "You are an AI mock interviewer. You ask the provided interview questions "
    "and listen to the candidate's spoken answers. You never invent questions."
)
_CLOSING_REMARK = "Thank you. That concludes the interview."


class InterviewAgent(Agent):
    """Speaks bank questions / follow-ups; routes answers to the text brain."""

    def __init__(
        self,
        *,
        interview_session_id: UUID,
        student_id: UUID,
        first_question_text: str | None,
    ) -> None:
        super().__init__(instructions=_INSTRUCTIONS)
        self._interview_session_id = interview_session_id
        self._student_id = student_id
        self._first_question_text = first_question_text

    async def on_enter(self) -> None:
        """Speak the current pending question when the agent joins the room."""
        if self._first_question_text:
            await self.session.say(self._first_question_text, allow_interruptions=False)

    async def on_user_turn_completed(
        self, turn_ctx: ChatContext, new_message: ChatMessage
    ) -> None:
        """Route the transcribed answer to the text brain; speak the result.

        ``StopResponse`` is raised at the end so the pipeline does not attempt
        a default (LLM) reply — we have already said everything we intend to.
        """
        del turn_ctx
        transcript = (new_message.text_content or "").strip()
        if not transcript:
            raise StopResponse()

        try:
            result = await bridge.handle_student_turn(
                self._interview_session_id, self._student_id, transcript
            )
        except Exception:
            logger.exception(
                "interview turn failed (session=%s)", self._interview_session_id
            )
            raise StopResponse() from None

        if result.speak_text:
            await self.session.say(result.speak_text)

        if result.is_finished:
            await self.session.say(_CLOSING_REMARK, allow_interruptions=False)
            ctx = get_job_context(required=False)
            if ctx is not None:
                ctx.shutdown(reason="interview_complete")

        raise StopResponse()


def build_agent_session(settings: Settings) -> AgentSession[None]:
    """Construct the STT→(brain)→TTS pipeline.

    STT/TTS reuse the OpenAI-compatible LLM credentials (same key/base_url as
    the gateway). silero VAD handles turn detection (KISS — no turn-detector
    model until manual E2E shows it is needed).
    """
    api_key = settings.llm_api_key or NOT_GIVEN
    base_url = settings.llm_base_url or NOT_GIVEN

    return AgentSession[None](
        stt=openai.STT(model=settings.whisper_model, api_key=api_key, base_url=base_url),
        tts=openai.TTS(api_key=api_key, base_url=base_url),
        vad=silero.VAD.load(),
    )


__all__ = ["InterviewAgent", "build_agent_session"]
