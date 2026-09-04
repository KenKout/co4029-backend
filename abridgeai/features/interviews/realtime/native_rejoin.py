"""Native-interview rejoin helpers (question re-read after a room rejoin).

Extracted from ``realtime.native_runtime`` (2026-09-01) to keep that module
under the interviews LOC ratchet.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from livekit.agents import (
    AgentSession,  # noqa: F401  -- annotation target
)

from abridgeai.core.observability import get_logger

logger = get_logger(__name__)


def _rejoin_question_text(question: str, language: str) -> str:
    """A short lead-in plus the verbatim question, for a mid-interview rejoin.

    The lead-in keeps the re-read out of the client's verbatim-dedup path (the
    pinned card already shows the bare question) and reads as a person re-stating
    the question, not a form being read aloud.
    """
    if (language or "en").lower().startswith("vi"):
        return f"Để tôi nhắc lại câu hỏi: {question}"
    return f"Let me repeat the question: {question}"


async def _re_read_question(
    session: AgentSession,
    question: str,
    language: str,
    announce: Callable[[str], Awaitable[None]] | None = None,
) -> None:
    """Re-speak the current question after a rejoin. Best-effort.

    The exact spoken text is announced on the control topic FIRST, so the
    client can dedupe this utterance against the pinned card by payload
    instead of pattern-matching the lead-in wording.
    """
    text = _rejoin_question_text(question, language)
    try:
        if announce is not None:
            await announce(text)
    except Exception:  # noqa: BLE001 -- the announcement is convenience only
        logger.warning("announcing the re-read failed; speaking it anyway")
    try:
        handle = session.say(text, allow_interruptions=False)
        await handle
    except Exception:  # noqa: BLE001 -- a failed re-read must not cost the session
        logger.exception("re-read question on rejoin failed")


__all__ = ["_re_read_question", "_rejoin_question_text"]
