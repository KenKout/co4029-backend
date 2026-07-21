"""Interview narration TTS (browser-played, agent-quality voice).

Text and hybrid interview sessions run over REST — the student reads (and,
in hybrid, can dictate) answers. To give those sessions the same *voice*
quality as the LiveKit voice agent WITHOUT mounting a realtime room (which
would race the REST loop for control of the session), we expose a small
server-side TTS endpoint: the browser POSTs the question text, we synthesize
it with the SAME OpenAI-compatible TTS the agent uses (``settings.llm_*``),
and stream back an MP3 the browser plays via an ``<audio>`` element.

Persona → voice mapping gives "strict" a firmer voice and "supportive" a
warmer one, delivering the persona promise the config makes.

This module is the ONLY place that talks to the gateway TTS endpoint, so if
that surface shifts this is the single file to touch.
"""

from __future__ import annotations

import logging

import httpx

from abridgeai.core.config import Settings

logger = logging.getLogger(__name__)

# Persona → OpenAI TTS voice. Voices are the six built-ins exposed by the
# gateway (verified available). Chosen for tonal fit:
#   - strict:     "onyx"    — deep, firm, authoritative.
#   - neutral:    "alloy"   — balanced, default.
#   - supportive: "shimmer" — soft, warm, encouraging.
_PERSONA_VOICE: dict[str, str] = {
    "strict": "onyx",
    "neutral": "alloy",
    "supportive": "shimmer",
}
_DEFAULT_VOICE = "alloy"

# Fast, cheap model — narration does not need HD. Verified available on the
# gateway alongside tts-1-hd and gpt-4o-mini-tts.
_TTS_MODEL = "tts-1"

# Guard rail: questions are short. Cap input so a malformed/huge payload can't
# turn into a runaway TTS bill or a multi-megabyte response.
_MAX_CHARS = 1200

_TIMEOUT = httpx.Timeout(30.0, connect=10.0)


class NarrationUnavailable(RuntimeError):  # noqa: N818 - public API name
    """Raised when TTS cannot be produced (no credentials, gateway error)."""


def voice_for_persona(persona: str | None) -> str:
    """Map an interview persona to a TTS voice name (falls back to neutral)."""
    if persona is None:
        return _DEFAULT_VOICE
    return _PERSONA_VOICE.get(persona, _DEFAULT_VOICE)


async def synthesize_speech(
    text: str,
    *,
    persona: str | None,
    settings: Settings,
) -> bytes:
    """Synthesize ``text`` to MP3 bytes using the gateway TTS endpoint.

    Raises :class:`NarrationUnavailable` when credentials are missing or the
    gateway returns a non-2xx response, so the router can map it to a 503
    (the browser then falls back to its local speech synthesizer).
    """
    clean = (text or "").strip()
    if not clean:
        raise NarrationUnavailable("empty text")
    if len(clean) > _MAX_CHARS:
        clean = clean[:_MAX_CHARS]

    base_url = (settings.llm_base_url or "").rstrip("/")
    api_key = settings.llm_api_key
    if not base_url or not api_key:
        raise NarrationUnavailable("TTS credentials not configured")

    url = f"{base_url}/audio/speech"
    payload = {
        "model": _TTS_MODEL,
        "voice": voice_for_persona(persona),
        "input": clean,
        "response_format": "mp3",
    }
    headers = {"Authorization": f"Bearer {api_key}"}

    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.post(url, json=payload, headers=headers)
    except httpx.HTTPError as exc:
        logger.warning("narration TTS request failed: %s", exc)
        raise NarrationUnavailable("TTS request failed") from exc

    if resp.status_code != 200:
        logger.warning(
            "narration TTS non-200: status=%s body=%.200s",
            resp.status_code,
            resp.text,
        )
        raise NarrationUnavailable(f"TTS gateway status {resp.status_code}")

    audio = resp.content
    if not audio:
        raise NarrationUnavailable("empty audio response")
    return audio


__all__ = ["synthesize_speech", "voice_for_persona", "NarrationUnavailable"]
