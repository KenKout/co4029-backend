"""Interview narration TTS (browser-played, agent-quality voice).

Text and hybrid interview sessions run over REST — the student reads (and,
in hybrid, can dictate) answers. To give those sessions the same *voice*
quality as the LiveKit voice agent WITHOUT mounting a realtime room (which
would race the REST loop for control of the session), we expose a small
server-side TTS endpoint: the browser POSTs the question text, we synthesize
it, and stream back an MP3 the browser plays via an ``<audio>`` element.

Provider routing (by language):

  * ENGLISH — Deepgram Aura-2 when ``settings.deepgram_api_key`` is set
    (falls back to the OpenAI-compatible gateway when it is not, or when
    Deepgram errors).
  * VIETNAMESE — always the OpenAI-compatible gateway. Deepgram TTS is
    English-only, so routing ``vi`` there would silently break voice for
    every Vietnamese interview.

Persona → voice mapping gives "strict" a firmer voice and "supportive" a
warmer one, delivering the persona promise the config makes, on both
providers.

This module is the ONLY place that talks to a TTS endpoint, so if that
surface shifts this is the single file to touch.
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


def _deepgram_model_for_persona(persona: str | None, *, settings: Settings) -> str:
    """Return the English Deepgram Aura voice model for narration.

    Deepgram exposes one voice per model, so persona tone is not varied here
    (unlike the OpenAI voices); a single ``deepgram_tts_model_en`` applies.
    """
    del persona  # single English voice; kept for signature symmetry
    return settings.deepgram_tts_model_en


def _is_english(language: str | None) -> bool:
    """True when narration should use the English voice pipeline.

    Defaults to English for unknown/absent language codes — the assessment
    surface is English-first and Deepgram (when configured) only handles EN.
    """
    return not (language or "en").strip().lower().startswith("vi")


async def _synthesize_deepgram(
    text: str,
    *,
    persona: str | None,
    settings: Settings,
) -> bytes:
    """Synthesize English ``text`` to MP3 bytes via Deepgram Aura-2.

    Raises :class:`NarrationUnavailable` on any error so the caller can fall
    back to the OpenAI-compatible gateway.
    """
    api_key = settings.deepgram_api_key
    if api_key is None:
        raise NarrationUnavailable("Deepgram credentials not configured")

    base_url = (settings.deepgram_tts_base_url or "").rstrip("/")
    if not base_url:
        raise NarrationUnavailable("Deepgram base URL not configured")

    model = _deepgram_model_for_persona(persona, settings=settings)
    # Deepgram /v1/speak: model + audio container are query params; the text
    # is a JSON body ``{"text": ...}``. encoding=mp3 returns an MP3 stream that
    # the browser plays through the exact same <audio> path as the gateway.
    url = f"{base_url}/speak"
    params = {"model": model, "encoding": settings.deepgram_tts_encoding}
    payload = {"text": text}
    headers = {
        "Authorization": f"Token {api_key.get_secret_value()}",
        "Content-Type": "application/json",
    }

    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(settings.deepgram_tts_timeout_seconds, connect=10.0)
        ) as client:
            resp = await client.post(url, params=params, json=payload, headers=headers)
    except httpx.HTTPError as exc:
        logger.warning("Deepgram narration request failed: %s", exc)
        raise NarrationUnavailable("Deepgram request failed") from exc

    if resp.status_code != 200:
        logger.warning(
            "Deepgram narration non-200: status=%s body=%.200s",
            resp.status_code,
            resp.text,
        )
        raise NarrationUnavailable(f"Deepgram status {resp.status_code}")

    audio = resp.content
    if not audio:
        raise NarrationUnavailable("empty audio response")
    return audio


async def _synthesize_openai(
    text: str,
    *,
    persona: str | None,
    settings: Settings,
) -> bytes:
    """Synthesize ``text`` to MP3 bytes using the OpenAI-compatible gateway."""
    base_url = (settings.llm_base_url or "").rstrip("/")
    api_key = settings.llm_api_key
    if not base_url or not api_key:
        raise NarrationUnavailable("TTS credentials not configured")

    url = f"{base_url}/audio/speech"
    payload = {
        "model": _TTS_MODEL,
        "voice": voice_for_persona(persona),
        "input": text,
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


async def synthesize_speech(
    text: str,
    *,
    persona: str | None,
    settings: Settings,
    language: str | None = None,
) -> bytes:
    """Synthesize ``text`` to MP3 bytes, routing by ``language``.

    English narration prefers Deepgram Aura-2 when a Deepgram key is
    configured, transparently falling back to the OpenAI-compatible gateway if
    Deepgram is unavailable or errors. Vietnamese always uses the gateway
    (Deepgram TTS is English-only).

    Raises :class:`NarrationUnavailable` when no provider can produce audio, so
    the router maps it to a 503 (the browser then falls back to its local
    speech synthesizer).
    """
    clean = (text or "").strip()
    if not clean:
        raise NarrationUnavailable("empty text")
    if len(clean) > _MAX_CHARS:
        clean = clean[:_MAX_CHARS]

    use_deepgram = _is_english(language) and settings.deepgram_api_key is not None
    if use_deepgram:
        try:
            return await _synthesize_deepgram(clean, persona=persona, settings=settings)
        except NarrationUnavailable as exc:
            # Deepgram is best-effort for English; degrade to the gateway
            # rather than dropping to the browser's local voice.
            logger.info("Deepgram narration unavailable, falling back to gateway: %s", exc)

    return await _synthesize_openai(clean, persona=persona, settings=settings)


__all__ = ["synthesize_speech", "voice_for_persona", "NarrationUnavailable"]
