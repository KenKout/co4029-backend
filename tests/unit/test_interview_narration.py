"""Unit coverage for browser-played interview narration."""

from __future__ import annotations

import json
from typing import Any, cast

import httpx
import pytest
import respx
from pydantic import SecretStr

from abridgeai.features.interviews.services.narration import (
    NarrationUnavailable,
    synthesize_speech,
)


class _NarrationSettings:
    """Gateway-only settings (no Deepgram key configured)."""

    llm_base_url = "https://gateway.example.test/v1"
    llm_api_key = "test-key"
    deepgram_api_key: SecretStr | None = None
    deepgram_tts_base_url = "https://api.deepgram.com/v1"
    deepgram_tts_model_strict = "aura-2-orion-en"
    deepgram_tts_model_neutral = "aura-2-thalia-en"
    deepgram_tts_model_supportive = "aura-2-luna-en"
    deepgram_tts_timeout_seconds = 30.0


class _DeepgramSettings(_NarrationSettings):
    deepgram_api_key = SecretStr("dg-test-key")


@pytest.mark.asyncio
async def test_narration_synthesizes_the_exact_approved_utterance() -> None:
    with respx.mock(assert_all_called=True) as router:
        route = router.post("https://gateway.example.test/v1/audio/speech").mock(
            return_value=httpx.Response(200, content=b"mp3-audio")
        )
        audio = await synthesize_speech(
            "Thank you and goodbye.",
            persona="neutral",
            settings=cast(Any, _NarrationSettings()),
        )

    payload = json.loads(route.calls[0].request.content)
    assert payload["input"] == "Thank you and goodbye."
    assert audio == b"mp3-audio"


@pytest.mark.asyncio
async def test_english_narration_uses_deepgram_when_key_present() -> None:
    with respx.mock(assert_all_called=True) as router:
        route = router.post("https://api.deepgram.com/v1/speak").mock(
            return_value=httpx.Response(200, content=b"dg-audio")
        )
        audio = await synthesize_speech(
            "Compare fact tables and factless fact tables.",
            persona="strict",
            settings=cast(Any, _DeepgramSettings()),
            language="en",
        )

    request = route.calls[0].request
    assert request.url.params["model"] == "aura-2-orion-en"
    assert request.url.params["encoding"] == "mp3"
    assert request.headers["authorization"] == "Token dg-test-key"
    payload = json.loads(request.content)
    assert payload["text"] == "Compare fact tables and factless fact tables."
    assert audio == b"dg-audio"


@pytest.mark.asyncio
async def test_vietnamese_narration_always_uses_gateway_not_deepgram() -> None:
    # Deepgram TTS is English-only; vi must route to the OpenAI-compatible
    # gateway even when a Deepgram key is configured.
    with respx.mock(assert_all_called=False) as router:
        gateway = router.post("https://gateway.example.test/v1/audio/speech").mock(
            return_value=httpx.Response(200, content=b"vi-audio")
        )
        deepgram = router.post("https://api.deepgram.com/v1/speak").mock(
            return_value=httpx.Response(200, content=b"should-not-be-called")
        )
        audio = await synthesize_speech(
            "Xin chào và tạm biệt.",
            persona="neutral",
            settings=cast(Any, _DeepgramSettings()),
            language="vi",
        )

    assert gateway.called
    assert not deepgram.called
    assert audio == b"vi-audio"


@pytest.mark.asyncio
async def test_english_falls_back_to_gateway_when_deepgram_errors() -> None:
    with respx.mock(assert_all_called=True) as router:
        deepgram = router.post("https://api.deepgram.com/v1/speak").mock(
            return_value=httpx.Response(500, text="boom")
        )
        gateway = router.post("https://gateway.example.test/v1/audio/speech").mock(
            return_value=httpx.Response(200, content=b"fallback-audio")
        )
        audio = await synthesize_speech(
            "A fact table stores measurable events.",
            persona="neutral",
            settings=cast(Any, _DeepgramSettings()),
            language="en",
        )

    assert deepgram.called
    assert gateway.called
    assert audio == b"fallback-audio"


@pytest.mark.asyncio
async def test_empty_text_raises_before_any_request() -> None:
    with pytest.raises(NarrationUnavailable):
        await synthesize_speech(
            "   ",
            persona="neutral",
            settings=cast(Any, _DeepgramSettings()),
            language="en",
        )
