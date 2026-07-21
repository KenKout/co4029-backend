"""Unit coverage for browser-played interview narration."""

from __future__ import annotations

import json
from typing import Any, cast

import httpx
import pytest
import respx

from abridgeai.features.interviews.services.narration import synthesize_speech


class _NarrationSettings:
    llm_base_url = "https://gateway.example.test/v1"
    llm_api_key = "test-key"


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
