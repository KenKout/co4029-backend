from __future__ import annotations

import time

import httpx
import pytest
import respx

from abridgeai.ai.llm.client import OpenAICompatibleClient
from abridgeai.ai.llm.errors import ProviderError
from abridgeai.ai.llm.roles import LLMRole, ModelBinding


def _binding(timeout: float = 10.0) -> ModelBinding:
    return ModelBinding(
        role=LLMRole.EXTRACTION,
        tier="small",
        base_url="https://api.example.test/v1",
        api_key="sk-test",
        model="gpt-4o-mini",
        extra_headers={},
        timeout_s=timeout,
    )


@pytest.mark.asyncio
async def test_429_retry_with_retry_after_succeeds_on_second_attempt() -> None:
    binding = _binding()
    client = OpenAICompatibleClient(binding)

    with respx.mock(assert_all_called=True) as respx_mock:
        route = respx_mock.post("https://api.example.test/v1/chat/completions")
        success_body = {"choices": [{"message": {"content": "{}"}}], "usage": {}}
        route.side_effect = [
            httpx.Response(429, headers={"Retry-After": "2"}, text="rate limited"),
            httpx.Response(200, json=success_body),
        ]

        started = time.perf_counter()
        body, _latency_ms = await client.chat_completions_json(
            [{"role": "user", "content": "hi"}],
            response_format={"type": "json_object"},
        )
        elapsed = time.perf_counter() - started

    assert body == success_body
    assert route.call_count == 2
    assert elapsed >= 1.9


@pytest.mark.asyncio
async def test_429_retry_exhaustion_raises_provider_error() -> None:
    binding = _binding()
    client = OpenAICompatibleClient(binding)

    with respx.mock(assert_all_called=True) as respx_mock:
        route = respx_mock.post("https://api.example.test/v1/chat/completions")
        route.side_effect = [
            httpx.Response(429, headers={"Retry-After": "0"}, text="rate limited"),
            httpx.Response(429, headers={"Retry-After": "0"}, text="rate limited"),
            httpx.Response(429, headers={"Retry-After": "0"}, text="rate limited"),
            httpx.Response(429, headers={"Retry-After": "0"}, text="rate limited"),
        ]

        with pytest.raises(ProviderError, match="HTTP 429 after 4 attempts"):
            await client.chat_completions_json(
                [{"role": "user", "content": "hi"}],
                response_format={"type": "json_object"},
            )

    assert route.call_count == 4


@pytest.mark.asyncio
async def test_non_429_4xx_raises_immediately_no_retry() -> None:
    binding = _binding()
    client = OpenAICompatibleClient(binding)

    with respx.mock(assert_all_called=True) as respx_mock:
        route = respx_mock.post("https://api.example.test/v1/chat/completions")
        route.return_value = httpx.Response(400, text="bad request")

        with pytest.raises(ProviderError, match="HTTP 400"):
            await client.chat_completions_json(
                [{"role": "user", "content": "hi"}],
                response_format={"type": "json_object"},
            )

    assert route.call_count == 1
