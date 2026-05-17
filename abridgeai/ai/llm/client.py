"""Thin httpx wrapper around an OpenAI-compatible HTTP API.

This module knows about ``POST /chat/completions`` and ``POST /embeddings``
and *nothing else*. It does not interpret response content, decide cost,
write audit rows, or apply business logic — those belong to the gateway and
embedding-client modules layered on top.

Generic retry is delegated to ``processing_jobs.retry_count`` at the job
level; adding HTTP-level retry for *all* failures would create double-retry
semantics and make ``latency_ms`` measurements unreliable.

We DO retry HTTP 429 (rate-limited) inline because:
  1. 429 is a documented "wait then retry" signal — the upstream is asking
     us to back off, not telling us the request is wrong.
  2. The standard ``Retry-After`` header tells us exactly how long to wait.
  3. High-fanout pipelines (chunking enrichment, coverage-mode quiz) emit
     dozens of concurrent calls; without 429 retry a free-tier provider
     hits the per-minute limit and wastes the entire fanout's budget.
  4. The retry is bounded (3 attempts) and exponential, so we never
     hammer a degraded provider.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from typing import Any

import httpx

from abridgeai.ai.llm.errors import ProviderError, ResponseFormatError
from abridgeai.ai.llm.roles import ModelBinding

logger = logging.getLogger(__name__)


# Bounded inline retry policy for HTTP 429 only. Any other failure
# (4xx other than 429, 5xx, network errors) propagates immediately and
# the job-level retry mechanism handles it from there.
_RATE_LIMIT_MAX_ATTEMPTS = 4  # 1 initial + 3 retries
_RATE_LIMIT_BASE_DELAY_S = 1.0
_RATE_LIMIT_MAX_DELAY_S = 30.0


def _parse_retry_after(header_value: str | None) -> float | None:
    """Parse the standard ``Retry-After`` header. Either an int (seconds)
    or an HTTP-date. We only honour the integer form; HTTP-dates are rare
    in API contexts and parsing them adds dependencies.
    """
    if not header_value:
        return None
    try:
        return max(0.0, float(header_value.strip()))
    except (TypeError, ValueError):
        return None


def _next_backoff(attempt: int, retry_after: float | None) -> float:
    """Decide how long to sleep before the next retry.

    Prefer ``Retry-After`` when the provider offers one (capped at
    ``_RATE_LIMIT_MAX_DELAY_S`` to bound the worst case). Fall back to
    exponential backoff with jitter so multiple parallel callers don't
    re-collide at exactly the same wall-clock instant.
    """
    if retry_after is not None:
        return min(retry_after, _RATE_LIMIT_MAX_DELAY_S)
    base: float = _RATE_LIMIT_BASE_DELAY_S * (2 ** (attempt - 1))
    jitter: float = random.uniform(0.0, _RATE_LIMIT_BASE_DELAY_S)  # noqa: S311  # nosec B311 - jitter, not security
    return min(_RATE_LIMIT_MAX_DELAY_S, base + jitter)


class OpenAICompatibleClient:
    """One client, one binding. Construct fresh per call (cheap)."""

    def __init__(self, binding: ModelBinding) -> None:
        self._binding = binding

    async def chat_completions_json(
        self,
        messages: list[dict[str, Any]],
        *,
        response_format: dict[str, Any],
    ) -> tuple[dict[str, Any], int]:
        """POST a chat-completion request, return ``(body, latency_ms)``.

        Raises:
            ProviderError: 4xx/5xx from upstream, network error, auth failure.
            ResponseFormatError: response is not JSON.
        """
        payload: dict[str, Any] = {
            "model": self._binding.model,
            "messages": messages,
            "response_format": response_format,
        }
        return await self._post("chat/completions", payload)

    async def embeddings(
        self,
        texts: list[str],
        *,
        dimensions: int,
    ) -> tuple[dict[str, Any], int]:
        """POST an embeddings request, return ``(body, latency_ms)``."""
        payload: dict[str, Any] = {
            "model": self._binding.model,
            "input": texts,
            "dimensions": dimensions,
        }
        return await self._post("embeddings", payload)

    async def _post(self, path: str, payload: dict[str, Any]) -> tuple[dict[str, Any], int]:
        url = f"{self._binding.base_url.rstrip('/')}/{path}"
        headers = {
            "Authorization": f"Bearer {self._binding.api_key}",
            "Content-Type": "application/json",
            **self._binding.extra_headers,
        }

        # ``latency_ms`` only measures the FINAL successful (or final-failure)
        # round-trip — retry sleeps are excluded because they're the upstream
        # policy, not our compute time. The retry loop tracks them separately
        # in the warn log so operators can spot quota-thrash.
        attempt = 0
        last_429_body: str | None = None
        last_retry_after: float | None = None

        while True:
            attempt += 1
            started = time.perf_counter()
            try:
                async with httpx.AsyncClient(timeout=self._binding.timeout_s) as client:
                    response = await client.post(url, headers=headers, json=payload)
            except httpx.HTTPError as exc:
                raise ProviderError(
                    f"HTTP error calling {url}: {type(exc).__name__}: {exc}"
                ) from exc

            latency_ms = int((time.perf_counter() - started) * 1000)

            if response.status_code == 429:
                last_429_body = response.text[:500]
                last_retry_after = _parse_retry_after(response.headers.get("Retry-After"))
                if attempt >= _RATE_LIMIT_MAX_ATTEMPTS:
                    raise ProviderError(
                        f"{path} returned HTTP 429 after {attempt} attempts: {last_429_body}"
                    )
                sleep_for = _next_backoff(attempt, last_retry_after)
                logger.warning(
                    "rate-limited on %s (attempt %d/%d); sleeping %.2fs",
                    path,
                    attempt,
                    _RATE_LIMIT_MAX_ATTEMPTS,
                    sleep_for,
                )
                await asyncio.sleep(sleep_for)
                continue

            if response.status_code >= 400:
                body_excerpt = response.text[:500]
                raise ProviderError(f"{path} returned HTTP {response.status_code}: {body_excerpt}")

            try:
                body = response.json()
            except ValueError as exc:
                raise ResponseFormatError(
                    f"{path} returned non-JSON body: {response.text[:500]}"
                ) from exc

            return body, latency_ms
