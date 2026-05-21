"""Unit tests for the Voyage rerank-2.5 HTTP client (Phase 4).

Mocks ``httpx.AsyncClient.post`` rather than going over the wire — the
contract under test is request shaping + response parsing, not network
behaviour. Real provider behaviour is exercised by the staged smoke
test in ``scripts/`` (out-of-band).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from abridgeai.ai.llm.errors import ProviderError, ResponseFormatError
from abridgeai.ai.llm.voyage_rerank import (
    RerankResult,
    VoyageRerankClient,
    _parse_rerank_response,
)


def _ok_response(rows: list[dict[str, float]]) -> MagicMock:
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {
        "object": "list",
        "data": rows,
        "model": "rerank-2.5",
        "usage": {"total_tokens": 100},
    }
    return response


def _client() -> VoyageRerankClient:
    return VoyageRerankClient(
        api_key="test-key",
        model="rerank-2.5",
        base_url="https://api.voyageai.com/v1",
        timeout_s=5.0,
    )


@pytest.mark.asyncio
async def test_rerank_returns_results_in_score_order() -> None:
    response = _ok_response(
        [
            {"index": 2, "relevance_score": 0.91},
            {"index": 0, "relevance_score": 0.74},
            {"index": 1, "relevance_score": 0.30},
        ]
    )
    post = AsyncMock(return_value=response)
    with patch("httpx.AsyncClient") as ctx:
        ctx.return_value.__aenter__.return_value = MagicMock(post=post)
        results, latency = await _client().rerank(
            "what is recursion?",
            ["doc-zero", "doc-one", "doc-two"],
            top_k=3,
        )

    assert [r.index for r in results] == [2, 0, 1]
    assert results[0].relevance_score == pytest.approx(0.91)
    assert latency >= 0


@pytest.mark.asyncio
async def test_rerank_short_circuits_empty_documents() -> None:
    post = AsyncMock(side_effect=AssertionError("must not POST"))
    with patch("httpx.AsyncClient") as ctx:
        ctx.return_value.__aenter__.return_value = MagicMock(post=post)
        results, latency = await _client().rerank("any", [])

    assert results == []
    assert latency == 0


@pytest.mark.asyncio
async def test_rerank_truncates_overlong_documents() -> None:
    captured: dict[str, object] = {}

    async def capture_post(url: str, **kwargs: object) -> MagicMock:
        captured["payload"] = kwargs.get("json")
        return _ok_response([{"index": 0, "relevance_score": 0.5}])

    with patch("httpx.AsyncClient") as ctx:
        ctx.return_value.__aenter__.return_value = MagicMock(post=capture_post)
        await _client().rerank("q", ["X" * 5000])

    payload = captured["payload"]
    assert isinstance(payload, dict)
    documents = payload["documents"]
    assert isinstance(documents, list)
    # _MAX_DOC_CHARS == 2000 in voyage_rerank.py
    assert len(documents[0]) == 2000


@pytest.mark.asyncio
async def test_rerank_caps_top_k_to_document_count() -> None:
    captured: dict[str, object] = {}

    async def capture_post(url: str, **kwargs: object) -> MagicMock:
        captured["payload"] = kwargs.get("json")
        return _ok_response([])

    with patch("httpx.AsyncClient") as ctx:
        ctx.return_value.__aenter__.return_value = MagicMock(post=capture_post)
        await _client().rerank("q", ["a", "b"], top_k=999)

    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert payload["top_k"] == 2


@pytest.mark.asyncio
async def test_rerank_raises_provider_error_on_4xx() -> None:
    response = MagicMock()
    response.status_code = 401
    response.text = '{"error":"invalid api key"}'
    post = AsyncMock(return_value=response)
    with patch("httpx.AsyncClient") as ctx:
        ctx.return_value.__aenter__.return_value = MagicMock(post=post)
        with pytest.raises(ProviderError) as exc_info:
            await _client().rerank("q", ["doc"])

    assert "401" in str(exc_info.value)


@pytest.mark.asyncio
async def test_rerank_raises_provider_error_on_network_failure() -> None:
    post = AsyncMock(side_effect=httpx.ConnectError("network down"))
    with patch("httpx.AsyncClient") as ctx:
        ctx.return_value.__aenter__.return_value = MagicMock(post=post)
        with pytest.raises(ProviderError) as exc_info:
            await _client().rerank("q", ["doc"])

    assert "ConnectError" in str(exc_info.value)


@pytest.mark.asyncio
async def test_rerank_raises_format_error_on_non_json_body() -> None:
    response = MagicMock()
    response.status_code = 200
    response.json.side_effect = ValueError("not json")
    response.text = "html instead of json"
    post = AsyncMock(return_value=response)
    with patch("httpx.AsyncClient") as ctx:
        ctx.return_value.__aenter__.return_value = MagicMock(post=post)
        with pytest.raises(ResponseFormatError):
            await _client().rerank("q", ["doc"])


def test_parse_rerank_response_rejects_missing_data_key() -> None:
    with pytest.raises(ResponseFormatError):
        _parse_rerank_response({"oops": "no data"})


def test_parse_rerank_response_rejects_malformed_row() -> None:
    with pytest.raises(ResponseFormatError):
        _parse_rerank_response({"data": [{"index": "not-int"}]})


def test_rerank_result_is_immutable() -> None:
    row = RerankResult(index=0, relevance_score=0.5)
    with pytest.raises(AttributeError):
        row.index = 1  # type: ignore[misc]
