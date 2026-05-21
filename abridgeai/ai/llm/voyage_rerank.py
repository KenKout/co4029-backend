"""Voyage AI rerank-2.5 client (Phase 4 contextual-RAG upgrade).

Anthropic's contextual-retrieval study reports a 67% reduction in retrieval
failures when contextual embeddings + BM25 are combined with a reranker;
Voyage rerank-2.5 sits at the speed/quality sweet spot for our scale and
the 200M free-tier tokens cover the production workload comfortably.

This module is intentionally minimal: one HTTP path
(``POST {base}/rerank``), one happy-path response shape, no audit row
(retrieval is fanned out enough that adding a row per call would dominate
the ``ai_model_calls`` table — we surface latency_ms/usage in logs instead).

Graceful degradation
--------------------
When ``VOYAGE_API_KEY`` is unset, :func:`rerank_chunks` is short-circuited
by the caller (``retrieve_chunks``) and the MMR-only path is used. This
module never raises ``ConfigError`` — that decision lives at the caller
boundary so tests can construct a client without env access.

Cost guardrails
---------------
Voyage charges per token (input + output documents). We cap the rerank
input at the top-K MMR pool (default 30) and clamp document length to
~2000 chars per chunk (a Voyage doc is ~2k tokens worst-case) so a single
rerank call stays well under 100K tokens.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import httpx

from abridgeai.ai.llm.errors import ProviderError, ResponseFormatError

logger = logging.getLogger(__name__)

# Per-call upper bounds. Voyage accepts up to 1000 docs per request but
# we never ship anywhere near that — quiz retrieval pools at ~30 chunks.
_MAX_DOCUMENTS = 200
_MAX_DOC_CHARS = 2000
_MAX_QUERY_CHARS = 4000


@dataclass(frozen=True)
class RerankResult:
    """Single result row from :func:`rerank_chunks`.

    ``index`` matches the position of the document in the input list — the
    caller uses it to reorder the original ``ChunkWithDistance`` pool.
    ``relevance_score`` is the Voyage cross-encoder score (range
    ``[0.0, 1.0]``, larger == more relevant).
    """

    index: int
    relevance_score: float


class VoyageRerankClient:
    """Thin httpx wrapper around ``POST /rerank``.

    Construct fresh per call (cheap — no connection pooling needed at our
    fanout). The client is plain-data: it does not write audit rows or
    apply business logic, exactly mirroring :class:`OpenAICompatibleClient`.
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str = "https://api.voyageai.com/v1",
        timeout_s: float = 15.0,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout_s = timeout_s

    async def rerank(
        self,
        query: str,
        documents: list[str],
        *,
        top_k: int | None = None,
    ) -> tuple[list[RerankResult], int]:
        """POST a rerank request, return ``(results, latency_ms)``.

        Parameters
        ----------
        query
            The query string (truncated at :data:`_MAX_QUERY_CHARS`).
        documents
            Candidate documents to score (each truncated at
            :data:`_MAX_DOC_CHARS`). Empty list short-circuits to ``[]``.
        top_k
            Optional cap on results returned. ``None`` returns scores
            for every document; smaller values are honoured by Voyage
            server-side and reduce response payload.

        Raises:
            ProviderError: 4xx/5xx from upstream, network error.
            ResponseFormatError: response is not JSON or shape mismatch.
        """
        if not documents:
            return [], 0

        truncated_docs = [doc[:_MAX_DOC_CHARS] for doc in documents[:_MAX_DOCUMENTS]]
        payload: dict[str, Any] = {
            "query": query[:_MAX_QUERY_CHARS],
            "documents": truncated_docs,
            "model": self._model,
        }
        if top_k is not None and top_k > 0:
            payload["top_k"] = min(int(top_k), len(truncated_docs))

        url = f"{self._base_url}/rerank"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        started = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=self._timeout_s) as client:
                response = await client.post(url, headers=headers, json=payload)
        except httpx.HTTPError as exc:
            raise ProviderError(
                f"HTTP error calling {url}: {type(exc).__name__}: {exc}"
            ) from exc

        latency_ms = int((time.perf_counter() - started) * 1000)

        if response.status_code >= 400:
            body_excerpt = response.text[:500]
            raise ProviderError(
                f"voyage rerank returned HTTP {response.status_code}: {body_excerpt}"
            )

        try:
            body = response.json()
        except ValueError as exc:
            raise ResponseFormatError(
                f"voyage rerank returned non-JSON body: {response.text[:500]}"
            ) from exc

        return _parse_rerank_response(body), latency_ms


def _parse_rerank_response(body: dict[str, Any]) -> list[RerankResult]:
    """Convert a Voyage rerank response body to :class:`RerankResult` rows.

    Voyage shape (verified against rerank-2.5, Nov 2025)::

        {
          "object": "list",
          "data": [
            {"index": 2, "relevance_score": 0.91},
            {"index": 0, "relevance_score": 0.74},
            ...
          ],
          "model": "rerank-2.5",
          "usage": {"total_tokens": 1234}
        }

    Results come back ordered by descending ``relevance_score`` already.
    """
    data = body.get("data")
    if not isinstance(data, list):
        raise ResponseFormatError(
            f"voyage rerank: expected 'data' list, got {type(data).__name__}"
        )

    results: list[RerankResult] = []
    for row in data:
        if not isinstance(row, dict):
            raise ResponseFormatError(
                f"voyage rerank: expected dict rows in 'data', got {type(row).__name__}"
            )
        try:
            index = int(row["index"])
            score = float(row["relevance_score"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ResponseFormatError(
                f"voyage rerank: malformed row {row!r}"
            ) from exc
        results.append(RerankResult(index=index, relevance_score=score))

    return results


__all__ = ["RerankResult", "VoyageRerankClient"]
