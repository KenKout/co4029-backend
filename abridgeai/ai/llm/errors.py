"""Exception types raised by the LLM integration layer.

These are caught by ``LLMGateway`` / ``EmbeddingClient`` and either re-raised as
``abridgeai.core.exceptions.AppError`` (so workers mark the job failed) or
surfaced at startup before any HTTP call happens.
"""

from __future__ import annotations


class ConfigError(Exception):
    """Raised at startup when LLM/embedding configuration is invalid.

    Examples: missing ``LLM_API_KEY``, disallowed key in
    ``LLM_EXTRA_HEADERS_JSON``, ``EMBEDDING_DIMENSIONS`` mismatching the
    ``document_chunks.embedding`` pgvector column width.
    """


class ProviderError(Exception):
    """Raised when the upstream OpenAI-compatible endpoint returns an HTTP
    error (4xx, 5xx) or refuses authentication.

    Wraps ``httpx.HTTPStatusError`` and similar so callers can catch a single
    exception type without depending on httpx internals.
    """


class ResponseFormatError(Exception):
    """Raised when the upstream response cannot be parsed into the expected
    shape (non-JSON content, missing required keys, JSON decode failure).
    """
