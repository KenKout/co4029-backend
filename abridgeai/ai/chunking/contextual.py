"""Contextual prepending for embeddings (Anthropic Contextual Retrieval).

Stage C semantic enrichment already produces the two fields Anthropic's
prompt asks an LLM to generate per chunk:

  - ``section_title`` — short topic label for this window
  - ``context_sentence`` — 1-2 sentences situating the window in the
    wider document

This module exposes :func:`build_contextual_text` which prepends those
fields onto a chunk's content before it goes to the embedder. Doing so
inline at embed time means we get the Anthropic technique's retrieval
quality lift (-35% retrieval failure on the BEIR-style benchmark) for
free — no extra LLM calls beyond what Stage C already runs.

Format:

    [Topic: {section_title}] {context_sentence} {content}

When enrichment is missing (e.g. timestamp-aware chunks for video, or
when Stage C was skipped) the helper returns ``content`` unchanged so
callers don't need to special-case anything.

The prefix is bounded at ``max_prefix_tokens`` (default 150) so a runaway
``context_sentence`` from the LLM can't blow past the embedding model's
input window. Token budget is computed via the project's tiktoken
encoding (currently ``o200k_base`` to match GPT-5/GPT-4o).
"""

from __future__ import annotations

from abridgeai.ai.chunking.base import EnrichedChunk, RawChunk
from abridgeai.ai.chunking.token_aware import count_tokens, truncate_to_tokens

_DEFAULT_MAX_PREFIX_TOKENS = 150
"""Default cap for the prepended prefix in tokens.

Anthropic's contextual retrieval prompt asks the LLM for "50-100 tokens"
of situating context, so 150 leaves slack for the [Topic: ...] tag
without crowding out chunk content in the embedding model's input
window."""


def _read_semantic_field(chunk: RawChunk, key: str) -> str:
    """Fetch a Stage C semantic field from either source of truth.

    ``EnrichedChunk.semantic_metadata`` is the typed home for Stage C
    output. The ingestion pipeline also flattens it into
    ``metadata['semantic']`` when persisting (see ``_build_chunk_metadata``
    in ``materials/ingestion/pipeline.py``). This helper accepts both
    so callers can pass either freshly-produced ``EnrichedChunk`` objects
    or rehydrated chunks built from ``DocumentChunk.metadata_json``.
    """
    if isinstance(chunk, EnrichedChunk):
        value = chunk.semantic_metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    nested = (chunk.metadata or {}).get("semantic")
    if isinstance(nested, dict):
        value = nested.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def build_contextual_text(
    chunk: RawChunk,
    *,
    max_prefix_tokens: int = _DEFAULT_MAX_PREFIX_TOKENS,
) -> str:
    """Prepend Stage C semantic context to chunk content for embedding.

    Returns ``"[Topic: {title}] {context_sentence} {content}"`` when the
    chunk carries Stage C enrichment, ``content`` unchanged otherwise.
    Truncates the prefix at ``max_prefix_tokens`` so a verbose context
    sentence can't crowd out chunk content.

    The original ``content`` is preserved unmodified — only the embedding
    input is contextualized. ``DocumentChunk.content`` continues to store
    the bare chunk text, so panel previews and quiz prompts read the
    same content the user uploaded.
    """
    title = _read_semantic_field(chunk, "section_title")
    ctx = _read_semantic_field(chunk, "context_sentence")

    if not title and not ctx:
        return chunk.content

    parts: list[str] = []
    if title:
        parts.append(f"[Topic: {title}]")
    if ctx:
        parts.append(ctx)
    prefix = " ".join(parts)

    if count_tokens(prefix) > max_prefix_tokens:
        prefix = truncate_to_tokens(prefix, max_prefix_tokens)

    return f"{prefix} {chunk.content}"
