"""Stage A — rule-based windowing for the SemanticChunker.

Splits an ``ExtractedContent`` body into overlapping token-bounded windows
that respect paragraph boundaries and source-location metadata. The output
is a list of ``RawChunk`` whose ``metadata`` carries:

* ``page`` / ``slide`` / ``line_start`` / ``line_end`` from the matched
  ``SourceLocation`` (when available).
* ``content_role`` — rule-based hint (``body`` / ``summary`` / ``review`` /
  ``front_matter``) used as a fallback when Stage C is disabled, mirroring
  the legacy ``_content_role`` classifier.
* ``section`` — heading chain like ``"Course > Module > Page 5"``.
* ``token_count`` — exact cl100k_base token count.

Stage B (``_glue``) reads ``token_count`` and ``content_role`` to decide
which windows can merge.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from abridgeai.ai.chunking.base import RawChunk
from abridgeai.ai.chunking.token_aware import TokenAwareChunker, count_tokens

if TYPE_CHECKING:
    from abridgeai.ai.extraction import ExtractedContent

_SUMMARY_HEADINGS_RE = re.compile(
    r"\b("
    r"summary|recap|review\s*(?:questions|exercises)|conclusion|"
    r"learning\s*objectives|outline|table\s*of\s*contents|toc|"
    r"what\s*we\s*(?:learned|covered)|key\s*takeaways"
    r")\b",
    re.IGNORECASE,
)
_REVIEW_HEADING_RE = re.compile(
    r"\breview\s*(?:questions|exercises)\b",
    re.IGNORECASE,
)
_FRONT_MATTER_HEADING_RE = re.compile(
    r"\b("
    r"cover|title\s*page|syllabus|course\s*(?:info|information|overview)|"
    r"about\s*the\s*(?:author|instructor|course)|copyright|acknowledg(?:e?ments?|ments)|"
    r"contents|toc|preface|foreword|disclaimer"
    r")\b",
    re.IGNORECASE,
)
_FRONT_MATTER_BODY_RE = re.compile(
    r"\b("
    r"semester\s*\d|faculty\s+of|university\s+of|department\s+of|"
    r"academic\s+year|@\S+\.\S+|lecturer|instructor|prof\.?\s*[a-z]"
    r")\b",
    re.IGNORECASE,
)

_BULLET_PREFIXES = ("- ", "• ", "* ", "– ")
_BULLET_HEAVY_MIN_BULLETS = 5
_BULLET_HEAVY_MAX_BODY_CHARS = 800
_FRONT_MATTER_PAGE_LIMIT = 3
_FRONT_MATTER_BODY_MAX_CHARS = 600


def window_chunks(
    content: ExtractedContent,
    *,
    max_tokens: int = 800,
    overlap_tokens: int = 80,
    section_context: str = "",
) -> list[RawChunk]:
    """Stage A: window the document into role-tagged token-bounded chunks."""
    chunker = TokenAwareChunker(
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
        section_context=section_context,
    )
    raw = chunker.chunk(content)

    enriched: list[RawChunk] = []
    for ch in raw:
        section = str(ch.metadata.get("section") or "")
        heading = section.rsplit(">", 1)[-1].strip() if section else ""
        page = ch.metadata.get("page") or ch.metadata.get("slide")
        page_int = page if isinstance(page, int) else None
        role = _content_role(heading, ch.content, page=page_int)

        new_md = dict(ch.metadata)
        new_md.setdefault("token_count", count_tokens(ch.content))
        new_md["content_role"] = role
        new_md.setdefault("source_type", content.source_type)
        enriched.append(RawChunk(content=ch.content, chunk_index=ch.chunk_index, metadata=new_md))
    return enriched


def _content_role(
    heading: str,
    body: str,
    *,
    page: int | None = None,
) -> str:
    """Classify a (heading, body, page) triple into one of:
    ``body`` | ``summary`` | ``review`` | ``front_matter``.

    Mirrors the legacy chunker's classifier. Quiz retrieval downweights
    summary/review/front_matter so cover slides do not soak up question
    budget.
    """
    heading = heading or ""
    body = body or ""

    if _REVIEW_HEADING_RE.search(heading):
        return "review"
    if _FRONT_MATTER_HEADING_RE.search(heading):
        return "front_matter"

    if (
        page is not None
        and page <= _FRONT_MATTER_PAGE_LIMIT
        and len(body) <= _FRONT_MATTER_BODY_MAX_CHARS
        and _FRONT_MATTER_BODY_RE.search(body) is not None
    ):
        return "front_matter"

    if _SUMMARY_HEADINGS_RE.search(heading):
        return "summary"

    bullet_lines = sum(
        1
        for line in body.splitlines()
        if any(line.lstrip().startswith(prefix) for prefix in _BULLET_PREFIXES)
    )
    if bullet_lines >= _BULLET_HEAVY_MIN_BULLETS and len(body) < _BULLET_HEAVY_MAX_BODY_CHARS:
        return "summary"

    return "body"


def page_range_of(chunks: list[RawChunk]) -> tuple[int | None, int | None]:
    """Return ``(min_page, max_page)`` across a window's member chunks."""
    pages: list[int] = []
    for ch in chunks:
        page: Any = ch.metadata.get("page") or ch.metadata.get("slide")
        if isinstance(page, int):
            pages.append(page)
    if not pages:
        return (None, None)
    return (min(pages), max(pages))


__all__ = ["page_range_of", "window_chunks"]
