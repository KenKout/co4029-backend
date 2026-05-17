"""HTML extractor.

Strips markup with BeautifulSoup, preserves heading hierarchy by emitting
``[Hn] ...`` markers, and records one ``SourceLocation`` per heading or
paragraph so chunkers can attribute output back to the source DOM section.
Falls back to a script/style stripper when ``beautifulsoup4`` is not
installed (matches the legacy graceful-degrade pattern).
"""

from __future__ import annotations

import asyncio
import re
from typing import BinaryIO

from abridgeai.ai.extraction.base import ExtractedContent, SourceLocation
from abridgeai.ai.extraction.registry import register_extractor

_HEADING_TAGS = ("h1", "h2", "h3", "h4", "h5", "h6")
_BODY_TAGS = (*_HEADING_TAGS, "p", "li")
_FALLBACK_SCRIPT_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.DOTALL | re.IGNORECASE)
_FALLBACK_TAG_RE = re.compile(r"<[^>]+>")
_FALLBACK_WS_RE = re.compile(r"[ \t]+")


def _read_source(source: BinaryIO | bytes | str) -> bytes:
    if isinstance(source, bytes):
        return source
    if isinstance(source, str):
        with open(source, "rb") as fh:
            return fh.read()
    return source.read()


def _decode(raw: bytes) -> str:
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("latin-1", errors="replace")


def _extract_with_bs4(html: str) -> ExtractedContent:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    parts: list[str] = []
    locations: list[SourceLocation] = []
    line_cursor = 0
    for element in soup.find_all(_BODY_TAGS):
        text = element.get_text(separator=" ", strip=True)
        if not text:
            continue
        tag_name = element.name
        rendered = f"[{tag_name.upper()}] {text}" if tag_name in _HEADING_TAGS else text
        parts.append(rendered)
        line_cursor += 1
        locations.append(SourceLocation(line_start=line_cursor, line_end=line_cursor))

    body = "\n\n".join(parts).strip()
    title = soup.title.string.strip() if soup.title and soup.title.string else None
    return ExtractedContent(
        text=body,
        metadata={
            "title": title,
            "block_count": len(parts),
            "fallback_parser": False,
        },
        source_type="html",
        source_locations=locations,
    )


def _extract_fallback(html: str) -> ExtractedContent:
    cleaned = _FALLBACK_SCRIPT_RE.sub(" ", html)
    cleaned = _FALLBACK_TAG_RE.sub("\n", cleaned)
    cleaned = _FALLBACK_WS_RE.sub(" ", cleaned)
    lines = [ln.strip() for ln in cleaned.splitlines() if ln.strip()]
    body = "\n".join(lines)
    locations = [SourceLocation(line_start=i, line_end=i) for i in range(1, len(lines) + 1)]
    return ExtractedContent(
        text=body,
        metadata={
            "title": None,
            "block_count": len(lines),
            "fallback_parser": True,
        },
        source_type="html",
        source_locations=locations,
    )


def _extract_sync(html: str) -> ExtractedContent:
    try:
        import bs4  # noqa: F401
    except ImportError:
        return _extract_fallback(html)
    return _extract_with_bs4(html)


@register_extractor("text/html")
class HtmlExtractor:
    async def extract(self, source: BinaryIO | bytes | str) -> ExtractedContent:
        raw = await asyncio.to_thread(_read_source, source)
        html = await asyncio.to_thread(_decode, raw)
        return await asyncio.to_thread(_extract_sync, html)


__all__ = ["HtmlExtractor"]
