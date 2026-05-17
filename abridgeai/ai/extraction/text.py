"""Plain-text + Markdown extractor.

Decodes raw bytes (UTF-8 with BOM stripped) and falls back to latin-1 when the
input is not valid UTF-8 — matches the legacy behaviour for files that arrive
from Windows / mainframe sources without proper encoding metadata.
"""

from __future__ import annotations

import asyncio
from typing import BinaryIO

from abridgeai.ai.extraction.base import ExtractedContent, SourceLocation
from abridgeai.ai.extraction.registry import register_extractor


def _decode(raw: bytes) -> str:
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("latin-1", errors="replace")


def _read_source(source: BinaryIO | bytes | str) -> bytes:
    if isinstance(source, bytes):
        return source
    if isinstance(source, str):
        with open(source, "rb") as fh:
            return fh.read()
    return source.read()


def _build(text: str, *, source_type: str, mime: str) -> ExtractedContent:
    body = text.strip()
    line_count = body.count("\n") + 1 if body else 0
    locations = [SourceLocation(line_start=1, line_end=line_count)] if line_count else []
    return ExtractedContent(
        text=body,
        metadata={"line_count": line_count, "mime_type": mime},
        source_type=source_type,
        source_locations=locations,
    )


@register_extractor("text/plain")
@register_extractor("text/markdown")
class TextExtractor:
    async def extract(self, source: BinaryIO | bytes | str) -> ExtractedContent:
        raw = await asyncio.to_thread(_read_source, source)
        text = await asyncio.to_thread(_decode, raw)
        return _build(text, source_type="text", mime="text/plain")


__all__ = ["TextExtractor"]
