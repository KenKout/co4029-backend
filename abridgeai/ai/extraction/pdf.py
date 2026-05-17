"""PDF extractor — pymupdf with a latin-1 fallback for binaries that ship with no parser.

Ports the PDF code path from ``backend/app/ai/haystack/components/extractors.py``.
Pages flow into the registry as one ``SourceLocation`` per page so downstream
chunkers can attribute every chunk to a specific page in the source PDF.
"""

from __future__ import annotations

import asyncio
import re
from typing import BinaryIO

from abridgeai.ai.extraction.base import ExtractedContent, SourceLocation
from abridgeai.ai.extraction.registry import register_extractor


@register_extractor("application/pdf")
class PdfExtractor:
    async def extract(self, source: BinaryIO | bytes | str) -> ExtractedContent:
        raw = await asyncio.to_thread(_read_source, source)
        return await asyncio.to_thread(_extract_sync, raw)


def _read_source(source: BinaryIO | bytes | str) -> bytes:
    if isinstance(source, bytes):
        return source
    if isinstance(source, str):
        with open(source, "rb") as fh:
            return fh.read()
    return source.read()


def _extract_sync(raw: bytes) -> ExtractedContent:
    try:
        import fitz  # type: ignore[import-untyped,unused-ignore]
    except ImportError:
        return _fallback(raw)

    doc = fitz.open(stream=raw, filetype="pdf")
    try:
        page_count = doc.page_count
        parts: list[str] = []
        locations: list[SourceLocation] = []
        for page_num, page in enumerate(doc, start=1):
            blocks = page.get_text("blocks")
            page_lines = [b[4].strip() for b in blocks if b[4].strip()]
            if page_lines:
                parts.append(f"[Page {page_num}]\n" + "\n".join(page_lines))
                locations.append(SourceLocation(page=page_num))
    finally:
        doc.close()

    text = "\n\n".join(parts).strip()
    return ExtractedContent(
        text=text,
        metadata={"page_count": page_count},
        source_type="pdf",
        source_locations=locations,
    )


def _fallback(raw: bytes) -> ExtractedContent:
    """Latin-1 decode + ASCII-only filter; used when pymupdf is unavailable.

    Preserved verbatim from the legacy extractor so behaviour for environments
    without the pymupdf wheel does not regress. Produces no source_locations.
    """
    text = raw.decode("latin-1", errors="replace")
    text = re.sub(r"[^\x20-\x7E\n\t]", " ", text)
    text = re.sub(r" {2,}", " ", text)
    lines = [ln.strip() for ln in text.splitlines() if len(ln.strip()) > 20]
    body = "\n".join(lines)
    return ExtractedContent(
        text=body,
        metadata={"page_count": None, "fallback_parser": True},
        source_type="pdf",
        source_locations=[],
    )


__all__ = ["PdfExtractor"]
