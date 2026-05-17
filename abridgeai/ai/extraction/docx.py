"""DOCX extractor — python-docx wrapping with paragraph-level source locations."""

from __future__ import annotations

import asyncio
import io
from typing import BinaryIO

from abridgeai.ai.extraction.base import ExtractedContent, SourceLocation
from abridgeai.ai.extraction.registry import register_extractor

_DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


@register_extractor(_DOCX_MIME)
class DocxExtractor:
    async def extract(self, source: BinaryIO | bytes | str) -> ExtractedContent:
        return await asyncio.to_thread(_extract_sync, source)


def _extract_sync(source: BinaryIO | bytes | str) -> ExtractedContent:
    import docx

    handle = io.BytesIO(source) if isinstance(source, bytes) else source
    document = docx.Document(handle)

    paragraphs: list[str] = []
    locations: list[SourceLocation] = []
    for index, paragraph in enumerate(document.paragraphs, start=1):
        text = paragraph.text.strip()
        if not text:
            continue
        paragraphs.append(text)
        locations.append(SourceLocation(line_start=index, line_end=index))

    body = "\n\n".join(paragraphs).strip()
    return ExtractedContent(
        text=body,
        metadata={"paragraph_count": len(paragraphs)},
        source_type="docx",
        source_locations=locations,
    )


__all__ = ["DocxExtractor"]
