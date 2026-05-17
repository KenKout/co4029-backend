"""Source-code extractor.

Treats source files as plain UTF-8 text but tags ``source_type="code"`` so
downstream chunkers can apply language-aware splitting. The set of registered
MIMEs covers the 23 extensions the legacy ``_language_from_filename`` helper
recognised plus their canonical IANA / IETF MIME strings.
"""

from __future__ import annotations

import asyncio
from typing import BinaryIO

from abridgeai.ai.extraction.base import ExtractedContent, SourceLocation
from abridgeai.ai.extraction.registry import register_extractor

CODE_MIMES: tuple[str, ...] = (
    "text/x-python",
    "application/javascript",
    "text/javascript",
    "application/typescript",
    "text/x-typescript",
    "text/x-java-source",
    "text/x-c",
    "text/x-c++",
    "text/x-go",
    "text/x-rust",
    "text/x-ruby",
    "application/x-php",
    "text/x-csharp",
    "text/x-swift",
    "text/x-kotlin",
    "application/sql",
    "text/x-shellscript",
    "application/json",
    "application/x-yaml",
    "text/css",
    "text/x-scala",
    "text/x-perl",
    "text/x-r",
)


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


class CodeExtractor:
    async def extract(self, source: BinaryIO | bytes | str) -> ExtractedContent:
        raw = await asyncio.to_thread(_read_source, source)
        text = await asyncio.to_thread(_decode, raw)
        body = text.rstrip()
        line_count = body.count("\n") + 1 if body else 0
        locations = [SourceLocation(line_start=1, line_end=line_count)] if line_count else []
        return ExtractedContent(
            text=body,
            metadata={"line_count": line_count},
            source_type="code",
            source_locations=locations,
        )


for _mime in CODE_MIMES:
    register_extractor(_mime)(CodeExtractor)


__all__ = ["CODE_MIMES", "CodeExtractor"]
