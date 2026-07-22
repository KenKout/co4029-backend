"""XLSX extractor — openpyxl wrapping with sheet-level source locations.

Reads every worksheet in a workbook and emits one text block per sheet,
prefixed with a ``[Sheet: <name>]`` marker so downstream chunkers and
citation builders can attribute a chunk back to a specific tab. Rows are
rendered as tab-separated cell values; fully-empty rows are skipped.

One ``SourceLocation`` is recorded per sheet using ``page`` as the sheet
ordinal (1-indexed) — mirroring the paged treatment PDF/PPTX get — since
``SourceLocation`` has no dedicated sheet field and reusing ``page`` keeps
the citation layer uniform.

``read_only=True`` + ``data_only=True`` keeps memory bounded on large
workbooks and surfaces the last-computed value of formula cells rather than
the formula text itself.
"""

from __future__ import annotations

import asyncio
import io
from typing import BinaryIO

from abridgeai.ai.extraction.base import ExtractedContent, SourceLocation
from abridgeai.ai.extraction.registry import register_extractor

_XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
_XLS_MIME = "application/vnd.ms-excel"


def _read_source(source: BinaryIO | bytes | str) -> bytes:
    if isinstance(source, bytes):
        return source
    if isinstance(source, str):
        with open(source, "rb") as fh:
            return fh.read()
    return source.read()


def _stringify(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _extract_sync(raw: bytes) -> ExtractedContent:
    import openpyxl

    workbook = openpyxl.load_workbook(io.BytesIO(raw), read_only=True, data_only=True)
    try:
        parts: list[str] = []
        locations: list[SourceLocation] = []
        sheet_total = 0
        for sheet_index, sheet in enumerate(workbook.worksheets, start=1):
            sheet_total = sheet_index
            row_lines: list[str] = []
            for row in sheet.iter_rows(values_only=True):
                cells = [_stringify(cell) for cell in row]
                if not any(cells):
                    continue
                row_lines.append("\t".join(cells).rstrip("\t"))
            if row_lines:
                parts.append(f"[Sheet: {sheet.title}]\n" + "\n".join(row_lines))
                locations.append(SourceLocation(page=sheet_index))
    finally:
        workbook.close()

    body = "\n\n".join(parts).strip()
    return ExtractedContent(
        text=body,
        metadata={"sheet_count": sheet_total, "populated_sheet_count": len(locations)},
        source_type="xlsx",
        source_locations=locations,
    )


@register_extractor(_XLSX_MIME)
@register_extractor(_XLS_MIME)
class XlsxExtractor:
    async def extract(self, source: BinaryIO | bytes | str) -> ExtractedContent:
        raw = await asyncio.to_thread(_read_source, source)
        return await asyncio.to_thread(_extract_sync, raw)


__all__ = ["XlsxExtractor"]
