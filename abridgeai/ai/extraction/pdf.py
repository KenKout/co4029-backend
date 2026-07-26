"""PDF extractor — pymupdf with a latin-1 fallback for binaries that ship with no parser.

Ports the PDF code path from ``backend/app/ai/haystack/components/extractors.py``.
Pages flow into the registry as one ``SourceLocation`` per page so downstream
chunkers can attribute every chunk to a specific page in the source PDF.

Beyond text, this extractor captures per-page geometry (``metadata["pages"]``)
in the SAME ``get_text("dict")`` pass: block counts, image and text area
coverage, median font size and the first/last lines with their vertical
position. ``ai/preprocessing`` needs all of it — margin bands for running
header/footer detection, area ratios to tell an image-only page from a blank
one, font size and page geometry for slide-deck detection — and re-deriving
it would mean opening the document a second time.

A page that carries imagery but no text layer IS emitted (with an empty
body). It used to be skipped outright, which is precisely backwards: that
page is the diagram slide, and dropping it at extraction meant the OCR tier
downstream never got the chance to recover it.
"""

from __future__ import annotations

import asyncio
import re
import statistics
from typing import Any, BinaryIO

from abridgeai.ai.extraction.base import ExtractedContent, SourceLocation
from abridgeai.ai.extraction.registry import register_extractor

_REPLACEMENT_CHARS = ("�", "\x00")


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


def _page_facts(
    page: Any,  # noqa: ANN401 -- a fitz.Page; pymupdf ships no type stubs
    page_num: int,
    page_dict: dict[str, Any],
) -> dict[str, Any]:
    """Measure one page from an already-parsed ``get_text("dict")`` payload."""
    width = float(page_dict.get("width") or 0.0)
    height = float(page_dict.get("height") or 0.0)
    area = width * height

    text_blocks = 0
    image_blocks = 0
    text_area = 0.0
    image_area = 0.0
    lines: list[dict[str, Any]] = []
    sizes: list[tuple[float, int]] = []
    char_count = 0
    bad_chars = 0

    for block in page_dict.get("blocks") or []:
        bbox = block.get("bbox") or (0.0, 0.0, 0.0, 0.0)
        block_area = max(0.0, (bbox[2] - bbox[0])) * max(0.0, (bbox[3] - bbox[1]))
        if block.get("type") == 1:
            image_blocks += 1
            image_area += block_area
            continue
        text_blocks += 1
        text_area += block_area
        for line in block.get("lines") or []:
            spans = line.get("spans") or []
            line_text = "".join(str(s.get("text") or "") for s in spans)
            if not line_text.strip():
                continue
            char_count += len(line_text)
            bad_chars += sum(line_text.count(ch) for ch in _REPLACEMENT_CHARS)
            for span in spans:
                span_text = str(span.get("text") or "")
                if span_text.strip():
                    sizes.append((float(span.get("size") or 0.0), len(span_text)))
            line_bbox = line.get("bbox") or bbox
            lines.append(
                {
                    "text": line_text.strip(),
                    "y0": float(line_bbox[1]),
                    "y1": float(line_bbox[3]),
                    "font_size": float(spans[0].get("size") or 0.0) if spans else 0.0,
                }
            )

    try:
        vector_count = len(page.get_drawings())
    except Exception:  # pragma: no cover - pymupdf version differences
        vector_count = 0

    # Weight font size by span length so a 40pt decorative letter does not
    # outvote a paragraph of 11pt body text.
    median_font = 0.0
    if sizes:
        expanded = [size for size, count in sizes for _ in range(min(count, 64))]
        median_font = float(statistics.median(expanded)) if expanded else 0.0

    word_count = sum(len(ln["text"].split()) for ln in lines)

    return {
        "page_number": page_num,
        "width": width,
        "height": height,
        "word_count": word_count,
        "char_count": char_count,
        "text_block_count": text_blocks,
        "image_block_count": image_blocks,
        "vector_count": vector_count,
        "text_area_ratio": min(1.0, text_area / area) if area else 0.0,
        "image_area_ratio": min(1.0, image_area / area) if area else 0.0,
        "median_font_size": median_font,
        "replacement_char_ratio": (bad_chars / char_count) if char_count else 0.0,
        # Only the outer lines matter to the margin-band rules; carrying every
        # line of a 500-page document through JSONB would be wasteful.
        "lines": lines[:4] + lines[-4:] if len(lines) > 8 else lines,
    }


def _extract_sync(raw: bytes) -> ExtractedContent:
    try:
        import fitz  # type: ignore[import-untyped,unused-ignore]
    except ImportError:
        return _fallback(raw)

    doc = fitz.open(stream=raw, filetype="pdf")
    try:
        page_count = doc.page_count
        doc_meta = dict(doc.metadata or {})
        parts: list[str] = []
        locations: list[SourceLocation] = []
        pages: list[dict[str, Any]] = []
        for page_num, page in enumerate(doc, start=1):
            page_dict = page.get_text("dict")
            facts = _page_facts(page, page_num, page_dict)
            pages.append(facts)

            block_texts: list[str] = []
            for block in page_dict.get("blocks") or []:
                if block.get("type") == 1:
                    continue
                block_lines = [
                    "".join(str(s.get("text") or "") for s in (line.get("spans") or []))
                    for line in (block.get("lines") or [])
                ]
                block_text = "\n".join(ln for ln in block_lines if ln.strip()).strip()
                if block_text:
                    block_texts.append(block_text)

            # Emit the page when it has text OR any non-text content. The
            # latter is the image-only page: empty body here, recovered by
            # the OCR tier in ``ai/preprocessing``.
            has_visual = facts["image_block_count"] > 0 or facts["vector_count"] > 0
            if block_texts:
                parts.append(f"[Page {page_num}]\n" + "\n".join(block_texts))
                locations.append(SourceLocation(page=page_num))
            elif has_visual:
                parts.append(f"[Page {page_num}]")
                locations.append(SourceLocation(page=page_num))
    finally:
        doc.close()

    text = "\n\n".join(parts).strip()
    return ExtractedContent(
        text=text,
        metadata={
            "page_count": page_count,
            "pages": pages,
            "producer": doc_meta.get("producer") or "",
            "creator": doc_meta.get("creator") or "",
        },
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
