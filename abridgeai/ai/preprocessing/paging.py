"""Page split/join — the layer that makes marker destruction impossible.

Page attribution in this codebase rides on ONE thing: the literal
``[Page 7]`` / ``[Slide 3]`` line that ``token_aware._SECTION_MARKER_RE``
re-parses into ``metadata['page']``. ``ExtractedContent.source_locations``
is populated by ``pdf.py`` but never read for PDFs, so if a cleaning regex
eats a marker the citation is gone silently — nothing asserts on it.

Rather than guard every rule with a negative lookahead, this module makes
the failure structurally impossible: the document is split into units whose
``marker`` is held separately from ``body``, every rule touches ``body``
only, and ``join_pages`` re-emits markers verbatim.

``test_preprocess_paging`` asserts ``text.count("[Page ")`` is identical
before and after the whole cascade, on every fixture.
"""

from __future__ import annotations

import re

from abridgeai.ai.preprocessing.base import PageFacts, PageUnit

# Deliberately mirrors ``token_aware._SECTION_MARKER_RE``'s page/slide arms.
# Kept as its own constant (rather than imported) so a change there is a
# visible test failure here instead of a silent behaviour shift.
_MARKER_RE = re.compile(r"^\[(?:Page|Slide) (\d+)\]$", re.MULTILINE)


def split_pages(text: str, facts_by_page: dict[int, PageFacts] | None = None) -> list[PageUnit]:
    """Split extracted text into marker-preserving page units.

    Formats without page markers (docx, html, text, code, transcripts) yield
    a single unit with an empty marker and ``page_number=None`` — the cascade
    still runs its text-level rules, just without geometry.
    """
    facts_by_page = facts_by_page or {}
    matches = list(_MARKER_RE.finditer(text))

    if not matches:
        return [PageUnit(marker="", page_number=None, body=text)]

    units: list[PageUnit] = []

    preamble = text[: matches[0].start()].strip()
    if preamble:
        units.append(PageUnit(marker="", page_number=None, body=preamble))

    for i, match in enumerate(matches):
        page_number = int(match.group(1))
        body_start = match.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        units.append(
            PageUnit(
                marker=match.group(0),
                page_number=page_number,
                body=text[body_start:body_end].strip("\n").rstrip(),
                facts=facts_by_page.get(page_number),
            )
        )
    return units


def join_pages(units: list[PageUnit]) -> str:
    """Re-emit surviving units, markers verbatim, in the extractor's format.

    Matches ``pdf.py``'s output shape exactly: ``"[Page N]\\n<body>"`` blocks
    joined by a blank line. A dropped unit contributes nothing — including
    its marker, which is correct: the page is gone, so a citation pointing
    at it would be a dangling reference.
    """
    parts: list[str] = []
    for unit in units:
        if unit.dropped:
            continue
        body = unit.body.strip()
        if not body:
            # A marker with no body would parse into a section whose body is
            # empty; ``token_aware._split_into_sections`` skips those anyway
            # (``if body:``), so emitting it is pure noise.
            continue
        parts.append(f"{unit.marker}\n{body}" if unit.marker else body)
    return "\n\n".join(parts).strip()


def marker_count(text: str) -> int:
    """Count page/slide markers — used by the integrity regression test."""
    return len(_MARKER_RE.findall(text))


__all__ = ["join_pages", "marker_count", "split_pages"]
