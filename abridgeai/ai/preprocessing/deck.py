"""Slide-deck detection and slide-aware grouping.

A PowerPoint deck exported to PDF arrives as ``application/pdf`` and is
currently chunked as if it were prose. That is wrong in both directions:
each slide is a semantic unit but far too small to be its own embedding
target, while ``_glue``'s tiny-window absorb silently swallows a 40-token
slide into its neighbour and destroys slide-level citation.

Detection is a soft score rather than a single rule, because no single
signal is reliable — a Beamer deck has no PowerPoint producer string, and a
two-column A4 handout can be word-sparse. Requiring 3.0 points from six
independent signals keeps false positives low, and detection changes
*grouping only, never deletion*, so a false positive costs retrieval
granularity rather than content.

Measured over pages 2+: page 1 is frequently a differently-sized cover.
"""

from __future__ import annotations

import re
import statistics

from abridgeai.ai.preprocessing.base import PageUnit

_DECK_SCORE_GATE = 3.0

_PRODUCER_RE = re.compile(
    r"powerpoint|keynote|impress|google\s*slides|beamer|canva|prezi",
    re.IGNORECASE,
)

# PowerPoint 4:3 = 10x7.5in = 720x540pt; 16:9 = 13.333x7.5in = 960x540pt;
# Google Slides 16:9 = 10x5.63in = 720x405pt.
_DECK_GEOMETRIES = ((720.0, 540.0), (960.0, 540.0), (720.0, 405.0))
_GEOMETRY_TOLERANCE = 2.0

_ASPECT_MIN = 1.2
_WORDS_PER_PAGE_MAX = 60
_FONT_SIZE_MIN = 14.0
_TEXT_COVERAGE_MAX = 0.35

# "Topic (cont.)", "Topic - Part 2", "Topic (2/3)", "Topic 2 of 3".
_CONTINUATION_RES = (
    re.compile(r"\s*\((?:cont(?:inued)?\.?|ctd\.?)\)\s*$", re.IGNORECASE),
    re.compile(r"\s*[-–—]\s*(?:part|pt\.?)\s*\d+\s*$", re.IGNORECASE),
    re.compile(r"\s*\(\d+\s*/\s*\d+\)\s*$"),
    re.compile(r"\s*\d+\s*of\s*\d+\s*$", re.IGNORECASE),
)


def detect_deck(units: list[PageUnit], doc_metadata: dict[str, object]) -> tuple[bool, float]:
    """Return ``(is_deck, score)``. Requires >= 3.0 points across 6 signals."""
    live = [u for u in units if not u.dropped and u.facts is not None]
    measured = [u for u in live if (u.page_number or 0) >= 2] or live
    if not measured:
        return False, 0.0

    score = 0.0

    producer = " ".join(
        str(doc_metadata.get(key) or "") for key in ("producer", "creator", "pdf_producer")
    )
    if _PRODUCER_RE.search(producer):
        score += 2.0

    widths = [u.facts.width for u in measured if u.facts]
    heights = [u.facts.height for u in measured if u.facts]
    if widths and heights:
        w, h = statistics.median(widths), statistics.median(heights)
        exact = any(
            abs(w - gw) <= _GEOMETRY_TOLERANCE and abs(h - gh) <= _GEOMETRY_TOLERANCE
            for gw, gh in _DECK_GEOMETRIES
        )
        if exact:
            score += 2.0
        elif h > 0 and (w / h) > _ASPECT_MIN:
            score += 1.0

    words = [u.facts.word_count for u in measured if u.facts]
    if words and statistics.median(words) < _WORDS_PER_PAGE_MAX:
        score += 1.0

    fonts = [u.facts.median_font_size for u in measured if u.facts and u.facts.median_font_size]
    if fonts and statistics.median(fonts) >= _FONT_SIZE_MIN:
        score += 1.0

    coverage = [u.facts.text_area_ratio for u in measured if u.facts]
    if coverage and statistics.median(coverage) < _TEXT_COVERAGE_MAX:
        score += 1.0

    return score >= _DECK_SCORE_GATE, score


def normalize_slide_title(title: str) -> str:
    """Strip continuation suffixes so ``Topic`` and ``Topic (cont.)`` match.

    The normalized form is what goes into the contextual-retrieval prefix —
    ``(cont.)`` in an embedding is pure noise.
    """
    out = title.strip()
    for pattern in _CONTINUATION_RES:
        out = pattern.sub("", out)
    return out.strip(" -–—:").strip()


def slide_title_of(unit: PageUnit) -> str:
    """First non-empty body line, which on a slide is the title."""
    for line in unit.body.splitlines():
        if line.strip():
            return line.strip()
    return ""


def assign_topic_groups(units: list[PageUnit]) -> None:
    """Give consecutive slides sharing a normalized title one group id.

    Lets a three-slide "Normalization (cont.)" run be retrieved and cited as
    one topic instead of three orphan fragments.
    """
    group_id = 0
    previous = None
    for unit in units:
        if unit.dropped:
            continue
        title = normalize_slide_title(slide_title_of(unit))
        if not title or title != previous:
            group_id += 1
            previous = title or None
        unit.topic_group_id = group_id
        unit.slide_title = title


__all__ = [
    "assign_topic_groups",
    "detect_deck",
    "normalize_slide_title",
    "slide_title_of",
]
