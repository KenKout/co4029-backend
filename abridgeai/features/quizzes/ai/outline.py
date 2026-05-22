"""Lesson outline component (FR-3).

Walks the chunks of a lesson in order and groups them into ``OutlineSection``
records. Used by:

* ``GET /lessons/{id}/outline`` — to render the file structure for the
  teacher before they pick a generation mode (FR-4);
* ``_run_coverage_pipeline`` — to allocate per-section question budgets and
  load section-scoped chunks at generation time (FR-12).

No LLM calls. No embeddings. Pure SQL + Python so the outline is cheap to
recompute on every request without adding a second source of truth.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.features.materials.models import DocumentChunk

if TYPE_CHECKING:
    from collections.abc import Sequence


# Maximum number of slides we'll bundle into a single section when the deck
# has no explicit headings. 4 is a deliberate trade-off: large enough that
# a 30-slide deck gives ~8 sections (matches user expectation that a
# "section" should hold ~5 minutes of material), small enough that the
# coverage budget allocator still has meaningful per-section weight.
_SLIDE_GROUP_SIZE = 4

# How much of the first chunk to surface as a section preview in the API.
_PREVIEW_CHARS = 240


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slug(value: str) -> str:
    """Lowercase ascii-slug, used to derive stable section ids."""
    return _SLUG_RE.sub("-", (value or "").lower()).strip("-") or "section"


# --- Data classes -----------------------------------------------------------


@dataclass(frozen=True)
class OutlineSection:
    """One section in a lesson outline.

    ``id`` is deterministic — see ``_section_id`` — so a redeploy that re-runs
    ``build_lesson_outline`` against the same chunks produces the same ids.
    This is important for ``coverage_options.section_ids`` to remain stable
    across calls.
    """

    id: str
    title: str
    depth: int
    chunk_ids: list[UUID]
    char_count: int
    page_range: tuple[int, int]
    content_role: str          # body | summary | review | front_matter (mode of chunk roles)
    preview: str               # first 240 chars of the first chunk


@dataclass(frozen=True)
class LessonOutline:
    """Outline of a single lesson, in document order."""

    lesson_id: UUID
    title: str
    sections: list[OutlineSection] = field(default_factory=list)


# --- Public API -------------------------------------------------------------


async def build_lesson_outline(
    db: AsyncSession,
    lesson_ids: list[UUID],
    *,
    slides_per_section: int = _SLIDE_GROUP_SIZE,
    force_bundle: bool = False,
) -> list[LessonOutline]:
    """Load chunks for ``lesson_ids`` and group into one outline per lesson.

    ``slides_per_section`` controls how many consecutive chunks are bundled
    into a single ``OutlineSection`` when the slide-deck fallback fires.
    Default 4. Pass 1 for "one section per slide" granularity (use
    sparingly — coverage mode will then generate one question per slide
    which is many LLM calls).

    ``force_bundle`` (default ``False``) overrides semantic-aware grouping
    and forces ``_group_slide_deck`` regardless of whether the chunks have
    real or semantic headings. Useful when the user explicitly wants
    fixed-size bundling from the panel UI ("4 pages/section" knob) instead
    of one section per detected topic. When ``False`` (auto mode), the
    builder uses heading + semantic enrichment to draw boundaries — that
    can produce very many sections for slide-deck PDFs where every page
    has its own topic.

    Returns one ``LessonOutline`` per lesson that has at least one chunk.
    Lessons with no chunks are silently skipped — the caller (route layer)
    decides how to render an empty result.
    """
    if not lesson_ids:
        return []

    result = await db.execute(
        select(DocumentChunk)
        .where(DocumentChunk.lesson_id.in_(lesson_ids))
        .order_by(DocumentChunk.lesson_id, DocumentChunk.chunk_index)
    )
    rows: list[DocumentChunk] = list(result.scalars().all())
    if not rows:
        return []

    # Group rows by lesson_id while preserving order within each lesson.
    by_lesson: dict[UUID, list[DocumentChunk]] = {}
    for chunk in rows:
        by_lesson.setdefault(chunk.lesson_id, []).append(chunk)

    outlines: list[LessonOutline] = []
    for lesson_id in lesson_ids:
        lesson_chunks = by_lesson.get(lesson_id)
        if not lesson_chunks:
            continue
        sections = _group_sections(
            lesson_chunks,
            slides_per_section=slides_per_section,
            force_bundle=force_bundle,
        )
        outlines.append(
            LessonOutline(
                lesson_id=lesson_id,
                title=_lesson_title_from_chunks(lesson_chunks),
                sections=sections,
            )
        )
    return outlines


# --- Grouping algorithm -----------------------------------------------------


def _group_sections(
    chunks: Sequence[DocumentChunk],
    *,
    slides_per_section: int = _SLIDE_GROUP_SIZE,
    force_bundle: bool = False,
) -> list[OutlineSection]:
    """Walk chunks in order, open a new section on heading change.

    Boundary rules:
      (a) ``metadata.section`` (heading path) changes between consecutive
          chunks → new section.
      (b) No real headings present (raw slide deck — heading is empty or
          synthetic ``Page N`` / ``Slide N`` only) → fallback to grouping
          ``slides_per_section`` consecutive slides per section.

    When ``force_bundle=True`` always falls through to the slide-deck
    bundling path even if chunks have semantic-enriched headings. The
    panel UI uses this to give the user a fixed-size grouping knob ("4
    pages/section") regardless of how the chunker enriched per-page
    semantic titles.
    """
    if not chunks:
        return []

    if force_bundle:
        return _group_slide_deck(chunks, slides_per_section=slides_per_section)

    has_any_real_heading = any(_is_real_heading(c) for c in chunks)
    if not has_any_real_heading:
        return _group_slide_deck(chunks, slides_per_section=slides_per_section)

    groups: list[list[DocumentChunk]] = []
    current_section_path: str | None = None
    for chunk in chunks:
        path = _section_heading(chunk)
        if path != current_section_path or not groups:
            groups.append([chunk])
            current_section_path = path
        else:
            groups[-1].append(chunk)

    return [_section_from_group(g) for g in groups]


_SYNTHETIC_HEADING_RE = re.compile(r"^(Page|Slide)\s+\d+$", re.IGNORECASE)


def _is_real_heading(chunk: DocumentChunk) -> bool:
    """``True`` if the chunk has an actual prose heading, not just a
    ``Page N`` / ``Slide N`` marker the extractor synthesized.

    The PDF extractor inserts ``[Page N]`` markers between pages, which
    the chunker turns into a heading per chunk. If the deck has no real
    `# heading` content, every chunk gets a unique pseudo-heading like
    "Page 5" — and ``_group_sections`` would otherwise open one section
    per chunk. Treat those as heading-less so the slide-grouping
    fallback fires and we get sensible 4-page sections instead.
    """
    full_path = _section_heading(chunk)
    if not full_path:
        return False
    leaf = full_path.rsplit(" > ", 1)[-1].strip()
    if not leaf:
        return False
    return not _SYNTHETIC_HEADING_RE.match(leaf)


def _group_slide_deck(
    chunks: Sequence[DocumentChunk],
    *,
    slides_per_section: int = _SLIDE_GROUP_SIZE,
) -> list[OutlineSection]:
    """Fallback for heading-less decks: bundle ``slides_per_section`` slides per section."""
    groups: list[list[DocumentChunk]] = []
    for chunk in chunks:
        if not groups or len(groups[-1]) >= slides_per_section:
            groups.append([chunk])
        else:
            groups[-1].append(chunk)
    return [_section_from_group(g) for g in groups]


def _section_from_group(group: list[DocumentChunk]) -> OutlineSection:
    first = group[0]

    full_path = _section_heading(first)
    title = _short_title(full_path) or _slide_title_from_group(group)

    # Page range
    ranges = [r for r in (_chunk_page_range(c) for c in group) if r is not None]
    page_range = (
        (min(r[0] for r in ranges), max(r[1] for r in ranges)) if ranges else (0, 0)
    )

    # Content role: mode of chunk roles, ties go to "body".
    role_counter = Counter(_chunk_role(c) for c in group)
    if role_counter:
        most_common = role_counter.most_common()
        # Tie-break: prefer body if it ties with anything else.
        top_count = most_common[0][1]
        tied = [role for role, count in most_common if count == top_count]
        content_role = "body" if "body" in tied else tied[0]
    else:
        content_role = "body"

    char_count = sum(len(c.content or "") for c in group)
    preview = (first.content or "")[:_PREVIEW_CHARS].strip()

    depth = _heading_depth(full_path)
    section_id = _section_id(first, title, page_range[0])

    return OutlineSection(
        id=section_id,
        title=title or "Untitled section",
        depth=depth,
        chunk_ids=[c.id for c in group],
        char_count=char_count,
        page_range=page_range,
        content_role=content_role,
        preview=preview,
    )


# --- Helpers ----------------------------------------------------------------


def _section_heading(chunk: DocumentChunk) -> str:
    """Read the chunk's heading path from metadata.

    Prefers ``metadata.semantic.section_title`` (LLM-enriched per-chunk
    topic title) over the synthetic ``metadata.section`` ("Page N" /
    "Slide N") emitted by the page-marker extractor. The semantic title
    is the human-meaningful label set by the chunking pipeline; the
    synthetic one is the boundary marker we use only as fallback.
    """
    metadata = chunk.metadata_json or {}
    semantic = metadata.get("semantic") or {}
    semantic_title = semantic.get("section_title")
    if isinstance(semantic_title, str) and semantic_title.strip():
        return semantic_title.strip()
    return str(metadata.get("section") or "")


def _chunk_role(chunk: DocumentChunk) -> str:
    """Resolve a chunk's content role from metadata.

    Prefers ``metadata.semantic.content_role`` (LLM-enriched, knows
    "summary" / "review" / "front_matter" beyond what the rule
    classifier catches) over top-level ``metadata.content_role`` (rule
    classifier output, defaults to "body" for anything unrecognised).
    Without this priority, slide-deck PDFs end up with every chunk
    tagged "body" at the top level, defeating ``skip_summaries`` in
    coverage allocation and inflating the eligible section count.
    """
    metadata = chunk.metadata_json or {}
    semantic = metadata.get("semantic") or {}
    semantic_role = semantic.get("content_role")
    if isinstance(semantic_role, str) and semantic_role.strip():
        return semantic_role.strip()
    return str(metadata.get("content_role") or "body")


def _chunk_page(chunk: DocumentChunk) -> int | None:
    """Resolve a chunk's page number from metadata.

    Reads (in order):
      1. ``metadata.page`` — top-level integer page (current chunker)
      2. ``metadata.page_range[0]`` — first page of a range
      3. ``metadata.source_location.page`` / ``.slide`` — legacy schema

    Returns ``None`` if no page hint is available.
    """
    metadata = chunk.metadata_json or {}
    page = metadata.get("page")
    if isinstance(page, int):
        return page
    page_range = metadata.get("page_range")
    if isinstance(page_range, list) and page_range and isinstance(page_range[0], int):
        return page_range[0]
    loc = metadata.get("source_location") or {}
    legacy_page = loc.get("page")
    if isinstance(legacy_page, int):
        return legacy_page
    legacy_slide = loc.get("slide")
    if isinstance(legacy_slide, int):
        return legacy_slide
    return None


def _chunk_page_range(chunk: DocumentChunk) -> tuple[int, int] | None:
    """Resolve a chunk's full page range from metadata.

    Used by ``_section_from_group`` to compute group page ranges without
    losing multi-page chunks. Returns ``None`` if no page info exists.
    """
    metadata = chunk.metadata_json or {}
    page_range = metadata.get("page_range")
    if (
        isinstance(page_range, list)
        and len(page_range) == 2
        and all(isinstance(p, int) for p in page_range)
    ):
        return (page_range[0], page_range[1])
    page = _chunk_page(chunk)
    if page is not None:
        return (page, page)
    return None


def _short_title(full_path: str) -> str:
    """Strip the breadcrumb prefix from ``Course > Module > Lesson > Heading``."""
    if not full_path:
        return ""
    return full_path.rsplit(" > ", 1)[-1].strip()


def _heading_depth(full_path: str) -> int:
    """Depth = number of ``>`` separators after the lesson title.

    The chunker writes ``"<course> > <module> > <lesson> > <heading>"`` for
    a top-level heading, so we treat 4 segments as depth 1, 5 as depth 2,
    and so on. Sections without a heading default to depth 1.
    """
    if not full_path:
        return 1
    parts = [p for p in full_path.split(" > ") if p.strip()]
    if len(parts) <= 3:
        return 1
    return max(1, len(parts) - 3)


def _slide_title_from_group(group: list[DocumentChunk]) -> str:
    pages = [p for p in (_chunk_page(c) for c in group) if p is not None]
    if not pages:
        return ""
    if min(pages) == max(pages):
        return f"Slide {min(pages)}"
    return f"Slides {min(pages)}–{max(pages)}"


def _section_id(first_chunk: DocumentChunk, title: str, first_page: int) -> str:
    """Deterministic, stable section id.

    Format: ``sec_{lesson_id}_{slug}_{first_page}`` — encoding both the
    title slug and the first page so two different sections that happen to
    share a title (rare, but possible across re-ingests) are distinguished
    by their position in the deck.
    """
    lesson_id = getattr(first_chunk, "lesson_id", None)
    lesson_part = str(lesson_id)[:8] if lesson_id else "lesson"
    slug = _slug(title)[:48]
    return f"sec_{lesson_part}_{slug}_{first_page or 0}"


def _lesson_title_from_chunks(chunks: list[DocumentChunk]) -> str:
    """Derive a lesson title from chunk metadata (best-effort)."""
    for chunk in chunks:
        metadata = chunk.metadata_json or {}
        title = metadata.get("lesson_title")
        if isinstance(title, str) and title:
            return title
    return ""


__all__ = [
    "LessonOutline",
    "OutlineSection",
    "allocate_question_budget",
    "build_lesson_outline",
]


# ----------------------------------------------------------------------------
# Coverage budget allocator (FR-12)
# ----------------------------------------------------------------------------


def allocate_question_budget(
    outline: LessonOutline | list[LessonOutline],
    total: int,
    *,
    min_per_section: int = 1,
    max_per_section: int = 5,
    skip_summaries: bool = True,
    section_ids: list[str] | None = None,
) -> dict[str, int]:
    """Allocate ``total`` questions across the outline's body sections.

    Algorithm:
      1. Filter sections by ``section_ids`` (if provided), then drop
         non-body sections when ``skip_summaries=True``.
      2. Weight remaining sections by ``char_count`` and produce raw
         floats that sum to ``total``.
      3. Clamp each to ``[min_per_section, max_per_section]``.
      4. Reconcile the rounded sum back to exactly ``total`` by adding /
         removing one question at a time from the section with the
         largest raw-vs-clamped delta and remaining headroom.

    Raises ``ValueError`` when there isn't enough total budget to give
    every eligible section at least ``min_per_section`` questions.
    """
    sections = _flatten_outlines(outline)

    eligible: list[OutlineSection] = []
    for sec in sections:
        if section_ids is not None and sec.id not in section_ids:
            continue
        if skip_summaries and sec.content_role != "body":
            continue
        eligible.append(sec)

    if not eligible:
        return {}

    if total < len(eligible) * min_per_section:
        raise ValueError(
            f"not enough questions for full coverage: total={total} but "
            f"{len(eligible)} eligible sections require at least "
            f"{len(eligible) * min_per_section} questions; lower "
            f"min_per_section or raise total"
        )

    weights = [max(sec.char_count, 1) for sec in eligible]
    total_weight = sum(weights)
    raw = [total * w / total_weight for w in weights]

    clamped = [
        max(min_per_section, min(max_per_section, round(r))) for r in raw
    ]

    # Reconcile to exactly ``total``.
    while sum(clamped) < total:
        # Add one to the section with the largest (raw - clamped) gap and
        # headroom under max_per_section. If no section has headroom, give
        # up — this means max_per_section * len(eligible) < total, which
        # is also a configuration error but we don't raise here so a
        # tiny over-allocation just returns the clamped sum.
        deltas = [
            (raw[i] - clamped[i], i)
            for i in range(len(eligible))
            if clamped[i] < max_per_section
        ]
        if not deltas:
            break
        _, idx = max(deltas, key=lambda t: t[0])
        clamped[idx] += 1

    while sum(clamped) > total:
        deltas = [
            (clamped[i] - raw[i], i)
            for i in range(len(eligible))
            if clamped[i] > min_per_section
        ]
        if not deltas:
            break
        _, idx = max(deltas, key=lambda t: t[0])
        clamped[idx] -= 1

    return {sec.id: clamped[i] for i, sec in enumerate(eligible)}


def _flatten_outlines(
    outline: LessonOutline | list[LessonOutline],
) -> list[OutlineSection]:
    if isinstance(outline, LessonOutline):
        return list(outline.sections)
    sections: list[OutlineSection] = []
    for o in outline:
        sections.extend(o.sections)
    return sections
