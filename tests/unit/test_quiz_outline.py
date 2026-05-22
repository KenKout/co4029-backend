"""Unit tests for the FR-3 lesson outline component (T5.14 Phase 3).

Covers the pure-Python parts of
:mod:`abridgeai.features.quizzes.ai.outline` — section grouping, slide
deck fallback, slug derivation, and the coverage budget allocator. The
``build_lesson_outline`` SQL path is exercised separately by the
quiz authoring integration suite.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast
from uuid import UUID

import pytest

from abridgeai.features.materials.models import DocumentChunk
from abridgeai.features.quizzes.ai.outline import (
    LessonOutline,
    OutlineSection,
    _group_sections,
    _section_id,
    _slug,
    allocate_question_budget,
)

_LESSON_ID = UUID("11111111-1111-1111-1111-111111111111")


def _chunk(
    *,
    chunk_id: UUID,
    content: str,
    section: str = "",
    page: int | None = None,
    role: str = "body",
    semantic_title: str | None = None,
    semantic_role: str | None = None,
    page_at_top_level: bool = False,
) -> DocumentChunk:
    """Build a duck-typed chunk that satisfies the outline component.

    The component only reads ``id``, ``lesson_id``, ``content``, and
    ``metadata_json``, so a SimpleNamespace cast to DocumentChunk is
    enough — no need to spin up the full SQLAlchemy ORM for pure
    grouping logic. Cast keeps the typed signatures in
    ``outline.py`` strict without forcing real DB rows in unit tests.

    ``semantic_title`` writes ``metadata.semantic.section_title`` to
    simulate the LLM-enriched per-chunk topic title.
    ``semantic_role`` writes ``metadata.semantic.content_role`` to
    simulate LLM-enriched role classification (e.g. "summary" /
    "review" / "front_matter") that overrides the rule classifier.
    ``page_at_top_level=True`` writes ``metadata.page`` (current
    chunker layout); ``False`` writes the legacy
    ``metadata.source_location.page`` instead.
    """
    metadata: dict[str, object] = {"section": section, "content_role": role}
    if page is not None:
        if page_at_top_level:
            metadata["page"] = page
            metadata["page_range"] = [page, page]
        else:
            metadata["source_location"] = {"page": page}
    semantic: dict[str, object] = {}
    if semantic_title is not None:
        semantic["section_title"] = semantic_title
    if semantic_role is not None:
        semantic["content_role"] = semantic_role
    if semantic:
        metadata["semantic"] = semantic
    fake = SimpleNamespace(
        id=chunk_id,
        lesson_id=_LESSON_ID,
        content=content,
        metadata_json=metadata,
    )
    return cast(DocumentChunk, fake)


# --- _slug -------------------------------------------------------------------


def test_slug_lowercases_and_dasherizes() -> None:
    assert _slug("Hello World!") == "hello-world"
    assert _slug("  Photosynthesis 101  ") == "photosynthesis-101"
    assert _slug("___---___") == "section"  # all-junk falls back
    assert _slug("") == "section"


# --- _section_id -------------------------------------------------------------


def test_section_id_is_deterministic() -> None:
    chunk = _chunk(chunk_id=UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"), content="x")
    a = _section_id(chunk, "Intro to Photosynthesis", 3)
    b = _section_id(chunk, "Intro to Photosynthesis", 3)
    assert a == b
    assert a.startswith("sec_11111111_")
    assert "intro-to-photosynthesis" in a
    assert a.endswith("_3")


# --- _group_sections — heading-based grouping -------------------------------


def test_group_sections_splits_on_heading_change() -> None:
    chunks = [
        _chunk(chunk_id=UUID(int=i), content=f"chunk {i}", section=section, page=i)
        for i, section in enumerate(
            [
                "Course > Module > Lesson > Intro",
                "Course > Module > Lesson > Intro",
                "Course > Module > Lesson > Body",
                "Course > Module > Lesson > Body",
                "Course > Module > Lesson > Summary",
            ],
            start=1,
        )
    ]
    chunks[-1].metadata_json["content_role"] = "summary"

    sections = _group_sections(chunks)
    assert len(sections) == 3
    titles = [s.title for s in sections]
    assert titles == ["Intro", "Body", "Summary"]
    # First section bundles 2 chunks
    assert len(sections[0].chunk_ids) == 2
    # Summary section keeps its role
    assert sections[2].content_role == "summary"


def test_group_sections_returns_empty_for_empty_input() -> None:
    assert _group_sections([]) == []


# --- _group_sections — slide-deck fallback ----------------------------------


def test_group_sections_falls_back_to_slide_groups_when_no_real_heading() -> None:
    chunks = [
        _chunk(
            chunk_id=UUID(int=i),
            content=f"slide {i}",
            section=f"Course > Module > Lesson > Page {i}",
            page=i,
        )
        for i in range(1, 11)
    ]

    sections = _group_sections(chunks, slides_per_section=4)
    # 10 slides / 4 per section = ceil(10/4) = 3 sections
    assert len(sections) == 3
    chunk_counts = [len(s.chunk_ids) for s in sections]
    assert chunk_counts == [4, 4, 2]
    # Slide-deck fallback uses the first chunk's leaf heading as the
    # title — for a synthetic ``Page N`` deck this is exactly "Page 1",
    # which is fine: the section_id keeps the page suffix so SectionPicker
    # can still distinguish them.
    assert sections[0].title == "Page 1"
    assert sections[1].page_range[0] == 5


def test_group_sections_picks_up_page_range() -> None:
    chunks = [
        _chunk(
            chunk_id=UUID(int=i),
            content=f"x{i}",
            section="Course > Mod > Less > Same",
            page=i,
        )
        for i in (3, 4, 5)
    ]
    sections = _group_sections(chunks)
    assert len(sections) == 1
    assert sections[0].page_range == (3, 5)


def test_group_sections_uses_semantic_title_over_synthetic_page_marker() -> None:
    """Real-world chunker layout: every chunk has ``metadata.section`` =
    ``Page N`` (synthetic boundary) but also enriched
    ``metadata.semantic.section_title`` (per-page LLM topic title).

    Builder must:
    1. Use the semantic title for the section heading
    2. NOT fall back to the slide-deck 4-chunk grouping (each semantic
       title is unique → one section per chunk)
    3. Read pages from top-level ``metadata.page``
    """
    chunks = [
        _chunk(
            chunk_id=UUID(int=i),
            content=f"page-{i}-content",
            section=f"Page {i}",
            page=i,
            page_at_top_level=True,
            semantic_title=title,
        )
        for i, title in enumerate(
            ["Introduction to DSS", "Definition of Data", "Knowledge Discovery"],
            start=1,
        )
    ]
    sections = _group_sections(chunks)
    assert len(sections) == 3
    titles = [s.title for s in sections]
    assert titles == [
        "Introduction to DSS",
        "Definition of Data",
        "Knowledge Discovery",
    ]
    assert [s.page_range for s in sections] == [(1, 1), (2, 2), (3, 3)]


def test_group_sections_prefers_semantic_role_over_top_level() -> None:
    """LLM enrichment writes ``metadata.semantic.content_role`` to override
    the rule classifier (which often defaults everything to ``body``).
    Builder must read the semantic role so coverage allocation can skip
    summary/review/front_matter sections correctly.
    """
    chunks = [
        _chunk(
            chunk_id=UUID(int=1),
            content="cover slide",
            section="Page 1",
            page=1,
            page_at_top_level=True,
            role="body",  # rule classifier got it wrong
            semantic_title="Course Cover",
            semantic_role="front_matter",  # LLM corrected it
        ),
        _chunk(
            chunk_id=UUID(int=2),
            content="real concept",
            section="Page 2",
            page=2,
            page_at_top_level=True,
            role="body",
            semantic_title="Star Schema Basics",
            semantic_role="body",
        ),
        _chunk(
            chunk_id=UUID(int=3),
            content="recap slide",
            section="Page 3",
            page=3,
            page_at_top_level=True,
            role="body",  # rule classifier missed the summary signal
            semantic_title="Recap of Concepts",
            semantic_role="summary",  # LLM caught it
        ),
    ]
    sections = _group_sections(chunks)
    assert len(sections) == 3
    roles = [s.content_role for s in sections]
    assert roles == ["front_matter", "body", "summary"]


def test_chunk_page_reads_top_level_metadata() -> None:
    """Top-level ``metadata.page`` (current chunker) must take precedence
    over the legacy ``metadata.source_location.page`` path."""
    chunks = [
        _chunk(
            chunk_id=UUID(int=42),
            content="pagey",
            section="Course > Mod > Less > Same Heading",
            page=7,
            page_at_top_level=True,
            semantic_title="Some Topic",
        ),
    ]
    sections = _group_sections(chunks)
    assert len(sections) == 1
    assert sections[0].page_range == (7, 7)


def test_group_sections_force_bundles_when_size_knob_set() -> None:
    """When ``force_bundle=True``, builder bundles by ``slides_per_section``
    regardless of semantic titles — gives user the legacy 'gọn' grouping
    for slide-deck PDFs.
    """
    chunks = [
        _chunk(
            chunk_id=UUID(int=i),
            content=f"page-{i}",
            section=f"Page {i}",
            page=i,
            page_at_top_level=True,
            semantic_title=f"Topic {i}",  # all unique → would normally be one section per chunk
        )
        for i in range(1, 9)
    ]
    sections = _group_sections(chunks, slides_per_section=4, force_bundle=True)
    assert len(sections) == 2
    assert [len(s.chunk_ids) for s in sections] == [4, 4]
    # Title comes from first chunk's semantic title (slide-deck path)
    assert sections[0].title == "Topic 1"
    assert sections[1].title == "Topic 5"
    assert sections[0].page_range == (1, 4)
    assert sections[1].page_range == (5, 8)


def test_group_sections_auto_mode_yields_per_topic_sections() -> None:
    """Counterpart to the force_bundle test — ``force_bundle=False`` (default)
    keeps semantic-aware boundary detection so each unique semantic title
    becomes its own section."""
    chunks = [
        _chunk(
            chunk_id=UUID(int=i),
            content=f"page-{i}",
            section=f"Page {i}",
            page=i,
            page_at_top_level=True,
            semantic_title=f"Topic {i}",
        )
        for i in range(1, 9)
    ]
    sections = _group_sections(chunks, slides_per_section=4)  # force_bundle defaults to False
    assert len(sections) == 8
    assert [s.title for s in sections] == [f"Topic {i}" for i in range(1, 9)]


# --- allocate_question_budget -----------------------------------------------


def _make_outline(
    *sections: tuple[str, int, str],
) -> LessonOutline:
    """Build a LessonOutline from ``(id, char_count, role)`` triples."""
    sec_objs = [
        OutlineSection(
            id=sid,
            title=sid,
            depth=1,
            chunk_ids=[],
            char_count=cc,
            page_range=(0, 0),
            content_role=role,
            preview="",
        )
        for sid, cc, role in sections
    ]
    return LessonOutline(lesson_id=_LESSON_ID, title="Lesson", sections=sec_objs)


def test_budget_skips_summaries_by_default() -> None:
    outline = _make_outline(
        ("a", 1000, "body"),
        ("b", 1000, "summary"),
        ("c", 1000, "body"),
    )
    budget = allocate_question_budget(outline, total=6)
    assert "b" not in budget
    assert sum(budget.values()) == 6


def test_budget_includes_summaries_when_flag_off() -> None:
    outline = _make_outline(
        ("a", 1000, "body"),
        ("b", 1000, "summary"),
    )
    budget = allocate_question_budget(outline, total=6, skip_summaries=False)
    assert set(budget) == {"a", "b"}
    assert sum(budget.values()) == 6


def test_budget_filters_by_section_ids() -> None:
    outline = _make_outline(
        ("a", 1000, "body"),
        ("b", 1000, "body"),
        ("c", 1000, "body"),
    )
    budget = allocate_question_budget(outline, total=4, section_ids=["a", "c"])
    assert set(budget) == {"a", "c"}
    assert sum(budget.values()) == 4


def test_budget_clamps_to_max_per_section() -> None:
    outline = _make_outline(
        ("a", 9999, "body"),  # huge so weight wants the whole budget
        ("b", 1, "body"),
    )
    budget = allocate_question_budget(outline, total=10, max_per_section=5)
    assert budget["a"] == 5  # clamped
    assert budget["b"] >= 1


def test_budget_raises_when_total_below_min_floor() -> None:
    outline = _make_outline(
        ("a", 1000, "body"),
        ("b", 1000, "body"),
        ("c", 1000, "body"),
    )
    with pytest.raises(ValueError, match="not enough questions"):
        allocate_question_budget(outline, total=2, min_per_section=1)


def test_budget_returns_empty_when_no_eligible_sections() -> None:
    outline = _make_outline(
        ("a", 1000, "summary"),
        ("b", 1000, "summary"),
    )
    # All summaries, default skip_summaries=True → nothing eligible
    assert allocate_question_budget(outline, total=4) == {}


def test_budget_handles_list_of_outlines() -> None:
    outlines = [
        _make_outline(("a", 1000, "body")),
        _make_outline(("b", 1000, "body")),
    ]
    budget = allocate_question_budget(outlines, total=4)
    assert set(budget) == {"a", "b"}
    assert sum(budget.values()) == 4


def test_budget_reconciles_to_exact_total() -> None:
    outline = _make_outline(
        ("a", 333, "body"),
        ("b", 333, "body"),
        ("c", 334, "body"),
    )
    # 7 doesn't divide cleanly across 3 sections — reconciler must
    # add/remove ones to land exactly on 7.
    budget = allocate_question_budget(outline, total=7)
    assert sum(budget.values()) == 7
