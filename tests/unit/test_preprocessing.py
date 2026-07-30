"""Unit tests for ``ai/preprocessing`` — the ingestion noise-filtering cascade.

The load-bearing test in this file is ``test_markers_survive_full_cascade``:
page attribution rides entirely on the inline ``[Page N]`` marker, nothing
downstream asserts on it, so a cleaning regex that ate one would break every
citation in the product silently.

The second theme is over-filtering. Most tests here are NEGATIVE — a
"Questions to consider" slide is not a closing slide, an authored compound
is not a line-break hyphen, an image-only page is not a blank page. Those
are the cases where an over-eager filter deletes content a student is
assessed on.
"""

from __future__ import annotations

import pytest

from abridgeai.ai.chunking._window import window_chunks
from abridgeai.ai.extraction.base import ExtractedContent
from abridgeai.ai.preprocessing import PreprocessConfig, run_preprocessing
from abridgeai.ai.preprocessing.base import (
    ROLE_BODY,
    ROLE_DIVIDER,
    ROLE_FRONT_MATTER,
    ROLE_REFERENCE,
    ROLE_SUMMARY,
    PageFacts,
    PageUnit,
)
from abridgeai.ai.preprocessing.blankness import classify_emptiness
from abridgeai.ai.preprocessing.deck import normalize_slide_title
from abridgeai.ai.preprocessing.dedup import (
    find_duplicates,
    link_semantic_duplicates,
    simhash64,
)
from abridgeai.ai.preprocessing.normalize import dehyphenate, normalize_text
from abridgeai.ai.preprocessing.page_roles import classify_page_role
from abridgeai.ai.preprocessing.paging import join_pages, marker_count, split_pages
from abridgeai.ai.preprocessing.running_marks import (
    is_page_number,
    is_sentence_like,
    normalize_key,
    strip_running_marks,
)


def _pdf(text: str, pages: list[dict] | None = None) -> ExtractedContent:
    return ExtractedContent(
        text=text,
        metadata={"pages": pages or []},
        source_type="pdf",
        source_locations=[],
    )


def _unit(page: int, body: str, **facts: object) -> PageUnit:
    return PageUnit(
        marker=f"[Page {page}]",
        page_number=page,
        body=body,
        facts=PageFacts(page_number=page, **facts) if facts else None,
    )


# --------------------------------------------------------------------------
# Marker integrity — the non-negotiable invariant
# --------------------------------------------------------------------------


class TestMarkerIntegrity:
    async def test_markers_survive_full_cascade(self) -> None:
        text = "\n\n".join(
            f"[Page {n}]\nUniversity of Example\n"
            f"Real teaching content about topic {n} that a student must learn here.\n{n}"
            for n in range(1, 11)
        )
        content = _pdf(text)
        out, report = await run_preprocessing(content, config=PreprocessConfig())

        assert marker_count(out.text) == marker_count(text) == 10
        assert report.lines_stripped > 0, "expected the repeated header to be stripped"

    def test_split_join_round_trips_markers(self) -> None:
        text = "[Page 1]\nalpha\n\n[Page 2]\nbeta"
        units = split_pages(text)
        assert [u.marker for u in units] == ["[Page 1]", "[Page 2]"]
        assert join_pages(units) == text

    def test_page_number_line_is_not_confused_with_marker(self) -> None:
        """``[Page 2]`` is structure; a bare ``2`` in the body is a folio."""
        units = split_pages("[Page 1]\ncontent here\n\n[Page 2]\nmore content\n2")
        strip_running_marks(units)
        assert marker_count(join_pages(units)) == 2

    def test_dropped_page_removes_its_marker(self) -> None:
        units = [_unit(1, "kept"), _unit(2, "gone")]
        units[1].dropped = True
        assert marker_count(join_pages(units)) == 1


# --------------------------------------------------------------------------
# Unicode + de-hyphenation
# --------------------------------------------------------------------------


class TestNormalize:
    def test_nfc_not_nfkc_preserves_stem_notation(self) -> None:
        """NFKC would rewrite x² -> x2 and ½ -> 1⁄2, corrupting the material."""
        out = normalize_text("x² ½ ⅓ H₂O")
        assert "x²" in out
        assert "½" in out
        assert "H₂O" in out

    def test_ligatures_and_fullwidth_are_folded(self) -> None:
        out = normalize_text("ﬁve ﬂat oﬀer ｆｕｌｌ")
        assert out == "five flat offer full"

    def test_nbsp_becomes_space_and_runs_collapse(self) -> None:
        assert normalize_text("a\xa0b  c") == "a b c"

    def test_zero_width_joiner_is_deleted_not_spaced(self) -> None:
        """ZWNJ inside a word is invisible noise; the word must close up."""
        assert normalize_text("b‌c") == "bc"

    def test_dehyphenation_joins_a_real_line_break(self) -> None:
        out, joins = dehyphenate("The decision sup-\nport system")
        assert out == "The decision support system"
        assert joins == 1

    @pytest.mark.parametrize(
        "text",
        [
            "Anh-\nminh",  # capitalized head: likely a proper noun
            "ISO-\n9001",  # digits: an identifier
            "Co-\nOp",  # capitalized tail
        ],
    )
    def test_dehyphenation_declines_ambiguous_cases(self, text: str) -> None:
        out, joins = dehyphenate(text)
        assert joins == 0
        assert out == text

    def test_authored_compound_blocks_the_join(self) -> None:
        """The document's own vocabulary is a free classifier."""
        text = "decision-support matters. The decision-\nsupport system."
        out, joins = dehyphenate(text)
        assert joins == 0
        assert "decision-\nsupport" in out


# --------------------------------------------------------------------------
# Blankness — the image-only trap
# --------------------------------------------------------------------------


class TestBlankness:
    def test_truly_blank_page_is_dropped(self) -> None:
        unit = _unit(1, "", word_count=0, image_block_count=0, vector_count=0)
        classify_emptiness(unit)
        assert unit.dropped is True

    def test_image_only_page_is_routed_to_ocr_not_dropped(self) -> None:
        """The diagram slide: no text layer, all the meaning."""
        unit = _unit(1, "", word_count=0, image_block_count=1, image_area_ratio=0.7)
        classify_emptiness(unit)
        assert unit.dropped is False
        assert unit.needs_ocr is True

    def test_vector_diagram_page_is_routed_to_ocr(self) -> None:
        """Native PowerPoint charts are vector art with NO image block."""
        unit = _unit(1, "", word_count=0, image_block_count=0, vector_count=12)
        classify_emptiness(unit)
        assert unit.dropped is False
        assert unit.needs_ocr is True

    def test_single_background_rect_is_still_blank(self) -> None:
        unit = _unit(1, "", word_count=0, image_block_count=0, vector_count=1)
        classify_emptiness(unit)
        assert unit.dropped is True

    def test_broken_text_layer_routes_to_ocr(self) -> None:
        unit = _unit(1, "���", word_count=3, replacement_char_ratio=0.9)
        classify_emptiness(unit)
        assert unit.needs_ocr is True

    def test_few_words_no_imagery_is_a_divider(self) -> None:
        unit = _unit(1, "Part Two", word_count=2)
        classify_emptiness(unit)
        assert unit.dropped is False
        assert unit.role == ROLE_DIVIDER

    def test_titled_figure_reaches_ocr_without_waiting_for_stripping(self) -> None:
        """A diagram slide whose only text was a running header.

        This used to need two passes: the header pushed the page over the
        near-empty gate, so it read as "text present" until running-mark
        stripping removed the header and the second pass reclassified it. Rule
        2b settles it on the first pass off image/vector coverage instead —
        which matters because the header is not always boilerplate the
        stripper recognises. A real lecture deck carries per-section titles
        ("The Steps of Decision Support") that appear once, survive stripping,
        and used to take the whole figure down with them.
        """
        unit = _unit(
            1, "CS310 - Decision Support Systems", word_count=5, vector_count=8
        )
        classify_emptiness(unit)
        assert unit.needs_ocr is True
        assert "titled_figure" in unit.noise_flags

        # Idempotent: the second pass (post-stripping) keeps the first verdict.
        unit.body = ""
        classify_emptiness(unit)
        assert unit.needs_ocr is True

    def test_text_slide_with_a_small_logo_is_not_a_figure(self) -> None:
        """Rule 2b must not drag ordinary bullet slides into the OCR tier.

        Coverage is the discriminator: a decorative logo sits near 2% of the
        page, a real diagram at 12%+.
        """
        unit = _unit(
            1,
            "Simon's decision-making process is a continuum ranging from highly "
            "structured programmed decisions to highly unstructured ones",
            word_count=25,
            image_block_count=1,
            image_area_ratio=0.02,
            vector_count=4,
        )
        classify_emptiness(unit)
        assert unit.needs_ocr is False
        assert unit.dropped is False


# --------------------------------------------------------------------------
# Running headers / footers / page numbers
# --------------------------------------------------------------------------


class TestRunningMarks:
    def test_digit_normalization_collapses_folio_variants(self) -> None:
        assert normalize_key("Page 3 of 12") == normalize_key("Page 4 of 12")

    @pytest.mark.parametrize(
        "line", ["7", "- 7 -", "Page 3", "Page 3 of 12", "Trang 5", "p. 9", "3 / 20"]
    )
    def test_page_number_shapes(self, line: str) -> None:
        assert is_page_number(line, page_has_other_content=True)

    def test_lone_roman_kept_when_it_is_the_whole_page(self) -> None:
        """A slide whose only content is ``IV`` is a section divider."""
        assert is_page_number("IV", page_has_other_content=True)
        assert not is_page_number("IV", page_has_other_content=False)

    def test_sentence_guard_protects_prose(self) -> None:
        assert is_sentence_like(
            "A decision support system combines data and models for the manager."
        )
        assert not is_sentence_like("Faculty of Computer Science")

    def test_repeated_footer_stripped_but_body_kept(self) -> None:
        units = split_pages(
            "\n\n".join(
                f"[Page {n}]\nUniversity of Example\n"
                f"Body sentence {n} explaining the concept to the student clearly.\n"
                f"Page {n} of 6"
                for n in range(1, 7)
            )
        )
        strip_running_marks(units)
        for unit in units:
            if unit.page_number == 1:
                continue  # page 1 is deliberately exempt
            assert "University of Example" not in unit.body
            assert "Body sentence" in unit.body

    def test_rare_line_is_not_stripped(self) -> None:
        """Appearing on one page of six is not a running mark."""
        pages = [f"[Page {n}]\nBody sentence {n} for the student to read." for n in range(1, 7)]
        pages[2] += "\nA one-off note"
        units = split_pages("\n\n".join(pages))
        strip_running_marks(units)
        assert "A one-off note" in units[2].body


# --------------------------------------------------------------------------
# Page roles — heavy on negative cases
# --------------------------------------------------------------------------


class TestPageRoles:
    def test_cover_page_with_instructor_block(self) -> None:
        unit = _unit(
            1,
            "Introduction to DSS\nInstructor: Dr. Jane Smith\n"
            "Faculty of Computer Science\nSemester 1",
        )
        classify_page_role(unit, total_pages=20)
        assert unit.role == ROLE_FRONT_MATTER
        assert unit.retrieval_excluded is True

    def test_single_signal_does_not_classify(self) -> None:
        """'the instructor will demonstrate' is ordinary prose."""
        unit = _unit(2, "In this lesson the instructor will demonstrate a star schema.")
        classify_page_role(unit, total_pages=20)
        assert unit.role == ROLE_BODY

    def test_vietnamese_instructor_block(self) -> None:
        unit = _unit(1, "Hệ hỗ trợ quyết định\nGiảng viên: TS. Nguyễn Văn A\nKhoa Công nghệ Thông tin\nHọc kỳ 1")
        classify_page_role(unit, total_pages=20)
        assert unit.role == ROLE_FRONT_MATTER

    def test_table_of_contents(self) -> None:
        unit = _unit(2, "Table of Contents\n1. Overview ... 3\n2. Schemas ... 12\n3. OLAP ... 25\n4. Mining ... 40")
        classify_page_role(unit, total_pages=20)
        assert unit.role == ROLE_SUMMARY
        assert "toc" in unit.noise_flags

    def test_references_page(self) -> None:
        unit = _unit(19, "References\nInmon, W. (2005). Building the Data Warehouse.")
        classify_page_role(unit, total_pages=20)
        assert unit.role == ROLE_REFERENCE
        assert unit.retrieval_excluded is True

    def test_closing_slide(self) -> None:
        unit = _unit(20, "Thank you! Questions?")
        classify_page_role(unit, total_pages=20)
        assert unit.retrieval_excluded is True

    def test_questions_to_consider_is_not_a_closing_slide(self) -> None:
        """The word gate is what stops this eating a real discussion slide."""
        unit = _unit(
            20,
            "Questions to consider\n- How does OLAP differ from OLTP?\n"
            "- What are the tradeoffs of a star schema?\n"
            "- When is denormalization justified?",
        )
        classify_page_role(unit, total_pages=20)
        assert unit.role == ROLE_BODY
        assert unit.retrieval_excluded is False

    def test_instructor_gate_does_not_reach_page_ten(self) -> None:
        unit = _unit(10, "Instructor: Dr. Jane Smith\nFaculty of Computer Science")
        classify_page_role(unit, total_pages=20)
        assert unit.role == ROLE_BODY


# --------------------------------------------------------------------------
# Deck handling + dedup
# --------------------------------------------------------------------------


class TestDeckAndDedup:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("Star Schema (cont.)", "Star Schema"),
            ("Star Schema (continued)", "Star Schema"),
            ("Normalization - Part 2", "Normalization"),
            ("Indexing (2/3)", "Indexing"),
            ("Joins", "Joins"),
        ],
    )
    def test_continuation_titles_normalize(self, raw: str, expected: str) -> None:
        assert normalize_slide_title(raw) == expected

    def test_exact_duplicates_link_to_first_occurrence(self) -> None:
        texts = ["alpha beta", "gamma", "alpha beta"]
        assert find_duplicates(texts)[2] == (0, "exact")

    def test_formatting_only_variant_is_linked(self) -> None:
        """What the LEXICAL pass is actually for: punctuation/case restyling."""
        original = (
            "A star schema places a central fact table at the middle and joins it "
            "to denormalized dimension tables for fast analytic queries."
        )
        restyled = original.lower().replace(".", "").replace(",", "")
        result = find_duplicates([original, restyled])
        assert result.get(1, (None, None))[0] == 0

    def test_reworded_recap_is_not_caught_lexically(self) -> None:
        """Documents the limit that motivates the semantic pass.

        A recap slide that swaps one word sits ~15 bits away in SimHash space,
        far beyond the Hamming<=3 gate. Loosening the gate to reach it would
        start merging genuinely distinct slides, so the reworded case is left
        to ``link_semantic_duplicates``, which runs where embeddings exist.
        """
        original = (
            "A star schema places a central fact table at the middle and joins it "
            "to denormalized dimension tables for fast analytic queries."
        )
        recap = original.replace("middle", "centre")
        assert find_duplicates([original, recap]) == {}

    def test_unrelated_text_is_not_a_duplicate(self) -> None:
        texts = [
            "A star schema places a central fact table at the middle of the model.",
            "Backpropagation computes gradients by applying the chain rule backwards.",
        ]
        assert find_duplicates(texts) == {}

    def test_simhash_is_stable(self) -> None:
        assert simhash64("hello world example text") == simhash64("hello world example text")

    def test_semantic_pass_links_reworded_near_duplicate(self) -> None:
        """What the lexical pass cannot reach, the embedding pass can."""
        original = [1.0, 0.0, 0.0]
        reworded = [0.98, 0.199, 0.0]  # cosine ~0.98
        assert link_semantic_duplicates([original, reworded]) == {1: 0}

    def test_semantic_pass_leaves_distinct_topics_alone(self) -> None:
        """0.94 is deliberately high: one course is all on-topic already."""
        assert link_semantic_duplicates([[1.0, 0.0, 0.0], [0.7, 0.7, 0.0]]) == {}

    def test_semantic_pass_ignores_zero_vectors(self) -> None:
        assert link_semantic_duplicates([[0.0, 0.0], [0.0, 0.0]]) == {}

    def test_semantic_chain_points_at_the_first_occurrence(self) -> None:
        """Three restatements collapse onto one canonical, not a chain."""
        vectors = [[1.0, 0.0], [0.999, 0.045], [0.998, 0.063]]
        assert link_semantic_duplicates(vectors) == {1: 0, 2: 0}


# --------------------------------------------------------------------------
# The regression the whole feature was blocked on
# --------------------------------------------------------------------------


class TestHeadingHintRegression:
    """``_window`` matched its role regexes against the literal ``"Page 7"``.

    ``token_aware`` sets ``section`` to the page marker for paged formats, so
    three of the four classifier paths could never fire for a PDF — and
    ``ai/retrieval/role_filter`` stayed inert because every chunk was tagged
    ``body``.
    """

    @pytest.mark.parametrize(
        ("body", "expected"),
        [
            ("Table of Contents\n1. Overview\n2. Schemas", ROLE_FRONT_MATTER),
            ("Copyright 2026 Acme Press. All rights reserved.", ROLE_FRONT_MATTER),
            ("Summary\n- point a\n- point b", ROLE_SUMMARY),
            ("Review Questions\n1. What is OLAP?", "review"),
        ],
    )
    def test_paged_content_classifies_from_its_first_line(
        self, body: str, expected: str
    ) -> None:
        content = _pdf(f"[Page 8]\n{body}")
        chunks = window_chunks(content, max_tokens=800, overlap_tokens=0)
        assert chunks[0].metadata["content_role"] == expected

    def test_body_content_stays_body(self) -> None:
        content = _pdf(
            "[Page 8]\nWhat is a DSS?\nA decision support system combines data, "
            "models and an interface to help managers analyse problems."
        )
        chunks = window_chunks(content, max_tokens=800, overlap_tokens=0)
        assert chunks[0].metadata["content_role"] == ROLE_BODY

    async def test_page_role_map_overrides_the_heuristic(self) -> None:
        """A page-level verdict outranks the per-chunk regex."""
        content = ExtractedContent(
            text="[Page 3]\nSome ordinary looking body text about schemas.",
            metadata={"page_roles": {"3": {"role": ROLE_REFERENCE, "retrieval_excluded": True}}},
            source_type="pdf",
            source_locations=[],
        )
        chunks = window_chunks(content, max_tokens=800, overlap_tokens=0)
        assert chunks[0].metadata["content_role"] == ROLE_REFERENCE
        assert chunks[0].metadata["retrieval_excluded"] is True


# --------------------------------------------------------------------------
# Cascade-level behaviour
# --------------------------------------------------------------------------


class TestCascade:
    async def test_non_paged_source_types_are_skipped(self) -> None:
        content = ExtractedContent(
            text="a transcript line", metadata={}, source_type="audio", source_locations=[]
        )
        out, report = await run_preprocessing(content)
        assert out is content
        assert report.enabled is False

    async def test_disabled_config_is_a_no_op(self) -> None:
        content = _pdf("[Page 1]\nhello")
        out, report = await run_preprocessing(content, config=PreprocessConfig(enabled=False))
        assert out is content
        assert report.enabled is False

    async def test_ocr_failure_leaves_page_untouched(self) -> None:
        """Fail open: a vision outage must never delete content."""

        class BoomOcr:
            async def ocr_page(self, page_number: int) -> str | None:
                raise RuntimeError("gateway down")

        content = _pdf(
            "[Page 1]\n",
            pages=[{"page_number": 1, "word_count": 0, "image_block_count": 1,
                    "image_area_ratio": 0.8, "width": 960, "height": 540}],
        )
        out, report = await run_preprocessing(
            content, config=PreprocessConfig(llm_adjudication=False), ocr=BoomOcr()
        )
        assert report.pages_ocr_routed == 0
        assert out.text is not None  # did not raise

    async def test_report_lands_in_metadata(self) -> None:
        content = _pdf("[Page 1]\nSome content on the page for the student.")
        out, _ = await run_preprocessing(content, config=PreprocessConfig(llm_adjudication=False))
        assert "preprocess" in out.metadata
        assert "page_roles" in out.metadata
