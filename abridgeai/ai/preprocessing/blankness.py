"""Empty / image-only / broken-encoding page classification.

The trap this module exists to avoid: "the page has no text" and "the page
has no content" are different statements, and conflating them deletes the
diagram slide that carries the whole point of the lecture. A slide holding
one large figure and a four-word title extracts as near-nothing, looks
identical to a blank page through a text-only lens, and is exactly the page
a student needs.

So emptiness is an *ordered* decision, not a threshold:

1. no words, no images, no vector drawings  -> genuinely blank, drop
2. almost no words but real image coverage  -> route to OCR, never drop
2b. a title's worth of words over a figure  -> route to OCR, never drop
3. high replacement-char ratio              -> broken text layer, route to OCR
4. a handful of words, no images            -> section divider, keep and tag

Rule 2b exists because rule 2 alone loses the most common diagram slide in a
lecture deck: the one that kept its running header. Nine words of title is
enough to clear a near-empty gate and take a full-page figure down with it.

Thresholds follow pymupdf4llm's page analysis: ``BAD_CHAR_THRESHOLD = 0.05``
for a broken text layer, and a no-content-of-any-kind early return that
declines to queue the page for OCR.
"""

from __future__ import annotations

from abridgeai.ai.preprocessing.base import (
    ROLE_DIVIDER,
    Action,
    Decision,
    PageFacts,
    PageUnit,
    ReasonCode,
)

# A slide-number-only page is 1-2 tokens and a section-divider title 2-5;
# the smallest genuine content slide observed is >= 6 words.
_NEAR_EMPTY_WORDS = 5
# A decorative logo covers <10% of a page; a real diagram covers >= 30%.
_IMAGE_AREA_MIN = 0.30
# pymupdf4llm's constant for "the text layer is garbage, go to OCR".
_BAD_CHAR_RATIO = 0.05
# Native PowerPoint charts, flowcharts and arrows are VECTOR art, not rasters:
# they carry no image block at all. A page of them is a diagram and must reach
# OCR. One or two paths is a decorative rule or a background box, so the floor
# sits at 3 — below it the page is treated as blank.
_VECTOR_DIAGRAM_MIN = 3

# Rule 2b (figure slide that kept its title). A slide whose text is just a
# running header plus a figure caption tops out around 12 words; the smallest
# genuine bullet slide observed runs 25. 20 splits them with room on both
# sides.
_FIGURE_PAGE_MAX_WORDS = 20
# Coverage floor for "this raster is the content, not decoration". Deliberately
# below ``_IMAGE_AREA_MIN``: rule 2b has already established the page carries
# no prose, so it can afford to trust a smaller figure than rule 2, which must
# also defend against a text page with one big background image.
_FIGURE_AREA_MIN = 0.12
# Vector equivalent. Higher than ``_VECTOR_DIAGRAM_MIN`` because a page in this
# band DOES have text, and text decorations (underlines, bullet rules, table
# borders) contribute paths — 3 would fire on ordinary slides.
_VECTOR_FIGURE_MIN = 8


def classify_emptiness(unit: PageUnit) -> None:
    """Apply the ordered emptiness decision to ``unit`` in place.

    Pages without ``facts`` (non-PDF sources: docx, html, transcripts) fall
    back to a text-only check — they have no geometry, and for those formats
    an empty unit really is empty because the extractor already skipped
    blank paragraphs.
    """
    # Idempotent: this runs twice — once before running-mark stripping to get
    # a clean frequency denominator, once after, because a page whose only
    # text was a running header IS an image-only page and the boilerplate was
    # masking it. Anything already decided keeps its first verdict.
    if unit.dropped or unit.needs_ocr or unit.role == ROLE_DIVIDER:
        return

    facts = unit.facts
    body = unit.body.strip()

    if facts is None:
        if not body:
            unit.dropped = True
            unit.record(
                Decision(
                    action=Action.DROP_PAGE,
                    reason=ReasonCode.BLANK_NO_CONTENT,
                    rule_name="blank_text_only",
                    page_number=unit.page_number,
                )
            )
        return

    # Count the LIVE body, not ``facts.word_count``: the latter is measured at
    # extraction time and still includes the running header/footer, which is
    # exactly enough text to push a diagram slide over the near-empty
    # threshold and hide it from the OCR tier.
    word_count = len(body.split())

    # 1. Nothing meaningful — no text, no raster, and at most a stray rule or
    #    background box worth of vector art.
    if (
        word_count == 0
        and facts.image_block_count == 0
        and facts.vector_count < _VECTOR_DIAGRAM_MIN
    ):
        unit.dropped = True
        unit.record(
            Decision(
                action=Action.DROP_PAGE,
                reason=ReasonCode.BLANK_NO_CONTENT,
                rule_name="blank_no_content",
                page_number=unit.page_number,
            )
        )
        return

    # 2. Image-only: looks empty to a text extractor, is not empty. Never drop.
    if word_count < _NEAR_EMPTY_WORDS and (
        facts.image_block_count >= 1
        or facts.image_area_ratio > _IMAGE_AREA_MIN
        or facts.vector_count >= _VECTOR_DIAGRAM_MIN
    ):
        unit.needs_ocr = True
        unit.ocr_reason = ReasonCode.IMAGE_ONLY_NEEDS_OCR.value
        unit.flag("image_only")
        unit.record(
            Decision(
                action=Action.ROUTE_OCR,
                reason=ReasonCode.IMAGE_ONLY_NEEDS_OCR,
                rule_name="image_only_page",
                page_number=unit.page_number,
                score=facts.image_area_ratio,
            )
        )
        return

    # 2b. Figure slide WITH a title. Rule 2 only sees pages that are near-empty
    #     of words, which is the wrong shape for the most common diagram slide
    #     in a lecture deck: a two-line running header ("A framework for
    #     decision support" / "The Steps of Decision Support") plus one
    #     full-bleed figure. That is 9-11 words — comfortably over the rule-2
    #     gate — so the page used to pass straight through as "text present"
    #     and the entire diagram was dropped on the floor, leaving a chunk that
    #     is nothing but its own heading.
    #
    #     The discriminator is COVERAGE, not word count: a decorative logo or a
    #     bullet glyph sits near 1-2% of the page, a real figure at 12%+. Body
    #     slides are excluded by the word gate (a genuine bullet slide runs
    #     25+ words), so this only ever fires on pages whose text is a title.
    if word_count < _FIGURE_PAGE_MAX_WORDS and (
        facts.image_area_ratio >= _FIGURE_AREA_MIN
        or facts.vector_count >= _VECTOR_FIGURE_MIN
    ):
        unit.needs_ocr = True
        unit.ocr_reason = ReasonCode.IMAGE_ONLY_NEEDS_OCR.value
        unit.flag("titled_figure")
        unit.record(
            Decision(
                action=Action.ROUTE_OCR,
                reason=ReasonCode.IMAGE_ONLY_NEEDS_OCR,
                rule_name="titled_figure_page",
                page_number=unit.page_number,
                score=facts.image_area_ratio,
            )
        )
        return

    # 3. Text layer present but garbled (bad embedded font encoding).
    if facts.replacement_char_ratio > _BAD_CHAR_RATIO:
        unit.needs_ocr = True
        unit.ocr_reason = ReasonCode.BROKEN_ENCODING.value
        unit.flag("broken_encoding")
        unit.record(
            Decision(
                action=Action.ROUTE_OCR,
                reason=ReasonCode.BROKEN_ENCODING,
                rule_name="broken_text_layer",
                page_number=unit.page_number,
                score=facts.replacement_char_ratio,
            )
        )
        return

    # 4. A few words, no imagery — a section divider. Keep it: ``_glue``'s
    #    tiny-window absorb will fold it into a neighbour, and the role tag
    #    stops it competing with body content in retrieval.
    if 0 < word_count < _NEAR_EMPTY_WORDS:
        unit.role = ROLE_DIVIDER
        unit.flag("divider")
        unit.record(
            Decision(
                action=Action.TAG_ROLE,
                reason=ReasonCode.NEAR_EMPTY_DIVIDER,
                rule_name="near_empty_divider",
                page_number=unit.page_number,
                content=body[:120],
            )
        )


def is_scanned_document(facts: list[PageFacts], *, threshold: float = 0.6) -> bool:
    """True when most pages carry imagery but no text layer.

    Used to decide whether the whole document is a scan (so OCR is the
    primary extraction path, not a per-page exception).
    """
    if not facts:
        return False
    image_only = sum(
        1
        for f in facts
        if f.word_count < _NEAR_EMPTY_WORDS
        and (f.image_block_count >= 1 or f.image_area_ratio > _IMAGE_AREA_MIN)
    )
    return image_only / len(facts) >= threshold


__all__ = ["classify_emptiness", "is_scanned_document"]
