"""Dataclasses + enums for the ingestion preprocessing cascade.

Pure data: no ORM, no session, no I/O. Mirrors the discipline in
``ai/retrieval/pgvector.py`` (which uses raw SQL specifically so the
retrieval layer never imports ``features.materials.models``).

The cascade runs between extraction and chunking and answers one question
per page: *is this page teachable content, noise, or something that needs
a different extractor?* It never mutates text in place — every rule emits
a ``Decision`` carrying the removed content and a reason code, so a wrong
threshold is a config change rather than a re-ingest.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Action(str, Enum):  # noqa: UP042 - StrEnum changes value coercion; match LLMRole
    """What a rule wants done with the unit it matched."""

    DROP_PAGE = "drop_page"
    STRIP_LINES = "strip_lines"
    TAG_ROLE = "tag_role"
    EXCLUDE_RETRIEVAL = "exclude_retrieval"
    ROUTE_OCR = "route_ocr"
    LINK_CANONICAL = "link_canonical"


class ReasonCode(str, Enum):  # noqa: UP042 - StrEnum changes value coercion; match LLMRole
    """Why a unit was acted on. Stored verbatim in the preprocess report."""

    BLANK_NO_CONTENT = "blank_no_content"
    IMAGE_ONLY_NEEDS_OCR = "image_only_needs_ocr"
    BROKEN_ENCODING = "broken_encoding"
    NEAR_EMPTY_DIVIDER = "near_empty_divider"
    RUNNING_HEADER = "running_header"
    RUNNING_FOOTER = "running_footer"
    PAGE_NUMBER = "page_number"
    TITLE_PAGE = "title_page"
    INSTRUCTOR_BLOCK = "instructor_block"
    TABLE_OF_CONTENTS = "table_of_contents"
    REFERENCES = "references"
    CLOSING_SLIDE = "closing_slide"
    EXACT_DUPLICATE = "exact_duplicate"
    NEAR_DUPLICATE_LEXICAL = "near_duplicate_lexical"
    LLM_ADJUDICATED = "llm_adjudicated"


# Roles understood by ``ai/chunking/_window._content_role`` and capped by
# ``ai/retrieval/role_filter``. ``divider`` and ``reference`` are new.
ROLE_BODY = "body"
ROLE_SUMMARY = "summary"
ROLE_REVIEW = "review"
ROLE_FRONT_MATTER = "front_matter"
ROLE_REFERENCE = "reference"
ROLE_DIVIDER = "divider"


@dataclass(frozen=True)
class LineFacts:
    """One extracted text line with the geometry needed by margin rules."""

    text: str
    y0: float
    y1: float
    font_size: float


@dataclass(frozen=True)
class PageFacts:
    """Per-page measurements captured at extraction time.

    Produced by ``ai/extraction/pdf.py`` from a single ``get_text("dict")``
    call. Every downstream geometric rule (margin bands, deck detection,
    image-only routing) reads these instead of re-opening the document.
    """

    page_number: int
    width: float = 0.0
    height: float = 0.0
    word_count: int = 0
    char_count: int = 0
    text_block_count: int = 0
    image_block_count: int = 0
    vector_count: int = 0
    text_area_ratio: float = 0.0
    image_area_ratio: float = 0.0
    median_font_size: float = 0.0
    replacement_char_ratio: float = 0.0
    lines: tuple[LineFacts, ...] = ()

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> PageFacts:
        """Rebuild from the JSON-shaped dict carried on ``ExtractedContent``."""
        lines = tuple(
            LineFacts(
                text=str(ln.get("text") or ""),
                y0=float(ln.get("y0") or 0.0),
                y1=float(ln.get("y1") or 0.0),
                font_size=float(ln.get("font_size") or 0.0),
            )
            for ln in (raw.get("lines") or [])
        )
        return cls(
            page_number=int(raw.get("page_number") or 0),
            width=float(raw.get("width") or 0.0),
            height=float(raw.get("height") or 0.0),
            word_count=int(raw.get("word_count") or 0),
            char_count=int(raw.get("char_count") or 0),
            text_block_count=int(raw.get("text_block_count") or 0),
            image_block_count=int(raw.get("image_block_count") or 0),
            vector_count=int(raw.get("vector_count") or 0),
            text_area_ratio=float(raw.get("text_area_ratio") or 0.0),
            image_area_ratio=float(raw.get("image_area_ratio") or 0.0),
            median_font_size=float(raw.get("median_font_size") or 0.0),
            replacement_char_ratio=float(raw.get("replacement_char_ratio") or 0.0),
            lines=lines,
        )


@dataclass(frozen=True)
class Decision:
    """One rule firing against one unit.

    ``content`` carries the exact text removed — this is what makes the
    preprocess report auditable and the drop reversible.
    """

    action: Action
    reason: ReasonCode
    rule_name: str
    page_number: int | None = None
    content: str = ""
    score: float | None = None
    stage: str = "deterministic"
    occurrences: int = 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.value,
            "reason": self.reason.value,
            "rule": self.rule_name,
            "page": self.page_number,
            "content": self.content[:500],
            "score": self.score,
            "stage": self.stage,
            "occurrences": self.occurrences,
        }


@dataclass
class PageUnit:
    """One page of the document, split so markers can never be destroyed.

    ``marker`` is the literal ``"[Page 7]"`` / ``"[Slide 3]"`` line that
    ``ai/chunking/token_aware._SECTION_MARKER_RE`` parses to recover page
    attribution. It is immutable: rules operate on ``body`` only, and
    ``join_pages`` re-emits markers verbatim. Page citation is the single
    thing this stage is not allowed to break.
    """

    marker: str
    page_number: int | None
    body: str
    facts: PageFacts | None = None
    dropped: bool = False
    role: str = ROLE_BODY
    retrieval_excluded: bool = False
    noise_flags: list[str] = field(default_factory=list)
    needs_ocr: bool = False
    ocr_reason: str | None = None
    # Deck mode only: consecutive slides sharing a normalized title get one
    # group id so a "Topic (cont.)" run retrieves and cites as a single unit.
    topic_group_id: int | None = None
    slide_title: str = ""
    decisions: list[Decision] = field(default_factory=list)

    def record(self, decision: Decision) -> None:
        self.decisions.append(decision)

    def flag(self, name: str) -> None:
        if name not in self.noise_flags:
            self.noise_flags.append(name)


@dataclass
class PreprocessReport:
    """Aggregate outcome, merged into ``version.extracted_metadata``."""

    enabled: bool = True
    page_count_in: int = 0
    page_count_out: int = 0
    pages_dropped: int = 0
    pages_ocr_routed: int = 0
    lines_stripped: int = 0
    chars_in: int = 0
    chars_out: int = 0
    hyphen_joins: int = 0
    is_deck: bool = False
    deck_score: float = 0.0
    llm_adjudicated: int = 0
    decisions: list[Decision] = field(default_factory=list)
    role_counts: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "page_count_in": self.page_count_in,
            "page_count_out": self.page_count_out,
            "pages_dropped": self.pages_dropped,
            "pages_ocr_routed": self.pages_ocr_routed,
            "lines_stripped": self.lines_stripped,
            "chars_in": self.chars_in,
            "chars_out": self.chars_out,
            "hyphen_joins": self.hyphen_joins,
            "is_deck": self.is_deck,
            "deck_score": round(self.deck_score, 2),
            "llm_adjudicated": self.llm_adjudicated,
            "role_counts": dict(self.role_counts),
            # Cap the decision log so a 500-page scan cannot bloat the JSONB
            # column; the counters above stay exact regardless.
            "decisions": [d.as_dict() for d in self.decisions[:200]],
            "decisions_truncated": max(0, len(self.decisions) - 200),
        }


__all__ = [
    "ROLE_BODY",
    "ROLE_DIVIDER",
    "ROLE_FRONT_MATTER",
    "ROLE_REFERENCE",
    "ROLE_REVIEW",
    "ROLE_SUMMARY",
    "Action",
    "Decision",
    "LineFacts",
    "PageFacts",
    "PageUnit",
    "PreprocessReport",
    "ReasonCode",
]
