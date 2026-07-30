"""Page-role classification: cover, instructor block, TOC, references, closing.

This is the module that answers the actual complaint — the title page, the
"Instructor: Dr. X / Faculty of / Semester 1" block, the agenda slide and
the "Thank you! Questions?" slide are all currently indexed as teachable
body content and compete with the lecture for retrieval budget.

Two disciplines run through every rule here:

**Two independent signals, never one.** ``instructor`` appears in ordinary
prose ("the instructor will demonstrate"), ``Overview`` is a legitimate
content-slide title, and a page of numbered lines might be an exercise set
rather than a table of contents. Single-signal rules generate exactly the
false positives that make teachers distrust the whole feature.

**Tag, don't delete.** Everything here sets ``content_role`` — which
``ai/retrieval/role_filter.py`` already knows how to cap at 25% of the
candidate pool — rather than removing text. Only the two cases that can
never carry an assessable fact (a bare title page, a closing slide) also
set ``retrieval_excluded``.

Patterns are bilingual because this product ships Vietnamese course
material alongside English.
"""

from __future__ import annotations

import re

from abridgeai.ai.preprocessing.base import (
    ROLE_FRONT_MATTER,
    ROLE_REFERENCE,
    ROLE_SUMMARY,
    Action,
    Decision,
    PageUnit,
    ReasonCode,
)

_FRONT_MATTER_PAGE_LIMIT = 3
_FRONT_MATTER_MAX_WORDS = 120
_TITLE_PAGE_MAX_WORDS = 25
_CLOSING_TAIL_PAGES = 2
_CLOSING_MAX_WORDS = 12

_INSTRUCTOR_SIGNALS = (
    ("title", re.compile(r"\b(prof\.?|dr\.?|assoc\.?\s*prof|ths\.?|pgs\.?|giảng\s*viên)\b", re.I)),
    ("email", re.compile(r"@\S+\.\S+")),
    (
        "institution",
        re.compile(
            r"\b(faculty\s+of|department\s+of|university\s+of|school\s+of|"
            r"khoa\b|trường\s+đại\s+học|bộ\s+môn|đại\s+học)\b",
            re.I,
        ),
    ),
    (
        "term",
        re.compile(
            r"\b(semester\s*\d|academic\s+year|học\s*kỳ|năm\s*học|fall\s*\d{4}|spring\s*\d{4})\b",
            re.I,
        ),
    ),
    ("course_code", re.compile(r"\b[A-Z]{2,4}\s?\d{3,4}\b")),
    ("admin", re.compile(r"\b(office\s+hours|credits?\s*:\s*\d|tín\s*chỉ|mã\s*môn)\b", re.I)),
)

_TOC_HEADING_RE = re.compile(
    r"^\s*(table\s+of\s+contents|contents|agenda|outline|roadmap|"
    r"mục\s*lục|nội\s*dung\s*chính?)\b",
    re.I,
)
_TOC_LEADER_RE = re.compile(r"(\.{2,}\s*\d{1,3}|\s\d{1,3})\s*$")
_TOC_MIN_LEADER_LINES = 4
_TOC_SHORT_LINE_CHARS = 60
_TOC_SHORT_LINE_RATIO = 0.6

_REFERENCES_HEADING_RE = re.compile(
    r"^\s*(references|bibliography|works\s+cited|further\s+reading|sources|"
    r"tài\s*liệu\s*tham\s*khảo)\b",
    re.I,
)
_CITATION_SHAPE_RE = re.compile(r"\(\d{4}\)|\[\d+\]|doi:|https?://|et\s+al\.", re.I)
_CITATION_RATIO = 0.5

_CLOSING_RE = re.compile(
    r"^\s*(thank\s*you|thanks|questions\s*\??|q\s*&\s*a|any\s+questions|"
    r"the\s+end|cảm\s*ơn|xin\s*cảm\s*ơn|hỏi\s*(?:&|và)\s*đáp)\b",
    re.I,
)


def _first_line(body: str) -> str:
    for line in body.splitlines():
        if line.strip():
            return line.strip()
    return ""


def _body_lines(body: str) -> list[str]:
    return [ln.strip() for ln in body.splitlines() if ln.strip()]


def classify_page_role(unit: PageUnit, *, total_pages: int) -> None:
    """Assign ``unit.role`` from the boilerplate taxonomy, in priority order.

    References and closing slides are checked before front matter because a
    bibliography on page 2 of a short handout would otherwise be swallowed
    by the front-matter page gate.
    """
    body = unit.body.strip()
    if not body:
        return
    lines = _body_lines(body)
    first = _first_line(body)
    word_count = len(body.split())
    page = unit.page_number or 0

    if _is_references(first, lines):
        unit.role = ROLE_REFERENCE
        unit.retrieval_excluded = True
        unit.flag("references")
        unit.record(
            Decision(
                action=Action.EXCLUDE_RETRIEVAL,
                reason=ReasonCode.REFERENCES,
                rule_name="references_page",
                page_number=unit.page_number,
                content=first[:120],
            )
        )
        return

    if _is_closing(first, word_count, page, total_pages):
        unit.role = ROLE_FRONT_MATTER
        unit.retrieval_excluded = True
        unit.flag("closing_slide")
        unit.record(
            Decision(
                action=Action.EXCLUDE_RETRIEVAL,
                reason=ReasonCode.CLOSING_SLIDE,
                rule_name="closing_slide",
                page_number=unit.page_number,
                content=body[:120],
            )
        )
        return

    if _is_toc(first, lines):
        unit.role = ROLE_SUMMARY
        unit.flag("toc")
        unit.record(
            Decision(
                action=Action.TAG_ROLE,
                reason=ReasonCode.TABLE_OF_CONTENTS,
                rule_name="table_of_contents",
                page_number=unit.page_number,
                content=first[:120],
            )
        )
        return

    matched = _instructor_signals(body)
    if (
        page
        and page <= _FRONT_MATTER_PAGE_LIMIT
        and word_count <= _FRONT_MATTER_MAX_WORDS
        and len(matched) >= 2
    ):
        unit.role = ROLE_FRONT_MATTER
        unit.flag("instructor_block")
        # A bare title page carries no assessable fact at all; a fuller
        # syllabus block might (course outcomes), so only the former is
        # excluded outright.
        is_title_page = page == 1 and word_count <= _TITLE_PAGE_MAX_WORDS
        if is_title_page:
            unit.retrieval_excluded = True
        unit.record(
            Decision(
                action=Action.EXCLUDE_RETRIEVAL if is_title_page else Action.TAG_ROLE,
                reason=ReasonCode.TITLE_PAGE if is_title_page else ReasonCode.INSTRUCTOR_BLOCK,
                rule_name="front_matter_signals",
                page_number=unit.page_number,
                content=body[:200],
                score=float(len(matched)),
            )
        )


def _instructor_signals(body: str) -> list[str]:
    return [name for name, pattern in _INSTRUCTOR_SIGNALS if pattern.search(body)]


def _is_toc(first: str, lines: list[str]) -> bool:
    if not _TOC_HEADING_RE.match(first):
        return False
    rest = lines[1:]
    if not rest:
        return False
    leaders = sum(1 for ln in rest if _TOC_LEADER_RE.search(ln))
    if leaders >= _TOC_MIN_LEADER_LINES:
        return True
    short = sum(1 for ln in rest if len(ln) <= _TOC_SHORT_LINE_CHARS)
    return short / len(rest) >= _TOC_SHORT_LINE_RATIO


def _is_references(first: str, lines: list[str]) -> bool:
    if _REFERENCES_HEADING_RE.match(first):
        return True
    if len(lines) < 3:
        return False
    cited = sum(1 for ln in lines if _CITATION_SHAPE_RE.search(ln))
    return cited / len(lines) >= _CITATION_RATIO


def _is_closing(first: str, word_count: int, page: int, total_pages: int) -> bool:
    if not page or not total_pages:
        return False
    if page < total_pages - _CLOSING_TAIL_PAGES + 1:
        return False
    # The word gate is what stops this eating a real "Questions to consider"
    # slide carrying five discussion prompts.
    return word_count <= _CLOSING_MAX_WORDS and bool(_CLOSING_RE.match(first))


__all__ = ["classify_page_role"]
