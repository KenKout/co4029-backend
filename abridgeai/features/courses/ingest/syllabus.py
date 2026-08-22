"""Parse an HCMUT-style ``ĐỀ CƯƠNG HỌC PHẦN`` / Course Syllabus PDF.

The template is fixed and bilingual: every label, section heading and
learning outcome carries a Vietnamese line followed by its English
counterpart. The importer asks the manager which language the course
should be created in and this module returns only that side.

What we pull out (everything else in the document is ignored):

``title``
    ``- Tên học phần: <vi>`` / ``Course title: <en>`` (§1.1).
``course_code``
    ``Mã học phần (Course ID): CO2017`` — used to build a stable slug.
``required_hours``
    The ``Tổng cộng (Total)`` row of the *course format* table in §1.1
    (e.g. ``126.5``). The evaluation-ratio table further down has a
    second ``Tổng cộng (Total)`` row holding ``100%``; we skip it by
    requiring the value to parse as a bare number.
``description``
    §2 ``Mô tả học phần (Course description)``.
``outcomes``
    §4.2 ``Chuẩn đầu ra học phần (Course learning outcomes)`` — the
    ``L.O.x.y`` codes give the parent/child structure directly.

Language separation
-------------------
There is no markup telling the two languages apart, so we use the one
signal that is always present: Vietnamese-specific letters. A run of
lines is Vietnamese up to and including the last line carrying a
Vietnamese letter; everything after it is English. §2 sometimes glues
the two halves onto one line (``… Xử lý song song The major contents
include:``), so the boundary line gets a second pass that splits it at
the first capitalised word whose remainder is pure ASCII — see
:func:`_split_mixed_line`.

Everything here is a pure function over ``bytes``: no DB, no storage,
no settings. :func:`parse_syllabus_pdf` raises
:class:`SyllabusParseError` with a human-readable reason when the
document is not a syllabus we recognise, and that reason is what the
manager is shown (and notified with) on a failed import.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import Literal

import fitz

SyllabusLanguage = Literal["vi", "en"]

# Vietnamese-only letters. Latin-1 accents (é, à, ô…) appear in English
# loanwords too, so we key off the letters that only Vietnamese uses:
# horned vowels, đ, and the tone marks stacked on them.
_VIETNAMESE_CHARS = frozenset("ăâđêôơưĂÂĐÊÔƠƯ")
_VIETNAMESE_COMBINING = frozenset("̣̀́̃̉")  # ̀ ́ ̃ ̉ ̣

# Header/footer noise. Page numbers and the print timestamp are matched by
# shape rather than by content so this is not tied to one university's
# letterhead; anything else repeated across most pages is dropped by
# frequency (see _strip_boilerplate).
_PAGE_NUMBER_RE = re.compile(r"^\s*\d+\s*/\s*\d+\s*$")
_TIMESTAMP_RE = re.compile(r"^\s*\d{1,2}:\d{2},\s*\d{1,2}/\d{1,2}/\d{4}\s*$")

_TITLE_VI_RE = re.compile(r"T[êe]n\s+h[ọo]c\s+ph[ầa]n\s*:\s*(.+)", re.IGNORECASE)
_TITLE_EN_RE = re.compile(r"Course\s+title\s*:\s*(.+)", re.IGNORECASE)
_COURSE_CODE_RE = re.compile(r"M[ãa]\s+h[ọo]c\s+ph[ầa]n[^:]*:\s*([A-Za-z0-9._-]+)")

# "Tổng cộng (Total)" — either half may be the one that survives extraction.
_TOTAL_ROW_RE = re.compile(r"^\s*(?:T[ổo]ng\s+c[ộo]ng|Total)\b", re.IGNORECASE)
_BARE_NUMBER_RE = re.compile(r"^\s*(\d+(?:[.,]\d+)?)\s*$")

_DESCRIPTION_START_RE = re.compile(r"^\s*2\.\s*(?:M[ôo]\s+t[ảa]|Course\s+description)")
_DESCRIPTION_END_RE = re.compile(r"^\s*3\.\s+\S")
_OUTCOMES_START_RE = re.compile(r"^\s*4\.2\.?\s+\S")
_OUTCOMES_END_RE = re.compile(r"^\s*5\.\s*\S")

# "L.O.1.2 - text", tolerating the LO / L.O / L.O. spelling variants and
# both hyphen flavours the template mixes.
_OUTCOME_RE = re.compile(r"^\s*L\.?\s*O\.?\s*(\d+(?:\.\d+)*)\s*[-–—:]\s*(.*)$")

_MAX_TITLE_LEN = 255
_MAX_SLUG_LEN = 100


class SyllabusParseError(ValueError):
    """The upload is not a syllabus we can read.

    The message is user-facing: it is returned to the manager and copied
    into the failure notification, so it must say what was missing, not
    which regex failed.
    """


@dataclass(slots=True)
class ParsedOutcome:
    """One ``L.O.x.y`` row, still keyed by its *source* code.

    ``code`` is what the PDF says (``"1.3"``). ``parent_code`` is the
    code with the last segment dropped, or ``None`` at the top level.
    The importer turns these into ``course_learning_outcomes`` rows;
    note that the stored code is re-derived from sibling order, so a
    gap in the source numbering does not survive (the importer reports
    that as a warning).
    """

    code: str
    text: str
    parent_code: str | None


@dataclass(slots=True)
class ParsedSyllabus:
    """Everything the importer needs to build a draft course."""

    language: SyllabusLanguage
    title: str
    course_code: str | None
    description: str | None
    required_hours: float | None
    outcomes: list[ParsedOutcome]
    warnings: list[str] = field(default_factory=list)

    @property
    def estimated_minutes(self) -> int | None:
        """``required_hours`` in the unit ``courses.estimated_minutes`` stores."""
        if self.required_hours is None:
            return None
        return round(self.required_hours * 60)

    def suggested_slug(self) -> str:
        """``co2017-operating-systems`` — course code first so it stays stable.

        The importer still has to resolve collisions against the org's
        existing slugs; this is only the preferred form.
        """
        parts = [p for p in (self.course_code, self.title) if p]
        return _slugify(" ".join(parts))[:_MAX_SLUG_LEN] or "imported-course"


def parse_syllabus_pdf(data: bytes, language: SyllabusLanguage) -> ParsedSyllabus:
    """Extract course fields from a syllabus PDF in the requested language.

    Thin wrapper: pull a text layer out of the PDF, then hand the pages to
    :func:`parse_syllabus_pages`. Raises :class:`SyllabusParseError` when
    the file is not a PDF, holds no extractable text (a scan — we do not
    OCR here), or is missing the course title, which is the one field a
    course cannot be created without.
    """
    return parse_syllabus_pages(_extract_pages(data), language)


def parse_syllabus_pages(pages: list[str], language: SyllabusLanguage) -> ParsedSyllabus:
    """Parse already-extracted page text — the whole parser minus PyMuPDF.

    Split out from :func:`parse_syllabus_pdf` so the field-level rules can
    be exercised on real Vietnamese text without round-tripping it through
    a generated PDF: the fonts PyMuPDF can synthesize do not cover
    Vietnamese, so a test fixture built that way would test the font, not
    the parser.
    """
    lines = _strip_boilerplate(pages)
    if not any(line.strip() for line in lines):
        raise SyllabusParseError(
            "no_text_in_pdf: the file contains no extractable text. Scanned or "
            "image-only syllabi are not supported — upload the original PDF."
        )

    warnings: list[str] = []
    title = _parse_title(lines, language, warnings)
    course_code = _first_match(lines, _COURSE_CODE_RE)
    required_hours = _parse_required_hours(lines, warnings)
    description = _parse_description(lines, language, warnings)
    outcomes = _parse_outcomes(lines, language, warnings)

    return ParsedSyllabus(
        language=language,
        title=title,
        course_code=course_code,
        description=description,
        required_hours=required_hours,
        outcomes=outcomes,
        warnings=warnings,
    )


# --------------------------------------------------------------------------
# text extraction
# --------------------------------------------------------------------------


def _extract_pages(data: bytes) -> list[str]:
    if not data:
        raise SyllabusParseError("empty_file: the uploaded file is empty.")
    try:
        with fitz.open(stream=data, filetype="pdf") as doc:
            return [page.get_text() for page in doc]
    except SyllabusParseError:
        raise
    except Exception as exc:  # noqa: BLE001 — any PyMuPDF failure means "not a PDF we can read"
        raise SyllabusParseError(
            "unreadable_pdf: the file could not be opened as a PDF. Make sure it "
            "is not password-protected or corrupted."
        ) from exc


def _strip_boilerplate(pages: list[str]) -> list[str]:
    """Flatten pages to lines, dropping running headers/footers.

    A line that shows up on most pages is letterhead, not content. That
    is frequency-based rather than a hardcoded list of the university's
    address lines, so a differently-branded syllabus on the same
    template still parses. Page numbers and the browser print timestamp
    differ per page, so those are matched by shape instead.
    """
    if not pages:
        return []

    per_page = [[line.rstrip() for line in page.splitlines()] for page in pages]
    repeats: Counter[str] = Counter()
    for page_lines in per_page:
        for line in {line.strip() for line in page_lines if line.strip()}:
            repeats[line] += 1

    # On a 6-page document that is 4 pages; on a 2-page one it is both,
    # which is why the floor is 3 — two pages sharing a line is not proof
    # of letterhead.
    threshold = max(3, int(len(pages) * 0.6))
    boilerplate = {line for line, count in repeats.items() if count >= threshold}

    out: list[str] = []
    for page_lines in per_page:
        for line in page_lines:
            stripped = line.strip()
            if stripped in boilerplate:
                continue
            if _PAGE_NUMBER_RE.match(line) or _TIMESTAMP_RE.match(line):
                continue
            out.append(line)
    return out


# --------------------------------------------------------------------------
# language classification
# --------------------------------------------------------------------------


def _has_vietnamese(text: str) -> bool:
    """Whether ``text`` carries a letter only Vietnamese uses.

    Decomposing first means both the precomposed (``ộ``) and combining
    (``o`` + U+0323) encodings of the same letter are caught — PDF text
    extraction produces either depending on the producer.
    """
    decomposed = unicodedata.normalize("NFD", text)
    return any(ch in _VIETNAMESE_CHARS or ch in _VIETNAMESE_COMBINING for ch in decomposed)


def _split_mixed_line(line: str) -> tuple[str, str] | None:
    """Split ``"… Xử lý song song The major contents include:"`` in two.

    Returns ``(vietnamese_part, english_part)`` when the line plausibly
    carries both halves, else ``None``. The English part must start at a
    capitalised word, contain no Vietnamese letters through to the end
    of the line, and be at least three words long — that last rule is
    what stops a trailing proper noun (``… tập lệnh MIPS``) from being
    mistaken for the start of the English half.
    """
    for match in re.finditer(r"\s+(?=[A-Z])", line):
        head, tail = line[: match.start()], line[match.end() :]
        if not head.strip() or _has_vietnamese(tail) or len(tail.split()) < 3:
            continue
        return head.strip(), tail.strip()
    return None


def _select_language(lines: list[str], language: SyllabusLanguage) -> list[str]:
    """Keep only the requested language's half of a bilingual block.

    The block is Vietnamese up to and including the last line holding a
    Vietnamese letter, English after it. A boundary line carrying both
    is split via :func:`_split_mixed_line`.
    """
    content = [line for line in lines if line.strip()]
    if not content:
        return []

    last_vi = -1
    for idx, line in enumerate(content):
        if _has_vietnamese(line):
            last_vi = idx

    if last_vi < 0:
        # Monolingual English document.
        return [] if language == "vi" else content
    if last_vi == len(content) - 1:
        # Monolingual Vietnamese document (no English half to split off).
        return content if language == "vi" else []

    vi_part = content[: last_vi + 1]
    en_part = content[last_vi + 1 :]
    if (split := _split_mixed_line(content[last_vi])) is not None:
        vi_head, en_head = split
        vi_part = [*content[:last_vi], vi_head]
        en_part = [en_head, *en_part]

    return vi_part if language == "vi" else en_part


# --------------------------------------------------------------------------
# field parsers
# --------------------------------------------------------------------------


def _first_match(lines: list[str], pattern: re.Pattern[str]) -> str | None:
    for line in lines:
        if (match := pattern.search(line)) is not None:
            return match.group(1).strip() or None
    return None


def _parse_title(lines: list[str], language: SyllabusLanguage, warnings: list[str]) -> str:
    """The course title in the requested language.

    Falls back to the other language rather than failing the whole
    import — a course with the wrong-language title is fixable in the
    draft; no course at all is not. Only a document with neither label
    is rejected.
    """
    vi = _first_match(lines, _TITLE_VI_RE)
    en = _first_match(lines, _TITLE_EN_RE)
    wanted, other = (vi, en) if language == "vi" else (en, vi)

    if wanted:
        return wanted[:_MAX_TITLE_LEN]
    if other:
        warnings.append(
            f"title_language_fallback: no {language.upper()} course title was found; "
            "used the other language's title instead."
        )
        return other[:_MAX_TITLE_LEN]
    raise SyllabusParseError(
        "missing_course_title: could not find a 'Tên học phần' / 'Course title' "
        "line. This does not look like a course syllabus."
    )


def _parse_required_hours(lines: list[str], warnings: list[str]) -> float | None:
    """The ``Tổng cộng (Total)`` hours from the §1.1 course-format table.

    PDF extraction emits the table cell-by-cell, so the number lands on
    a line of its own somewhere after the label. We scan the next few
    lines for a bare number and take the first hit — the evaluation
    table's own ``Tổng cộng`` row holds ``100%``, which never matches.
    """
    for idx, line in enumerate(lines):
        if not _TOTAL_ROW_RE.match(line):
            continue
        for candidate in lines[idx + 1 : idx + 5]:
            if (match := _BARE_NUMBER_RE.match(candidate)) is not None:
                return float(match.group(1).replace(",", "."))
    warnings.append(
        "missing_required_hours: no 'Tổng cộng (Total)' hours row was found; "
        "the course was created without an estimated duration."
    )
    return None


def _section(lines: list[str], start: re.Pattern[str], end: re.Pattern[str]) -> list[str] | None:
    """Lines strictly between a section heading and the next one."""
    start_idx: int | None = None
    for idx, line in enumerate(lines):
        if start_idx is None:
            if start.match(line):
                start_idx = idx + 1
            continue
        if end.match(line):
            return lines[start_idx:idx]
    return None if start_idx is None else lines[start_idx:]


def _parse_description(
    lines: list[str], language: SyllabusLanguage, warnings: list[str]
) -> str | None:
    block = _section(lines, _DESCRIPTION_START_RE, _DESCRIPTION_END_RE)
    if not block:
        warnings.append(
            "missing_description: section 2 'Mô tả học phần (Course description)' "
            "was not found; the course was created without a description."
        )
        return None

    selected = _select_language(block, language)
    if not selected:
        warnings.append(
            f"description_language_fallback: the description has no {language.upper()} "
            "text; kept the whole section instead."
        )
        selected = [line for line in block if line.strip()]
    return "\n".join(selected).strip() or None


def _parse_outcomes(
    lines: list[str], language: SyllabusLanguage, warnings: list[str]
) -> list[ParsedOutcome]:
    """§4.2 learning outcomes, one per ``L.O.x.y`` code.

    Each outcome runs from its code line to the next one, wrapping over
    however many lines the layout needed. Within that blob the English
    half is the trailing parenthesised group; the Vietnamese half is
    everything before it.
    """
    block = _section(lines, _OUTCOMES_START_RE, _OUTCOMES_END_RE)
    if block is None:
        warnings.append(
            "missing_outcomes: section 4.2 'Chuẩn đầu ra học phần (Course learning "
            "outcomes)' was not found; no learning outcomes were imported."
        )
        return []

    raw: list[tuple[str, list[str]]] = []
    for line in block:
        if (match := _OUTCOME_RE.match(line)) is not None:
            raw.append((match.group(1), [match.group(2)]))
        elif raw and line.strip():
            raw[-1][1].append(line.strip())

    outcomes: list[ParsedOutcome] = []
    seen: set[str] = set()
    for code, chunks in raw:
        text = _pick_outcome_language(" ".join(c.strip() for c in chunks if c.strip()), language)
        if not text:
            warnings.append(
                f"empty_outcome: L.O.{code} had no {language.upper()} text and was skipped."
            )
            continue
        if code in seen:
            warnings.append(f"duplicate_outcome: L.O.{code} appears twice; kept the first.")
            continue
        seen.add(code)
        parent = code.rsplit(".", 1)[0] if "." in code else None
        outcomes.append(ParsedOutcome(code=code, text=text, parent_code=parent))

    if not outcomes:
        warnings.append("no_outcomes_parsed: section 4.2 was found but held no 'L.O.x' rows.")
    _warn_on_numbering_gaps(outcomes, warnings)
    return outcomes


def _pick_outcome_language(blob: str, language: SyllabusLanguage) -> str:
    """Split ``"<vi text> (<en text>)"`` and return the requested half.

    The English half is the *last* parenthesised group that closes at
    the end of the blob, and it only counts as English if it carries no
    Vietnamese letters — otherwise it is Vietnamese text that merely
    happens to end in parentheses (``… mô phỏng (ngôn ngữ C)``), and the
    row is treated as Vietnamese-only.
    """
    blob = blob.strip()
    if not blob:
        return ""

    english = ""
    vietnamese = blob
    if blob.endswith(")"):
        depth = 0
        for idx in range(len(blob) - 1, -1, -1):
            if blob[idx] == ")":
                depth += 1
            elif blob[idx] == "(":
                depth -= 1
                if depth == 0:
                    candidate = blob[idx + 1 : -1].strip()
                    if candidate and not _has_vietnamese(candidate):
                        english = candidate
                        vietnamese = blob[:idx].strip()
                    break

    if language == "en":
        # A Vietnamese-only row has no English half; fall back to the whole
        # blob so the outcome is imported rather than silently dropped.
        return english or (blob if not _has_vietnamese(blob) else "")
    return vietnamese or blob


def _warn_on_numbering_gaps(outcomes: list[ParsedOutcome], warnings: list[str]) -> None:
    """Flag source codes that will not survive the round trip.

    ``course_learning_outcomes`` derives the displayed ``L.O.x.y`` code
    from sibling order, so a syllabus that skips a number (CO2017 has
    L.O.1.1 then L.O.1.3) comes back renumbered. The stored PDF still
    says 1.3, so the manager is told rather than left to notice.
    """
    by_parent: dict[str | None, list[int]] = {}
    for outcome in outcomes:
        last = outcome.code.rsplit(".", 1)[-1]
        if last.isdigit():
            by_parent.setdefault(outcome.parent_code, []).append(int(last))

    for parent, numbers in by_parent.items():
        if sorted(numbers) != list(range(1, len(numbers) + 1)):
            label = f"L.O.{parent}" if parent else "the top level"
            warnings.append(
                f"outcome_numbering_gap: the codes under {label} are not "
                f"consecutive ({', '.join(str(n) for n in sorted(numbers))}); "
                "imported outcomes are renumbered consecutively."
            )


def _slugify(value: str) -> str:
    """ASCII, lowercase, hyphen-joined — Vietnamese letters folded to ASCII."""
    folded = unicodedata.normalize("NFD", value)
    ascii_only = "".join(ch for ch in folded if unicodedata.category(ch) != "Mn")
    ascii_only = ascii_only.replace("đ", "d").replace("Đ", "D")
    return re.sub(r"[^a-z0-9]+", "-", ascii_only.lower()).strip("-")


__all__ = [
    "ParsedOutcome",
    "ParsedSyllabus",
    "SyllabusLanguage",
    "SyllabusParseError",
    "parse_syllabus_pages",
    "parse_syllabus_pdf",
]
