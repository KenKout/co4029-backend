"""Unit tests for the course-syllabus parser.

The fixture is a TEXT block rather than a committed PDF: the real inputs
are third-party university documents, and hand-written text lets each test
isolate one layout quirk instead of relying on which quirks happen to
co-occur in a sample file. It is modelled on the actual
``ĐỀ CƯƠNG HỌC PHẦN`` template — the same section numbering, the same
bilingual label pairs, and the quirks that make it awkward to parse:

* the §1.1 course-format table and the evaluation-ratio table BOTH end in a
  ``Tổng cộng (Total)`` row, and only the first holds hours;
* §2 sometimes runs the Vietnamese and English halves together on one line;
* an outcome may end in a Vietnamese parenthetical that must not be
  mistaken for the English translation.

Field-level tests go through :func:`parse_syllabus_pages`, which takes the
page text PyMuPDF would have produced. Feeding them through a *generated*
PDF instead would test the font: the faces PyMuPDF can synthesize have no
Vietnamese coverage, so every diacritic would come back as ``?``. The PDF
layer is covered separately by the rejection tests at the bottom, which
only need ASCII.

The parser is pure, so none of this needs a database.
"""

from __future__ import annotations

import fitz
import pytest

from abridgeai.features.courses.ingest import (
    SyllabusParseError,
    parse_syllabus_pages,
    parse_syllabus_pdf,
)

# A faithful reduction of the real template: bilingual title lines, the
# course-format table with its Total row, the evaluation table with its own
# (percentage) Total row, a bilingual §2, and a §4.2 outcome tree.
_SYLLABUS = """\
ĐỀ CƯƠNG HỌC PHẦN
Course Syllabus
1. Thông tin về học phần (Course information)
1.1. Thông tin tổng quan (General information)
- Tên học phần: Hệ điều hành
  Course title: Operating Systems
- Mã học phần (Course ID): CO2017
- Số tín chỉ (Credits): 3
Lý thuyết (LT)
30
Tự học (Self-study)
95
Tổng cộng (Total)
126.5
3
- Tỷ lệ đánh giá (Evaluation form & ratio)
Thi (Final Exam)
50%
Tổng cộng (Total)
100%
2. Mô tả học phần (Course description)
Môn học gồm các nội dung chính sau: Quá trình và luồng.

Topics covered in this course include: Process and threads.

3. Giáo trình và tài liệu học tập (Course materials)
[1] Operating System Concepts.
4. Mục tiêu và kết quả học tập mong đợi (Goals and Learning outcomes)
4.1. Mục tiêu của học phần (Course goals)
Truyền đạt kiến thức.
4.2. Chuẩn đầu ra học phần (Course learning outcomes)
L.O.1 - Mô tả kiến thức nền tảng
            (Describe fundamental knowledge)
L.O.1.1 - Định nghĩa chức năng
            (Define functionality)
L.O.1.3 - Giải thích bộ nhớ ảo
            (Explain virtual memory)
L.O.2 - Mối quan hệ giữa hiệu suất và tài nguyên
            (Report the tradeoffs between performance and resources)
L.O.2.1 - So sánh các giải thuật định thời
            (Compare common scheduling algorithms)
5. Phương thức giảng dạy và học tập (Teaching and assessment methods)
Thuyết giảng.
"""


def _pages(text: str) -> list[str]:
    """The syllabus as a single page of extracted text."""
    return [text]


@pytest.fixture(scope="module")
def syllabus() -> list[str]:
    return _pages(_SYLLABUS)


def test_parses_vietnamese_title_and_code(syllabus: list[str]) -> None:
    parsed = parse_syllabus_pages(syllabus, "vi")
    assert parsed.title == "Hệ điều hành"
    assert parsed.course_code == "CO2017"


def test_parses_english_title(syllabus: list[str]) -> None:
    assert parse_syllabus_pages(syllabus, "en").title == "Operating Systems"


def test_required_hours_come_from_the_course_format_table(syllabus: list[str]) -> None:
    """Not from the evaluation table's ``Tổng cộng (Total) 100%`` row.

    Both tables end in an identically-labelled Total row, so a naive
    "first Total row" reader would return 100 (percent) as the hours.
    """
    parsed = parse_syllabus_pages(syllabus, "vi")
    assert parsed.required_hours == 126.5
    assert parsed.estimated_minutes == 7590


def test_description_is_language_scoped(syllabus: list[str]) -> None:
    vi = parse_syllabus_pages(syllabus, "vi").description or ""
    en = parse_syllabus_pages(syllabus, "en").description or ""
    assert "Quá trình và luồng" in vi
    assert "Process and threads" not in vi
    assert "Process and threads" in en
    assert "Quá trình và luồng" not in en


def test_description_stops_before_section_three(syllabus: list[str]) -> None:
    """The course-materials section must not bleed into the description."""
    for language in ("vi", "en"):
        description = parse_syllabus_pages(syllabus, language).description or ""
        assert "Operating System Concepts" not in description


def test_outcome_tree_uses_the_lo_codes(syllabus: list[str]) -> None:
    parsed = parse_syllabus_pages(syllabus, "vi")
    by_code = {o.code: o for o in parsed.outcomes}
    assert set(by_code) == {"1", "1.1", "1.3", "2", "2.1"}
    assert by_code["1"].parent_code is None
    assert by_code["1.1"].parent_code == "1"
    assert by_code["2.1"].parent_code == "2"


def test_outcome_text_splits_on_the_trailing_parenthetical(syllabus: list[str]) -> None:
    """``<vi text> (<en text>)`` — the parenthesised half is the English one."""
    vi = {o.code: o.text for o in parse_syllabus_pages(syllabus, "vi").outcomes}
    en = {o.code: o.text for o in parse_syllabus_pages(syllabus, "en").outcomes}
    assert vi["1.1"] == "Định nghĩa chức năng"
    assert en["1.1"] == "Define functionality"


def test_outcomes_stop_before_section_five(syllabus: list[str]) -> None:
    codes = {o.code for o in parse_syllabus_pages(syllabus, "vi").outcomes}
    assert codes == {"1", "1.1", "1.3", "2", "2.1"}


def test_numbering_gap_is_reported_as_a_warning(syllabus: list[str]) -> None:
    """L.O.1.1 then L.O.1.3 — the stored code is re-derived from sibling
    order, so 1.3 comes back as 1.2 and the manager has to be told."""
    parsed = parse_syllabus_pages(syllabus, "vi")
    assert any(w.startswith("outcome_numbering_gap") for w in parsed.warnings)


def test_clean_syllabus_has_no_numbering_warning() -> None:
    clean = _SYLLABUS.replace("L.O.1.3 -", "L.O.1.2 -")
    parsed = parse_syllabus_pages(_pages(clean), "vi")
    assert not any(w.startswith("outcome_numbering_gap") for w in parsed.warnings)


def test_glued_bilingual_description_line_is_split() -> None:
    """Some syllabi run both languages together on one line.

    CO2007 does exactly this: ``• Xử lý song song The major contents
    include:``. The Vietnamese half must not swallow the English lead-in,
    and the English half must not inherit the Vietnamese bullet.
    """
    glued = _SYLLABUS.replace(
        "Môn học gồm các nội dung chính sau: Quá trình và luồng.\n\n"
        "Topics covered in this course include: Process and threads.\n",
        "Các nội dung chính bao gồm:\n"
        "• Xử lý song song The major contents include:\n"
        "• Parallel processing.\n",
    )
    parsed_vi = parse_syllabus_pages(_pages(glued), "vi")
    parsed_en = parse_syllabus_pages(_pages(glued), "en")
    assert "Xử lý song song" in (parsed_vi.description or "")
    assert "The major contents" not in (parsed_vi.description or "")
    assert "The major contents include:" in (parsed_en.description or "")
    assert "Xử lý song song" not in (parsed_en.description or "")


def test_trailing_vietnamese_parenthetical_is_not_mistaken_for_english() -> None:
    """``… mô phỏng (ngôn ngữ C)`` is Vietnamese, not the English half.

    Taking "the last parenthesised group" blindly would strip real
    Vietnamese text off the outcome and leave the English side holding it.
    """
    text = _SYLLABUS.replace(
        "L.O.2.1 - So sánh các giải thuật định thời\n"
        "            (Compare common scheduling algorithms)\n",
        "L.O.2.1 - Thực hành mô phỏng (ngôn ngữ C)\n",
    )
    parsed = parse_syllabus_pages(_pages(text), "vi")
    outcome = next(o for o in parsed.outcomes if o.code == "2.1")
    assert outcome.text == "Thực hành mô phỏng (ngôn ngữ C)"


def test_suggested_slug_folds_vietnamese_to_ascii(syllabus: list[str]) -> None:
    assert parse_syllabus_pages(syllabus, "vi").suggested_slug() == "co2017-he-dieu-hanh"
    assert (
        parse_syllabus_pages(syllabus, "en").suggested_slug() == "co2017-operating-systems"
    )


def test_missing_title_is_a_hard_failure() -> None:
    """Everything else degrades to a warning; a course needs a title."""
    text = _SYLLABUS.replace("- Tên học phần: Hệ điều hành\n", "").replace(
        "  Course title: Operating Systems\n", ""
    )
    with pytest.raises(SyllabusParseError, match="missing_course_title"):
        parse_syllabus_pages(_pages(text), "vi")


def test_missing_hours_row_degrades_to_a_warning() -> None:
    text = _SYLLABUS.replace("Tổng cộng (Total)\n126.5\n3\n", "")
    parsed = parse_syllabus_pages(_pages(text), "vi")
    assert parsed.required_hours is None
    assert parsed.estimated_minutes is None
    assert any(w.startswith("missing_required_hours") for w in parsed.warnings)


def test_missing_outcomes_section_degrades_to_a_warning() -> None:
    text = _SYLLABUS.split("4.2. Chuẩn đầu ra")[0]
    parsed = parse_syllabus_pages(_pages(text), "vi")
    assert parsed.outcomes == []
    assert any(w.startswith("missing_outcomes") for w in parsed.warnings)


def test_title_falls_back_to_the_other_language_with_a_warning() -> None:
    text = _SYLLABUS.replace("  Course title: Operating Systems\n", "")
    parsed = parse_syllabus_pages(_pages(text), "en")
    assert parsed.title == "Hệ điều hành"
    assert any(w.startswith("title_language_fallback") for w in parsed.warnings)


def test_non_pdf_bytes_are_rejected() -> None:
    with pytest.raises(SyllabusParseError, match="unreadable_pdf"):
        parse_syllabus_pdf(b"this is not a pdf at all", "vi")


def test_empty_upload_is_rejected() -> None:
    with pytest.raises(SyllabusParseError, match="empty_file"):
        parse_syllabus_pdf(b"", "vi")


def test_pdf_without_a_text_layer_is_rejected() -> None:
    """A scanned syllabus — we do not OCR here, so say so rather than
    returning an empty course."""
    doc = fitz.open()
    doc.new_page()
    with pytest.raises(SyllabusParseError, match="no_text_in_pdf"):
        parse_syllabus_pdf(bytes(doc.tobytes()), "vi")
