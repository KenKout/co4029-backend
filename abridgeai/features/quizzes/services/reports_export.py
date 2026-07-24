"""Report export serializers (Phase 10): table model + CSV/XLSX writers.

Format-agnostic: reports are flattened to (headers, rows) by small adapters,
then serialized to a streaming CSV or an XLSX byte blob. No DB access here.
"""

from __future__ import annotations

import csv
import io
from collections.abc import Iterator
from typing import Any

from abridgeai.features.quizzes.schemas.reports import (
    ResponsesReportRead,
    StatisticsReportRead,
)


def responses_to_table(report: ResponsesReportRead) -> tuple[list[str], list[list[Any]]]:
    headers = [
        "Student",
        "Attempt #",
        "Question #",
        "Prompt",
        "Type",
        "Their answer",
        "Correct answer",
        "Correct?",
        "Points",
    ]
    rows = [
        [
            r.student_name or str(r.student_id),
            r.attempt_number,
            r.question_position,
            r.prompt_text,
            r.question_type,
            r.student_answer,
            r.correct_answer,
            "yes" if r.is_correct else "no",
            r.points_awarded,
        ]
        for r in report.rows
    ]
    return headers, rows


def statistics_to_table(report: StatisticsReportRead) -> tuple[list[str], list[list[Any]]]:
    headers = [
        "Question #",
        "Prompt",
        "Answered",
        "Correct",
        "Facility (%)",
        "Discrimination",
        "Note",
    ]
    rows = [
        [
            r.question_position,
            r.prompt_text,
            r.answered_count,
            r.correct_count,
            round(r.facility_index * 100, 1) if r.facility_index is not None else "",
            round(r.discrimination_index, 4) if r.discrimination_index is not None else "",
            r.discrimination_note or "",
        ]
        for r in report.rows
    ]
    return headers, rows


def stream_csv(headers: list[str], rows: list[list[Any]]) -> Iterator[str]:
    """Yield CSV text row-by-row for a StreamingResponse (stdlib quoting)."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(headers)
    yield buf.getvalue()
    buf.seek(0)
    buf.truncate(0)
    for row in rows:
        writer.writerow(row)
        yield buf.getvalue()
        buf.seek(0)
        buf.truncate(0)


def build_xlsx(headers: list[str], rows: list[list[Any]]) -> bytes:
    """Serialize a table to an .xlsx byte blob (bold + frozen header)."""
    from openpyxl import Workbook  # noqa: PLC0415
    from openpyxl.styles import Font  # noqa: PLC0415

    wb = Workbook()
    ws = wb.active
    ws.append(headers)
    for cell in ws[1]:
        cell.font = Font(bold=True)
    ws.freeze_panes = "A2"
    for row in rows:
        ws.append(row)
    out = io.BytesIO()
    wb.save(out)
    return out.getvalue()


__all__ = [
    "build_xlsx",
    "responses_to_table",
    "statistics_to_table",
    "stream_csv",
]
