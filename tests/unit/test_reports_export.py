"""Unit tests for report CSV/XLSX export serializers (Phase 10)."""

from __future__ import annotations

import csv
import io

import pytest

from abridgeai.features.quizzes.services.reports_export import build_xlsx, stream_csv


def test_stream_csv_roundtrips():
    headers = ["Student", "Question", "Answer"]
    rows = [["Alice", "Q1", "42"], ["Bob", "Q2", "yes"]]
    body = "".join(stream_csv(headers, rows))
    parsed = list(csv.reader(io.StringIO(body)))
    assert parsed[0] == headers
    assert parsed[1] == ["Alice", "Q1", "42"]


def test_stream_csv_escapes_commas_and_quotes():
    headers = ["Prompt"]
    rows = [['a, b "c"']]
    body = "".join(stream_csv(headers, rows))
    parsed = list(csv.reader(io.StringIO(body)))
    assert parsed[1] == ['a, b "c"']


def test_build_xlsx_opens():
    openpyxl = pytest.importorskip("openpyxl")
    content = build_xlsx(["A", "B"], [["1", "2"]])
    wb = openpyxl.load_workbook(io.BytesIO(content))
    ws = wb.active
    assert ws["A1"].value == "A"
    assert ws["A2"].value == "1"
