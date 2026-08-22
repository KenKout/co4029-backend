"""Deterministic parsers that turn an uploaded document into course fields.

Pure functions only — no DB, no storage, no settings. The orchestration
(create the course, persist the file, notify) lives in
``features.courses.services.syllabus_import``; keeping the parsing side
effect-free is what lets it be unit-tested against real PDFs without a
database or a running worker.
"""

from abridgeai.features.courses.ingest.syllabus import (
    ParsedOutcome,
    ParsedSyllabus,
    SyllabusParseError,
    parse_syllabus_pages,
    parse_syllabus_pdf,
)

__all__ = [
    "ParsedOutcome",
    "ParsedSyllabus",
    "SyllabusParseError",
    "parse_syllabus_pages",
    "parse_syllabus_pdf",
]
