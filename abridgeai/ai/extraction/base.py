"""Protocol + dataclasses describing the extraction contract.

The extraction layer turns a raw uploaded artifact (PDF, DOCX, PPTX, ...) into
text plus a list of ``SourceLocation`` records that downstream chunkers and
citation builders can use to point back at the original document.

Extractors implement ``MaterialExtractor`` *structurally* — they do not inherit
from a base class. Each format lives in its own module under
``abridgeai.ai.extraction`` and registers itself against one or more MIME
types via the ``register_extractor`` decorator from
``abridgeai.ai.extraction.registry``.

The Protocol is marked ``runtime_checkable`` so tests can perform
``isinstance(PdfExtractor(), MaterialExtractor)`` smoke checks; the production
dispatch path uses MIME-string lookup, not isinstance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, BinaryIO, Protocol, runtime_checkable


@dataclass(frozen=True)
class SourceLocation:
    """Pointer back to a region of the original document.

    Every field is optional so a single dataclass can describe paged formats
    (PDF/PPTX -> ``page``), text-line formats (DOCX paragraphs, code, plain
    text -> ``line_start``/``line_end``), and time-coded formats (audio/video
    -> ``timestamp_start_ms``/``timestamp_end_ms``). ``bbox`` is reserved for
    OCR/image extractors that can produce pixel-space rectangles.

    Citations downstream pick whichever fields are populated.
    """

    page: int | None = None
    line_start: int | None = None
    line_end: int | None = None
    timestamp_start_ms: int | None = None
    timestamp_end_ms: int | None = None
    bbox: tuple[float, float, float, float] | None = None


@dataclass(frozen=True)
class ExtractedContent:
    """Result of running an extractor against one source artifact.

    ``text`` is the full concatenated body, suitable for direct chunking.
    ``source_locations`` is parallel to logical units within ``text`` (pages,
    paragraphs, slides, ...): chunkers map character offsets back to a
    ``SourceLocation`` by walking this list. ``source_type`` is a short tag
    (``"pdf"``, ``"docx"``, ...) that lets chunkers pick a strategy without
    re-deriving it from the MIME type.
    """

    text: str
    metadata: dict[str, Any]
    source_type: str
    source_locations: list[SourceLocation] = field(default_factory=list)


@runtime_checkable
class MaterialExtractor(Protocol):
    """Structural contract every extractor satisfies.

    ``source`` is polymorphic so callers can hand off the most efficient
    representation they have:

    * ``BinaryIO`` — a streaming file handle (e.g. the response body of an S3
      ``get_object`` call).
    * ``bytes`` — the entire artifact already in memory.
    * ``str`` — a filesystem path, used by tests and CLI tools.

    The implementation MUST be ``async`` so heavy synchronous parsers can be
    wrapped in ``asyncio.to_thread`` without forcing every caller to do so.
    """

    async def extract(self, source: BinaryIO | bytes | str) -> ExtractedContent: ...
