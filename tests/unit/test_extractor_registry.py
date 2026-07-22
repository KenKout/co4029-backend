from __future__ import annotations

import pytest

from abridgeai.ai.extraction import (
    DocxExtractor,
    PdfExtractor,
    PptxExtractor,
    UnsupportedMimeError,
    XlsxExtractor,
    dispatch_extractor,
    register_extractor,
)
from abridgeai.ai.extraction.base import ExtractedContent, MaterialExtractor
from abridgeai.ai.extraction.registry import EXTRACTOR_REGISTRY


def test_pdf_dispatch() -> None:
    extractor = dispatch_extractor("application/pdf")
    assert isinstance(extractor, PdfExtractor)


def test_docx_dispatch() -> None:
    mime = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    extractor = dispatch_extractor(mime)
    assert isinstance(extractor, DocxExtractor)


def test_pptx_dispatch() -> None:
    mime = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
    extractor = dispatch_extractor(mime)
    assert isinstance(extractor, PptxExtractor)


def test_xlsx_dispatch() -> None:
    mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    extractor = dispatch_extractor(mime)
    assert isinstance(extractor, XlsxExtractor)


def test_xls_dispatch() -> None:
    extractor = dispatch_extractor("application/vnd.ms-excel")
    assert isinstance(extractor, XlsxExtractor)


def test_unknown_raises() -> None:
    with pytest.raises(UnsupportedMimeError) as excinfo:
        dispatch_extractor("application/x-no-such-extractor")
    assert "application/x-no-such-extractor" in str(excinfo.value)


def test_register_extractor_decorator_returns_class_unchanged() -> None:
    sentinel_mime = "application/x-test-extractor"

    @register_extractor(sentinel_mime)
    class StubExtractor:
        async def extract(self, source: object) -> ExtractedContent:
            return ExtractedContent(
                text="stub",
                metadata={},
                source_type="stub",
                source_locations=[],
            )

    try:
        assert EXTRACTOR_REGISTRY[sentinel_mime] is StubExtractor
        instance = dispatch_extractor(sentinel_mime)
        assert isinstance(instance, StubExtractor)
        assert isinstance(instance, MaterialExtractor)
    finally:
        EXTRACTOR_REGISTRY.pop(sentinel_mime, None)


def test_built_in_mimes_registered() -> None:
    assert "application/pdf" in EXTRACTOR_REGISTRY
    docx_mime = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    pptx_mime = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
    assert docx_mime in EXTRACTOR_REGISTRY
    assert pptx_mime in EXTRACTOR_REGISTRY
    xlsx_mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    assert xlsx_mime in EXTRACTOR_REGISTRY
    assert "application/vnd.ms-excel" in EXTRACTOR_REGISTRY
