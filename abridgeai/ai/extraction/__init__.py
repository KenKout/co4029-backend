"""Public API for the extraction package.

Importing this module imports every built-in extractor module, which triggers
their ``@register_extractor`` decorators and populates ``EXTRACTOR_REGISTRY``.
Callers should treat ``abridgeai.ai.extraction`` as the single entry point.
"""

from abridgeai.ai.extraction.base import (
    ExtractedContent,
    MaterialExtractor,
    SourceLocation,
)
from abridgeai.ai.extraction.docx import DocxExtractor
from abridgeai.ai.extraction.pdf import PdfExtractor
from abridgeai.ai.extraction.pptx import PptxExtractor
from abridgeai.ai.extraction.registry import (
    EXTRACTOR_REGISTRY,
    UnsupportedMimeError,
    dispatch_extractor,
    register_extractor,
)

__all__ = [
    "EXTRACTOR_REGISTRY",
    "DocxExtractor",
    "ExtractedContent",
    "MaterialExtractor",
    "PdfExtractor",
    "PptxExtractor",
    "SourceLocation",
    "UnsupportedMimeError",
    "dispatch_extractor",
    "register_extractor",
]
