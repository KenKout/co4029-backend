"""Public API for the extraction package.

Importing this module imports every built-in extractor module, which triggers
their ``@register_extractor`` decorators and populates ``EXTRACTOR_REGISTRY``.
Callers should treat ``abridgeai.ai.extraction`` as the single entry point.
"""

from abridgeai.ai.extraction._local_mock import (
    LocalMockExtractor,
    maybe_local_mock_extractor,
)
from abridgeai.ai.extraction.audio import AudioExtractor
from abridgeai.ai.extraction.base import (
    ExtractedContent,
    MaterialExtractor,
    SourceLocation,
)
from abridgeai.ai.extraction.code import CODE_MIMES, CodeExtractor
from abridgeai.ai.extraction.docx import DocxExtractor
from abridgeai.ai.extraction.html import HtmlExtractor
from abridgeai.ai.extraction.image import ImageExtractor
from abridgeai.ai.extraction.pdf import PdfExtractor
from abridgeai.ai.extraction.pptx import PptxExtractor
from abridgeai.ai.extraction.registry import (
    EXTRACTOR_REGISTRY,
    UnsupportedMimeError,
    dispatch_extractor,
    register_extractor,
)
from abridgeai.ai.extraction.text import TextExtractor
from abridgeai.ai.extraction.video import VideoExtractor
from abridgeai.ai.extraction.xlsx import XlsxExtractor

__all__ = [
    "CODE_MIMES",
    "EXTRACTOR_REGISTRY",
    "AudioExtractor",
    "CodeExtractor",
    "DocxExtractor",
    "ExtractedContent",
    "HtmlExtractor",
    "ImageExtractor",
    "LocalMockExtractor",
    "MaterialExtractor",
    "PdfExtractor",
    "PptxExtractor",
    "SourceLocation",
    "TextExtractor",
    "UnsupportedMimeError",
    "VideoExtractor",
    "XlsxExtractor",
    "dispatch_extractor",
    "maybe_local_mock_extractor",
    "register_extractor",
]
