"""Ingestion preprocessing — noise filtering between extraction and chunking.

Answers one question per page: is this teachable content, noise, or
something that needs a different extractor? Cover pages, instructor blocks,
tables of contents, running headers/footers, page numbers, blank pages and
"Thank you! Questions?" slides are tagged or stripped; image-only pages are
routed to OCR rather than dropped.

Nothing is hard-deleted without a recorded ``Decision`` carrying the removed
text and a reason code, and page/slide markers are structurally protected so
citation can never break.
"""

from abridgeai.ai.preprocessing.base import (
    ROLE_BODY,
    ROLE_DIVIDER,
    ROLE_FRONT_MATTER,
    ROLE_REFERENCE,
    ROLE_REVIEW,
    ROLE_SUMMARY,
    Action,
    Decision,
    LineFacts,
    PageFacts,
    PageUnit,
    PreprocessReport,
    ReasonCode,
)
from abridgeai.ai.preprocessing.pipeline import (
    PAGED_SOURCE_TYPES,
    PageAdjudicator,
    PageOcr,
    PreprocessConfig,
    run_preprocessing,
)

__all__ = [
    "PAGED_SOURCE_TYPES",
    "ROLE_BODY",
    "ROLE_DIVIDER",
    "ROLE_FRONT_MATTER",
    "ROLE_REFERENCE",
    "ROLE_REVIEW",
    "ROLE_SUMMARY",
    "Action",
    "Decision",
    "LineFacts",
    "PageAdjudicator",
    "PageFacts",
    "PageOcr",
    "PageUnit",
    "PreprocessConfig",
    "PreprocessReport",
    "ReasonCode",
    "run_preprocessing",
]
