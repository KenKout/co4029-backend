"""Quiz ideation stage (T5.5) — extracted from god file lines 567-757.

One LLM call maps the lesson outline + per-section budget to a list of
section-tagged quiz templates. The public entry point is
:func:`ideate_for_outline`; :func:`parse_ideation_response` and helpers
are re-exported for the generation stage and tests.
"""

from __future__ import annotations

from abridgeai.features.quizzes.ai.stages.ideation.logic import (
    _default_bloom_distribution,
    _redistribute_chunk_anchors_within_section,
    _render_outline_for_prompt,
    ideate_for_outline,
)
from abridgeai.features.quizzes.ai.stages.ideation.parsers import (
    IdeationResponse,
    Template,
    parse_ideation_response,
)

__all__ = [
    "IdeationResponse",
    "Template",
    "_default_bloom_distribution",
    "_redistribute_chunk_anchors_within_section",
    "_render_outline_for_prompt",
    "ideate_for_outline",
    "parse_ideation_response",
]
