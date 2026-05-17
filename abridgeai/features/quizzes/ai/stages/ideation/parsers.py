"""Pydantic validation + parsing for the ideation LLM JSON response (T5.5).

Ports the implicit shape contract from the legacy
``backend/app/ai/haystack/pipelines/quiz_generation.py:_ideate_for_outline``
which inspected ``llm_result.content_json["templates"]`` ad-hoc. We make
that shape explicit so a malformed LLM response fails fast with a
structured error (audit row already written by the gateway) instead of
silently dropping templates and starving the generation stage.

Validation policy
-----------------
* ``templates`` is required (LLM contract; our JSON schema mandates it).
* Each template's ``section_id`` and ``source_chunk_ids`` are required —
  the downstream stage uses them as routing keys.
* Optional metadata (``topic``, ``rationale``, ``bloom_level``,
  ``difficulty``, ``question_type``, ``position``) are coerced to strings
  / ints when present and defaulted otherwise. The LLM occasionally
  returns ``"position": "1"`` as a string; we accept that.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, ValidationError, field_validator

from abridgeai.ai.llm.errors import ResponseFormatError


class Template(BaseModel):
    """A single ideated quiz template.

    Field order follows the LLM JSON schema in
    ``prompts/user.j2``. ``model_dump()`` round-trips through the legacy
    ``dict[str, Any]`` shape that the generation stage (T5.6) consumes.
    """

    position: int = 0
    section_id: str = Field(..., min_length=1)
    topic: str = ""
    question_type: str = "mcq"
    bloom_level: str = "understand"
    difficulty: str = "medium"
    source_chunk_ids: list[str] = Field(default_factory=list)
    rationale: str = ""

    @field_validator("source_chunk_ids", mode="before")
    @classmethod
    def _coerce_chunk_ids(cls, value: Any) -> list[str]:  # noqa: ANN401 -- raw LLM JSON
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item) for item in value if item is not None]
        return [str(value)]

    @field_validator("position", mode="before")
    @classmethod
    def _coerce_position(cls, value: Any) -> int:  # noqa: ANN401 -- raw LLM JSON
        if value is None or value == "":
            return 0
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0


class IdeationResponse(BaseModel):
    """Top-level shape of the ideation LLM JSON response."""

    templates: list[Template] = Field(default_factory=list)


def parse_ideation_response(content_json: Any) -> list[Template]:  # noqa: ANN401 -- raw LLM JSON
    """Validate the LLM ideation response into a list of templates.

    Parameters
    ----------
    content_json
        Whatever ``LLMResult.content_json`` returned. The gateway already
        guarantees this is a dict or list (json-decoded). We require a
        dict carrying a ``templates`` key per the prompt schema.

    Returns
    -------
    list[Template]
        Validated template objects in input order. Empty list when the
        LLM returned an empty array (legal — outline could have produced
        zero budget).

    Raises
    ------
    ResponseFormatError
        When ``content_json`` is not a dict, lacks ``templates``, or any
        template entry fails validation (missing ``section_id``,
        non-list shape, etc.). The gateway has already written a
        ``success`` audit row for this LLM call; this error then bubbles
        up so the worker marks the run failed.
    """
    if not isinstance(content_json, dict):
        raise ResponseFormatError(
            f"ideation response must be a JSON object, got {type(content_json).__name__}"
        )

    raw_templates = content_json.get("templates")
    if raw_templates is None:
        raise ResponseFormatError("ideation response missing required 'templates' key")
    if not isinstance(raw_templates, list):
        raise ResponseFormatError(
            f"ideation response 'templates' must be a list, got {type(raw_templates).__name__}"
        )

    parsed: list[Template] = []
    for index, entry in enumerate(raw_templates):
        if not isinstance(entry, dict):
            # Skip non-dict entries silently — matches legacy behaviour
            # which used ``isinstance(entry, dict)`` as a soft filter.
            continue
        try:
            parsed.append(Template.model_validate(entry))
        except ValidationError as exc:
            raise ResponseFormatError(
                f"ideation template at index {index} failed validation: {exc.errors()[:3]}"
            ) from exc
    return parsed


__all__ = ["IdeationResponse", "Template", "parse_ideation_response"]
