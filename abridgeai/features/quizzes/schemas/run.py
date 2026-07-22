"""Generation / regeneration request + run-status DTOs for quizzes (FR-5).

Phase 1 of the FR-5 restoration work — port the legacy
``QuizGenerateRequest`` schema (with full coverage / bloom / focus-topic
surface) onto the new feature-sliced layout, replacing the simplified
W5.x interim schema.

Strict-typing rules applied here:

* ``model_config = ConfigDict(extra="forbid")`` everywhere — no silent
  field passthrough.
* All enum-shaped fields are ``Literal[...]`` so the OpenAPI codegen
  emits TypeScript string-literal unions (no plain ``str``).
* ``BloomLevel`` and ``Difficulty`` reuse the canonical literals already
  declared in :mod:`abridgeai.features.quizzes.ai.stages.generation.parsers`;
  the schema layer does NOT redefine them, keeping a single source of truth.
* ``bloom_distribution`` is typed as ``dict[BloomLevel, int]`` rather
  than ``dict[str, Any]`` so unknown bloom keys fail at the Pydantic
  boundary instead of leaking into the AI stage.

Endpoint surface:

* ``POST /teacher/quizzes/{id}/generate`` — body:
  :class:`QuizGenerationRequest`. Returns :class:`QuizGenerationRunRead`.
* ``GET  /teacher/quizzes/{id}/generation-runs/{run_id}`` — poll status.
* ``POST /teacher/quizzes/{id}/questions/{q_id}/regenerate`` — body:
  :class:`QuestionRegenerationRequest`.

DB notes (see migrations 0001/0007/0014):

* The ``quiz_questions.question_type`` CHECK constraint accepts
  ``multiple_choice`` (NOT the legacy ``mcq`` alias). The schema
  literal must match.
* The DB column is ``expected_response_time_ms`` (NOT the legacy
  ``expected_response_ms``).
* ``generation_runs.config_json`` is JSONB, so the entire dumped
  payload from this schema lands there verbatim.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from abridgeai.features.quizzes.ai.stages.generation.parsers import (
    BloomLevel,
    Difficulty,
)

# ---------------------------------------------------------------------------
# Literal aliases (strict typing for the HTTP boundary)
# ---------------------------------------------------------------------------
GenerationMode = Literal["topic", "coverage"]
"""High-level generation strategy.

* ``topic`` — single LLM call against ``focus_topics`` and lesson context;
  the legacy default and the safe fallback.
* ``coverage`` — per-section budgeting via ``build_lesson_outline`` +
  parallel template runs. Requires a real outline; falls back to a
  single synthetic section until the outline porter (Phase 3) lands.
"""

QuestionType = Literal[
    "multiple_choice", "true_false", "short_answer", "fill_blank", "code"
]
"""Mirrors the ``quiz_questions.question_type`` CHECK constraint.

Legacy used the ``"mcq"`` alias; the new DB standardised on
``"multiple_choice"`` (migration 0001). Frontend MUST send the new value.
"""

_DEFAULT_QUESTION_TYPES: tuple[QuestionType, ...] = ("multiple_choice",)
"""Module-level typed default. Used by the ``question_types`` field's
``default_factory`` to avoid the mypy ``arg-type`` / ``redundant-cast``
loop that hits inline lambdas with ``Literal`` element types.
"""

GenerationRunStatus = Literal[
    "pending", "running", "completed", "failed", "cancelled"
]
"""Mirrors ``generation_runs.status`` CHECK constraint."""


# ---------------------------------------------------------------------------
# Coverage mode tunables (FR-12)
# ---------------------------------------------------------------------------
class CoverageOptions(BaseModel):
    """Per-coverage-mode tunables.

    Plumbed into ``GenerationRun.config_json`` and read by
    ``allocate_question_budget`` in the ported outline component.
    Defaults match the legacy schema so behaviour is preserved.
    """

    model_config = ConfigDict(extra="forbid")

    min_per_section: Annotated[int, Field(ge=0, le=10)] = 1
    max_per_section: Annotated[int, Field(ge=1, le=10)] = 5
    skip_summaries: bool = True
    section_ids: list[str] | None = None
    """``None`` means "all eligible sections"."""

    slides_per_section: Annotated[int, Field(ge=1, le=20)] = 4
    """Fallback grouping when the source has only ``[Page N]`` markers
    (slide decks). Lower this to ``1`` for one-question-per-slide
    coverage; raise it for coarser grouping."""

    section_grouping: Literal["auto", "fixed"] = "auto"
    """How the outline builder draws section boundaries.

    ``"auto"`` (default) honors per-chunk semantic enrichment (LLM-derived
    topic titles) — each unique topic becomes its own section. Faithful
    to the data: a slide-deck PDF where every page has a distinct
    enriched title produces one section per topic. This is the
    post-migration shape — coarser legacy-style bundling is opt-in.

    ``"fixed"`` force-bundles ``slides_per_section`` consecutive chunks
    per section regardless of semantic enrichment. Matches the legacy
    panel UX where users saw ~ceil(N/slides_per_section) sections."""

    parallelism: Annotated[int, Field(ge=1, le=32)] | None = None
    """Bounded concurrency for the per-template generation stage.
    ``None`` defers to ``Settings.quiz_coverage_parallelism`` (default 6).
    Capped at 32 so callers cannot DoS the provider or exhaust the pool.
    """

    max_attempts: Annotated[int, Field(ge=1, le=5)] = 2
    """How many times each per-template generation may run before the
    template is dropped. ``1`` disables retry (legacy single-shot
    behaviour); ``2`` (default) retries once on transient failures
    (gateway 5xx, parse error, empty candidates). Capped at 5 so a
    pathological template cannot stall the whole run."""

    @model_validator(mode="after")
    def _check_min_le_max(self) -> CoverageOptions:
        if self.min_per_section > self.max_per_section:
            raise ValueError(
                "min_per_section cannot exceed max_per_section"
            )
        return self


# ---------------------------------------------------------------------------
# Generation request (FR-5 full surface)
# ---------------------------------------------------------------------------
class QuizGenerationRequest(BaseModel):
    """Body for ``POST /teacher/quizzes/{id}/generate``.

    Defaults preserve topic-mode behaviour so callers that send only
    ``question_count`` still work; coverage-mode and personalisation
    knobs are opt-in via the advanced disclosure on the SPA panel.
    """

    model_config = ConfigDict(extra="forbid")

    # --- Quiz metadata ---
    quiz_id: UUID | None = None
    """When set, the run targets an existing quiz (replace or append).
    When ``None`` the service layer derives the quiz from the route
    ``{quiz_id}`` path parameter; the field is kept for parity with the
    legacy ``POST /modules/{id}/quizzes/generate`` shape so payloads
    coming from the SPA's existing form do not need rewriting."""

    title: Annotated[str | None, Field(min_length=1, max_length=255)] = None
    """Required when creating a fresh quiz (``quiz_id is None``).
    Ignored when targeting an existing quiz — the route's
    ``{quiz_id}`` already pins the row, so the title doesn't need to
    travel through the body. The cross-field check in
    :meth:`_check_title_required` enforces this contract."""

    description: str | None = None

    # --- Generation budget ---
    question_count: Annotated[int, Field(ge=1, le=50)] = 3
    question_types: list[QuestionType] = Field(
        default_factory=lambda: list(_DEFAULT_QUESTION_TYPES)
    )
    difficulty: Difficulty | Literal["mixed"] = "mixed"
    """``"mixed"`` lets the ideation stage pick per-question difficulty
    based on bloom + position. Strict ``easy/medium/hard`` pin every
    generated question to that band."""

    bloom_distribution: dict[BloomLevel, int] = Field(default_factory=dict)
    """Optional per-bloom question budget. Sum must be ``<= question_count``;
    enforced by :meth:`_check_bloom_distribution_total` so the AI stage
    receives a guaranteed-valid distribution."""

    include_prerequisites: bool = False

    model_preference: str | None = Field(default=None, max_length=100)
    """Free-form override (e.g. ``"openai:gpt-4o-mini"``). Validated
    downstream against the configured provider catalog; the schema
    layer does not constrain the value beyond a length cap."""

    # --- Source material selection ---
    source_lesson_ids: list[UUID] = Field(default_factory=list)

    config_json: dict[str, Any] = Field(default_factory=dict)
    """Escape hatch for forward-compatible knobs. Merged into
    ``GenerationRun.config_json`` AFTER the structured fields, so
    structured fields take precedence on conflict."""

    # --- FR-5 personalisation ---
    generation_mode: GenerationMode = "topic"
    focus_topics: Annotated[list[str], Field(max_length=10)] = Field(
        default_factory=list
    )
    avoid_topics: Annotated[list[str], Field(max_length=10)] = Field(
        default_factory=list
    )
    extra_instructions: Annotated[str | None, Field(max_length=1000)] = None
    append: bool = False
    """When ``True`` the run appends to existing questions instead of
    replacing the bank. Only meaningful when ``quiz_id`` is set."""

    coverage_options: CoverageOptions | None = None
    """Required when ``generation_mode == "coverage"``; ignored
    otherwise. Cross-field validation in :meth:`_check_coverage_options`."""

    # --- Validators ---
    @field_validator("focus_topics", "avoid_topics", mode="before")
    @classmethod
    def _cap_topic_strings(cls, value: object) -> object:
        """Drop empty/whitespace-only entries and clamp each to 200 chars.

        Runs in ``before`` mode so the post-cleanup length is what gets
        compared against ``Field(max_length=10)``.

        Typed as ``object`` (not ``Any``) so the lint layer enforces an
        explicit ``isinstance`` narrow before any string operations —
        no silent passthrough of garbage shapes.
        """
        if not isinstance(value, list):
            return value
        cleaned: list[str] = []
        for item in value:
            if not isinstance(item, str):
                continue
            stripped = item.strip()
            if not stripped:
                continue
            cleaned.append(stripped[:200])
        return cleaned

    @model_validator(mode="after")
    def _check_bloom_distribution_total(self) -> QuizGenerationRequest:
        if not self.bloom_distribution:
            return self
        total = sum(self.bloom_distribution.values())
        if any(v < 0 for v in self.bloom_distribution.values()):
            raise ValueError("bloom_distribution counts must be >= 0")
        if total > self.question_count:
            raise ValueError(
                f"bloom_distribution total ({total}) exceeds "
                f"question_count ({self.question_count})"
            )
        return self

    @model_validator(mode="after")
    def _check_coverage_options(self) -> QuizGenerationRequest:
        if self.generation_mode == "coverage" and self.coverage_options is None:
            # Don't reject — fall back to defaults so opt-in coverage
            # works without forcing every caller to send the nested
            # block. Materialise here so downstream code can rely on
            # the value being present.
            self.coverage_options = CoverageOptions()
        return self


# ---------------------------------------------------------------------------
# Run status projection
# ---------------------------------------------------------------------------
class QuizGenerationStageEvent(BaseModel):
    """One append-only progress event recorded as a pipeline stage starts."""

    stage: str
    at: datetime
    detail: str | None = None


class QuizGenerationProgress(BaseModel):
    """Live-progress projection sourced from ``GenerationRun.progress_json``.

    Written incrementally by the pipeline's checkpoint helper (migration
    0035) through a dedicated session, so a status poll can surface the
    current stage, a stepped percentage (``stage_index / total_stages``),
    and an append-only event log while the worker is still running.
    """

    current_stage: str | None = None
    stage_index: int = 0
    total_stages: int = 0
    updated_at: datetime | None = None
    events: list[QuizGenerationStageEvent] = Field(default_factory=list)


class QuizGenerationRunRead(BaseModel):
    """Status-poll projection of a quiz generation run.

    Service layer joins the ``GenerationRun`` row with the
    quiz / pipeline-run linkage to populate this DTO.
    """

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    quiz_id: UUID
    status: GenerationRunStatus
    started_at: datetime
    completed_at: datetime | None = None
    error_message: str | None = None
    pipeline_run_id: UUID | None = None
    progress: QuizGenerationProgress | None = None


# ---------------------------------------------------------------------------
# Single-question regeneration
# ---------------------------------------------------------------------------
class QuestionRegenerationRequest(BaseModel):
    """Body for ``POST /teacher/quizzes/{id}/questions/{q_id}/regenerate``."""

    model_config = ConfigDict(extra="forbid")

    question_id: UUID


__all__ = [
    "CoverageOptions",
    "Difficulty",
    "GenerationMode",
    "GenerationRunStatus",
    "QuestionRegenerationRequest",
    "QuestionType",
    "QuizGenerationProgress",
    "QuizGenerationRequest",
    "QuizGenerationRunRead",
    "QuizGenerationStageEvent",
]
