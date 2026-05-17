"""Pydantic specs for eval scenario YAMLs.

A scenario YAML describes one capability under test plus a list of fixtures
to run it against and a list of judge criteria. The runner discovers YAMLs
under `eval/scenarios/`, validates each against `ScenarioSpec`, and
dispatches to the registered capability runner.

T8.2 ships the spec + scenario discovery + dry-run summary. T8.4 will wire
the runners up to real LLM calls.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

ScenarioMode = Literal["full", "coverage"]


class CriterionSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, description="Stable machine-readable criterion id.")
    description: str = Field(min_length=1)
    expected_score_threshold: float = Field(
        ge=1.0,
        le=5.0,
        description="Below this 1-5 score the scenario is flagged as a regression.",
    )


class ScenarioSpec(BaseModel):
    """Validated shape of a `scenarios/*.yaml` file."""

    model_config = ConfigDict(extra="forbid")

    scenario_id: str = Field(
        min_length=1,
        pattern=r"^[a-z][a-z0-9_]*$",
        description="Snake_case id; should match the YAML filename stem.",
    )
    description: str = Field(min_length=1)
    capability: str = Field(
        min_length=1,
        description="Key into eval.runners.REGISTRY (e.g. quiz_generation).",
    )
    mode: ScenarioMode | None = Field(
        default=None,
        description="Capability-specific mode (e.g. quiz: 'full' vs 'coverage').",
    )
    fixtures: list[str] = Field(
        min_length=1,
        description="Fixture ids; each must resolve to a file in eval/fixtures/.",
    )
    inputs: dict[str, Any] = Field(default_factory=dict)
    criteria: list[CriterionSpec] = Field(min_length=1)
    estimated_cost_usd_per_fixture: float = Field(
        ge=0.0,
        description="Used by the budget pre-check; multiplied by len(fixtures).",
    )

    @field_validator("fixtures")
    @classmethod
    def _no_empty_fixture_id(cls, v: list[str]) -> list[str]:
        for fid in v:
            if not fid or not fid.strip():
                raise ValueError("fixture ids must be non-empty strings")
        return v

    @field_validator("criteria")
    @classmethod
    def _criterion_ids_unique(cls, v: list[CriterionSpec]) -> list[CriterionSpec]:
        ids = [c.id for c in v]
        if len(ids) != len(set(ids)):
            raise ValueError("criterion ids must be unique within a scenario")
        return v

    def total_estimated_cost_usd(self) -> float:
        """Sum cost across all fixtures for budget pre-checks."""
        return self.estimated_cost_usd_per_fixture * len(self.fixtures)
