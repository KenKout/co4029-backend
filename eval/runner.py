"""Eval scenario discovery + dispatch.

Discovers scenario YAMLs under ``scenarios/``, validates each against
:class:`eval.spec.ScenarioSpec`, and dispatches to a capability-specific
runner from :data:`eval.runners.REGISTRY`. Outputs flow through the
LLM-as-judge prompts in ``judges/prompts/`` (T8.3) before being
accumulated into per-scenario results.

T8.2 ships scenario discovery + Pydantic validation + dry-run summary.
T8.4 will populate ``ScenarioResult.outputs_*`` with real LLM output by
having each runner drive the production AI pipeline.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from eval.budget import Budget
from eval.runners import REGISTRY
from eval.spec import ScenarioSpec


@dataclass
class ScenarioResult:
    scenario_name: str
    inputs: dict[str, Any]
    outputs_old: dict[str, Any] | None
    outputs_new: dict[str, Any] | None
    judge_scores_old: dict[str, float] | None
    judge_scores_new: dict[str, float] | None
    cost_breakdown: list[dict[str, Any]] = field(default_factory=list)
    latency_ms_old: int | None = None
    latency_ms_new: int | None = None
    errors: list[str] = field(default_factory=list)


class EvalRunner:
    def __init__(
        self,
        scenarios_dir: Path,
        scenarios_filter: list[str] | None,
        backend: str,
        judge_model: str,
        budget: Budget,
        dry_run: bool,
        fixtures_dir: Path | None = None,
    ) -> None:
        self.scenarios_dir = scenarios_dir
        self.scenarios_filter = scenarios_filter
        self.backend = backend
        self.judge_model = judge_model
        self.budget = budget
        self.dry_run = dry_run
        self.fixtures_dir = fixtures_dir or scenarios_dir.parent / "fixtures"

    def discover_scenarios(self) -> list[Path]:
        if not self.scenarios_dir.exists():
            return []
        all_yamls = sorted(self.scenarios_dir.glob("*.yaml"))
        if self.scenarios_filter is None:
            return all_yamls
        wanted = set(self.scenarios_filter)
        return [p for p in all_yamls if p.stem in wanted]

    def load_specs(self) -> list[tuple[Path, ScenarioSpec | None, str | None]]:
        loaded: list[tuple[Path, ScenarioSpec | None, str | None]] = []
        for path in self.discover_scenarios():
            try:
                raw = yaml.safe_load(path.read_text())
            except yaml.YAMLError as exc:
                loaded.append((path, None, f"YAML parse error: {exc}"))
                continue
            if not isinstance(raw, dict):
                loaded.append((path, None, "scenario YAML must be a mapping"))
                continue
            try:
                spec = ScenarioSpec.model_validate(raw)
            except ValidationError as exc:
                loaded.append((path, None, f"scenario validation failed: {exc}"))
                continue
            loaded.append((path, spec, None))
        return loaded

    def run(self) -> list[ScenarioResult]:
        loaded = self.load_specs()
        if not loaded:
            return []

        if self.dry_run:
            self._print_dry_run_summary(loaded)

        results: list[ScenarioResult] = []
        for path, spec, err in loaded:
            if spec is None or err is not None:
                results.append(
                    ScenarioResult(
                        scenario_name=path.stem,
                        inputs={},
                        outputs_old=None,
                        outputs_new=None,
                        judge_scores_old=None,
                        judge_scores_new=None,
                        errors=[err or "unknown scenario load error"],
                    )
                )
                continue

            runner_errors = self._validate_runtime(spec)
            results.append(
                ScenarioResult(
                    scenario_name=spec.scenario_id,
                    inputs=dict(spec.inputs),
                    outputs_old=None,
                    outputs_new=None,
                    judge_scores_old=None,
                    judge_scores_new=None,
                    errors=runner_errors or ["T8.4 will populate"],
                )
            )
        return results

    def _validate_runtime(self, spec: ScenarioSpec) -> list[str]:
        errors: list[str] = []
        if spec.capability not in REGISTRY:
            errors.append(
                f"unknown capability {spec.capability!r}; "
                f"register a runner in eval.runners.REGISTRY"
            )
        for fixture_id in spec.fixtures:
            if not self._fixture_exists(fixture_id):
                errors.append(f"fixture {fixture_id!r} not found in {self.fixtures_dir}")
        return errors

    def _fixture_exists(self, fixture_id: str) -> bool:
        if not self.fixtures_dir.exists():
            return False
        return any(self.fixtures_dir.glob(f"{fixture_id}.*"))

    def _print_dry_run_summary(
        self, loaded: list[tuple[Path, ScenarioSpec | None, str | None]]
    ) -> None:
        valid = [(p, s) for p, s, e in loaded if s is not None and e is None]
        invalid = [(p, e) for p, s, e in loaded if s is None or e is not None]
        total_combinations = sum(len(s.fixtures) for _, s in valid)
        total_cost = sum(s.total_estimated_cost_usd() for _, s in valid)
        print(  # noqa: T201
            f"{len(valid)} scenarios, {total_combinations} fixture combinations, "
            f"est. cost ${total_cost:.2f}",
            file=sys.stdout,
        )
        for path, spec in valid:
            print(  # noqa: T201
                f"  - {spec.scenario_id} [{spec.capability}] "
                f"fixtures={len(spec.fixtures)} "
                f"est=${spec.total_estimated_cost_usd():.2f} "
                f"({path.name})",
                file=sys.stdout,
            )
        for path, err in invalid:
            print(f"  ! {path.name}: {err}", file=sys.stderr)  # noqa: T201
