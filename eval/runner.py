"""Eval scenario execution engine.

Discovers scenario YAMLs under `scenarios/`, dispatches each to a
capability-specific runner under `runners/`, passes outputs through an
LLM-as-judge prompt under `judges/prompts/`, and accumulates per-scenario
results.

T8.1 ships scaffold only: scenario discovery returns empty list when
`scenarios/` has no YAMLs, and `run()` exits cleanly with `[]`. T8.2 fills
in the actual capability runners and scenario fixtures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from eval.budget import Budget


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
    ) -> None:
        self.scenarios_dir = scenarios_dir
        self.scenarios_filter = scenarios_filter
        self.backend = backend
        self.judge_model = judge_model
        self.budget = budget
        self.dry_run = dry_run

    def discover_scenarios(self) -> list[Path]:
        if not self.scenarios_dir.exists():
            return []
        all_yamls = sorted(self.scenarios_dir.glob("*.yaml"))
        if self.scenarios_filter is None:
            return all_yamls
        wanted = set(self.scenarios_filter)
        return [p for p in all_yamls if p.stem in wanted]

    def run(self) -> list[ScenarioResult]:
        scenarios = self.discover_scenarios()
        if not scenarios:
            return []
        # T8.2 will dispatch to capability-specific runners; T8.3 will add
        # LLM-as-judge scoring. For T8.1 the scaffold acknowledges discovery
        # but does not execute (no runners registered yet).
        return [
            ScenarioResult(
                scenario_name=path.stem,
                inputs={},
                outputs_old=None,
                outputs_new=None,
                judge_scores_old=None,
                judge_scores_new=None,
                errors=["scaffold: no runner registered (T8.2)"],
            )
            for path in scenarios
        ]
