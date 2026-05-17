"""Base classes for capability runners.

A runner takes one scenario's `inputs` plus a fixture id, drives the
corresponding production AI pipeline, and returns a `RunResult`. T8.2
ships the interface + dry-run skeleton; T8.4 plugs in the real
`abridgeai.features.<capability>.ai` calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from eval.budget import Budget

Backend = Literal["old", "new"]


@dataclass
class RunResult:
    fixture_id: str
    backend: Backend
    outputs: dict[str, Any]
    cost_breakdown: list[dict[str, Any]] = field(default_factory=list)
    latency_ms: int = 0
    errors: list[str] = field(default_factory=list)


class CapabilityRunner:
    """Subclass per capability. Override `run` to call the real pipeline.

    Contract:
    - Call ``budget.assert_can_spend(estimate)`` BEFORE issuing any LLM call.
    - Call ``budget.spend(actual)`` after each call resolves.
    - Honor ``dry_run=True`` by short-circuiting before any outbound call.
    - Return a deterministic `RunResult` shape so the judge stage and
      results JSON serialization stay stable across capabilities.
    """

    capability: str = ""

    def __init__(self, *, budget: Budget, dry_run: bool) -> None:
        self.budget = budget
        self.dry_run = dry_run

    def run(
        self,
        *,
        fixture_id: str,
        backend: Backend,
        inputs: dict[str, Any],
    ) -> RunResult:
        if self.dry_run:
            return RunResult(
                fixture_id=fixture_id,
                backend=backend,
                outputs={},
                cost_breakdown=[],
                latency_ms=0,
                errors=[],
            )
        return RunResult(
            fixture_id=fixture_id,
            backend=backend,
            outputs={},
            cost_breakdown=[],
            latency_ms=0,
            errors=["T8.4 will populate"],
        )
