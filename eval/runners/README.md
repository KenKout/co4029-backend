# Capability runners

Each runner is a module here that exports a callable matching:

```python
def run(
    *,
    scenario_inputs: dict[str, Any],
    backend: Literal["old", "new"],
    budget: Budget,
    dry_run: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], int]:
    """Returns (outputs, cost_breakdown, latency_ms)."""
```

The eval engine looks up a runner by the scenario's `capability` field. Runners MUST:

- Call `budget.assert_can_spend(estimated_cost)` BEFORE issuing any LLM call.
- Call `budget.spend(actual_cost)` after each call resolves.
- Honour `dry_run=True` by short-circuiting before any outbound call and returning empty outputs.
- Use the production AI pipeline modules under `abridgeai.features.<capability>.ai` so the eval signal reflects what users actually run.

T8.1 ships no runners. T8.2 ships the first batch.
