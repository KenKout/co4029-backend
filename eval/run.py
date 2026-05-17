"""Manual eval runner CLI.

Usage:
    uv run python -m eval.run --help
    uv run python -m eval.run --scenarios all --backend new --budget-usd 5.00
    uv run python -m eval.run --scenarios quiz_generation --backend both --judge-model gpt-4o-mini
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from eval.budget import Budget, BudgetExceededError
from eval.runner import EvalRunner, ScenarioResult


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eval.run",
        description=(
            "Manual real-LLM eval runner. NOT part of CI. Hard $ ceiling enforced via --budget-usd."
        ),
    )
    parser.add_argument(
        "--scenarios",
        default="all",
        help="Comma-separated scenario names, or 'all' (default: all).",
    )
    parser.add_argument(
        "--backend",
        choices=["old", "new", "both"],
        default="new",
        help="Which backend(s) to evaluate (default: new).",
    )
    parser.add_argument(
        "--budget-usd",
        type=float,
        required=True,
        help=(
            "Hard $ ceiling. Runner refuses to start if budget would be "
            "exceeded by estimated cost, and aborts mid-flight if a spend "
            "would push running total over the ceiling."
        ),
    )
    parser.add_argument(
        "--judge-model",
        default="gpt-4o-mini",
        help=(
            "LLM-as-judge model id. MUST be smaller/cheaper than the "
            "subject model so the judge stays independent (default: "
            "gpt-4o-mini)."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path; default eval/results/run-<timestamp>.json.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print scenarios + estimated cost; do not call LLMs.",
    )
    return parser


def _serialize_result(result: ScenarioResult) -> dict[str, Any]:
    return dataclasses.asdict(result)


def _serialize_run(
    *,
    run_id: str,
    started_at: datetime,
    finished_at: datetime,
    args: argparse.Namespace,
    budget: Budget,
    results: list[ScenarioResult],
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "budget_usd": budget.limit_usd,
        "spent_usd": budget.spent_usd,
        "backend": args.backend,
        "judge_model": args.judge_model,
        "dry_run": args.dry_run,
        "results": [_serialize_result(r) for r in results],
    }


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    started_at = datetime.now(UTC)
    run_id = started_at.strftime("%Y%m%dT%H%M%SZ")
    output: Path = args.output or Path("eval/results") / f"run-{run_id}.json"

    if args.budget_usd < 0:
        print(
            f"ERROR: --budget-usd must be non-negative, got {args.budget_usd}",
            file=sys.stderr,
        )
        return 2

    budget = Budget(limit_usd=args.budget_usd)
    if args.scenarios == "all":
        scenarios_filter: list[str] | None = None
    else:
        scenarios_filter = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    runner = EvalRunner(
        scenarios_dir=Path("eval/scenarios"),
        scenarios_filter=scenarios_filter,
        backend=args.backend,
        judge_model=args.judge_model,
        budget=budget,
        dry_run=args.dry_run,
    )

    try:
        results = runner.run()
    except BudgetExceededError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    finished_at = datetime.now(UTC)
    payload = _serialize_run(
        run_id=run_id,
        started_at=started_at,
        finished_at=finished_at,
        args=args,
        budget=budget,
        results=results,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, default=str))

    if not results:
        print("No scenarios discovered. T8.1 ships scaffold only — see eval/scenarios/README.md.")
    print(f"Results written to {output}")
    print(f"Total cost: ${budget.spent_usd:.4f} / ${budget.limit_usd:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
