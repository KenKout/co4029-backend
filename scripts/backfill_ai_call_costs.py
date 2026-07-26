"""Recompute ``ai_model_calls.estimated_cost_usd`` for rows that have none.

Cost is stamped at call time from ``ai_model_pricing``. A model with no
pricing row records ``NULL`` rather than guessing — correct, but it means
every call made before someone added that model's rate is invisible to the
cost dashboard and to ``rebuild_knowledge_graph.py --budget-usd`` forever,
because nothing ever revisits those rows.

This is the revisit. Run it after adding or correcting a rate:

    uv run --no-sync python scripts/backfill_ai_call_costs.py --dry-run
    uv run --no-sync python scripts/backfill_ai_call_costs.py

Only rows where ``estimated_cost_usd IS NULL`` are touched, so a call priced
at the rate in force when it happened keeps that number — a later rate change
does not silently rewrite history. Pass ``--model`` to restrict the sweep, and
``--repricepriced`` is deliberately not offered: overwriting a stamped cost
would make the audit trail unreproducible.

Rows whose model still has no pricing row are reported and left alone.
"""

from __future__ import annotations

import argparse
import asyncio
from decimal import Decimal

from sqlalchemy import text

from abridgeai.core.db import get_sessionmaker

_SUMMARY_SQL = text(
    """
    SELECT k.model_name                       AS model_name,
           count(*)                           AS calls,
           sum(coalesce(k.input_tokens, 0))   AS input_tokens,
           sum(coalesce(k.output_tokens, 0))  AS output_tokens,
           p.input_usd_per_1m                 AS input_rate,
           p.output_usd_per_1m                AS output_rate
    FROM ai_model_calls k
    LEFT JOIN ai_model_pricing p ON p.model_name = k.model_name
    WHERE k.estimated_cost_usd IS NULL
      AND k.input_tokens IS NOT NULL
      AND (CAST(:model AS text) IS NULL OR k.model_name = CAST(:model AS text))
    GROUP BY k.model_name, p.input_usd_per_1m, p.output_usd_per_1m
    ORDER BY count(*) DESC
    """
)

# Cost is computed in SQL against the CURRENT rate rather than row-by-row in
# Python: one statement, one transaction, and no chance of a partial sweep
# leaving half the rows priced at a rate that changed mid-run.
_BACKFILL_SQL = text(
    """
    UPDATE ai_model_calls k
    SET estimated_cost_usd =
            (coalesce(k.input_tokens, 0)::numeric / 1000000) * p.input_usd_per_1m
          + (coalesce(k.output_tokens, 0)::numeric / 1000000) * p.output_usd_per_1m
    FROM ai_model_pricing p
    WHERE p.model_name = k.model_name
      AND k.estimated_cost_usd IS NULL
      AND k.input_tokens IS NOT NULL
      AND (CAST(:model AS text) IS NULL OR k.model_name = CAST(:model AS text))
    """
)


def _fmt_usd(value: Decimal) -> str:
    return f"${value:,.4f}"


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None, help="restrict to one model_name")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report what would be priced without writing",
    )
    args = parser.parse_args()

    sessionmaker = get_sessionmaker()
    async with sessionmaker() as session:
        rows = (await session.execute(_SUMMARY_SQL, {"model": args.model})).mappings().all()

        if not rows:
            print("Nothing to backfill: no unpriced calls with recorded usage.")
            return 0

        priced_total = Decimal(0)
        priced_calls = 0
        unpriced: list[tuple[str, int]] = []

        print(f"{'model':28} {'calls':>7} {'in tok':>10} {'out tok':>10}  cost")
        for row in rows:
            if row["input_rate"] is None:
                unpriced.append((row["model_name"], row["calls"]))
                print(
                    f"{row['model_name']:28} {row['calls']:>7} {row['input_tokens']:>10} "
                    f"{row['output_tokens']:>10}  -- no pricing row --"
                )
                continue
            cost = (Decimal(row["input_tokens"]) / 1_000_000) * row["input_rate"] + (
                Decimal(row["output_tokens"]) / 1_000_000
            ) * row["output_rate"]
            priced_total += cost
            priced_calls += row["calls"]
            print(
                f"{row['model_name']:28} {row['calls']:>7} {row['input_tokens']:>10} "
                f"{row['output_tokens']:>10}  {_fmt_usd(cost)}"
            )

        print(f"\n{priced_calls} call(s) can be priced, totalling {_fmt_usd(priced_total)}.")
        if unpriced:
            print(
                "\nStill unpriced — add a rate via the admin pricing API, then re-run:"
            )
            for model_name, calls in unpriced:
                print(f"  {model_name}  ({calls} call(s))")

        if args.dry_run:
            print("\n--dry-run: nothing written.")
            return 0

        result = await session.execute(_BACKFILL_SQL, {"model": args.model})
        await session.commit()
        print(f"\nUpdated {result.rowcount} row(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
