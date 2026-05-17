"""Static price table and per-call cost computation.

Pricing is hand-maintained in this module. Models not present record a
``NULL`` cost rather than guessing — honest is better than wrong.

To update prices: edit ``PRICE_TABLE``, bump the dated comment for each
changed entry, and ship a release. There is no runtime configuration path.
"""

from __future__ import annotations

from decimal import Decimal

# USD per 1,000 tokens, separate input vs output rates.
# Format: model_name -> (input_per_1k, output_per_1k)
#
# Updated: 2026-05-16 (OpenAI list pricing).
# When OpenAI changes pricing, update the relevant entry and the date.
PRICE_TABLE: dict[str, tuple[Decimal, Decimal]] = {
    "gpt-4o": (Decimal("0.005"), Decimal("0.015")),
    "gpt-4o-mini": (Decimal("0.00015"), Decimal("0.0006")),
    "text-embedding-3-small": (Decimal("0.00002"), Decimal("0")),
    "text-embedding-3-large": (Decimal("0.00013"), Decimal("0")),
}


def compute_cost(
    model: str,
    input_tokens: int | None,
    output_tokens: int | None,
) -> Decimal | None:
    """Return the estimated USD cost for a single call, or ``None``.

    Returns ``None`` when:
      * the model is not in ``PRICE_TABLE`` (we refuse to guess), or
      * ``input_tokens`` is ``None`` (the upstream did not report usage).

    ``output_tokens=None`` is treated as ``0`` so embedding-shaped responses
    that omit the output side still produce a cost.
    """
    entry = PRICE_TABLE.get(model)
    if entry is None or input_tokens is None:
        return None
    input_rate, output_rate = entry
    input_cost = input_rate * input_tokens / Decimal(1000)
    output_cost = output_rate * (output_tokens or 0) / Decimal(1000)
    return input_cost + output_cost
