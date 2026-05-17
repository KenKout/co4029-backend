"""Budget tracking and enforcement for the eval runner.

Hard ceiling. No override. Every LLM call must pass through `Budget.spend()`,
which raises `BudgetExceededError` the instant the next call would push the
running total above `limit_usd`.
"""

from __future__ import annotations

from dataclasses import dataclass


class BudgetExceededError(Exception):
    """Raised when a spend would push the running total above the limit."""


@dataclass
class Budget:
    limit_usd: float
    spent_usd: float = 0.0

    def assert_can_spend(self, additional_usd: float) -> None:
        if additional_usd < 0:
            raise ValueError(f"additional_usd must be non-negative, got {additional_usd}")
        if self.spent_usd + additional_usd > self.limit_usd:
            raise BudgetExceededError(
                f"would exceed budget: spent=${self.spent_usd:.4f}, "
                f"adding=${additional_usd:.4f}, limit=${self.limit_usd:.4f}"
            )

    def spend(self, amount_usd: float) -> None:
        self.assert_can_spend(amount_usd)
        self.spent_usd += amount_usd

    def remaining_usd(self) -> float:
        return max(0.0, self.limit_usd - self.spent_usd)
