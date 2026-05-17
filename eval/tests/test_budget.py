from __future__ import annotations

import pytest

from eval.budget import Budget, BudgetExceededError


def test_assert_can_spend_succeeds_within_limit() -> None:
    b = Budget(limit_usd=5.00)
    b.assert_can_spend(2.50)


def test_assert_can_spend_raises_if_exceeds() -> None:
    b = Budget(limit_usd=5.00, spent_usd=4.50)
    with pytest.raises(BudgetExceededError):
        b.assert_can_spend(1.00)


def test_assert_can_spend_at_exact_limit_succeeds() -> None:
    b = Budget(limit_usd=5.00, spent_usd=4.50)
    b.assert_can_spend(0.50)


def test_assert_can_spend_rejects_negative() -> None:
    b = Budget(limit_usd=5.00)
    with pytest.raises(ValueError, match="non-negative"):
        b.assert_can_spend(-0.01)


def test_spend_accumulates() -> None:
    b = Budget(limit_usd=5.00)
    b.spend(1.00)
    b.spend(2.00)
    assert b.spent_usd == 3.00
    assert b.remaining_usd() == 2.00


def test_spend_raises_at_boundary() -> None:
    b = Budget(limit_usd=5.00, spent_usd=4.99)
    with pytest.raises(BudgetExceededError):
        b.spend(0.02)


def test_spend_does_not_advance_when_blocked() -> None:
    b = Budget(limit_usd=5.00, spent_usd=4.99)
    with pytest.raises(BudgetExceededError):
        b.spend(0.02)
    assert b.spent_usd == 4.99


def test_remaining_clamped_to_zero() -> None:
    b = Budget(limit_usd=5.00, spent_usd=5.00)
    assert b.remaining_usd() == 0.0


def test_zero_budget_blocks_any_positive_spend() -> None:
    b = Budget(limit_usd=0.0)
    with pytest.raises(BudgetExceededError):
        b.spend(0.01)


def test_zero_budget_allows_zero_spend() -> None:
    b = Budget(limit_usd=0.0)
    b.spend(0.0)
    assert b.spent_usd == 0.0
