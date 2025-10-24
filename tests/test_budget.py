"""Tests for the budget manager utility."""

import pytest

from utils.budget import BudgetExceededError, BudgetManager


def test_budget_tracks_calls() -> None:
    manager = BudgetManager(max_calls=2)
    manager.register_call()
    manager.register_call()
    assert manager.call_count == 2
    assert manager.remaining == 0
    with pytest.raises(BudgetExceededError):
        manager.register_call()


def test_budget_reset() -> None:
    manager = BudgetManager(max_calls=1)
    manager.register_call()
    assert manager.call_count == 1
    manager.reset()
    assert manager.call_count == 0
    assert manager.remaining == 1