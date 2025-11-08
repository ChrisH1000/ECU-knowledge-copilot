"""Tests for the budget manager utility.

Verifies that the BudgetManager correctly tracks LLM call counts,
enforces limits, and allows resets.
"""

import pytest

from utils.budget import BudgetExceededError, BudgetManager


def test_budget_tracks_calls() -> None:
    """Verify that register_call increments counters and enforces limits.

    Tests that:
    - Each register_call() increments call_count
    - remaining decreases accordingly
    - BudgetExceededError is raised when limit is reached
    """
    # Create manager with 2-call limit
    manager = BudgetManager(max_calls=2)

    # Use both allowed calls
    manager.register_call()
    manager.register_call()

    # Verify counters are accurate
    assert manager.call_count == 2
    assert manager.remaining == 0

    # Third call should raise exception
    with pytest.raises(BudgetExceededError):
        manager.register_call()


def test_budget_reset() -> None:
    """Verify that reset() clears the call count.

    Tests that:
    - reset() restores call_count to 0
    - remaining is recalculated correctly after reset
    """
    # Create manager with 1-call limit
    manager = BudgetManager(max_calls=1)

    # Exhaust the budget
    manager.register_call()
    assert manager.call_count == 1

    # Reset should clear counters
    manager.reset()
    assert manager.call_count == 0
    assert manager.remaining == 1