"""Session budget management utilities."""

from __future__ import annotations

from dataclasses import dataclass


class BudgetExceededError(RuntimeError):
    """Raised when the call budget for the session has been exhausted."""


@dataclass
class BudgetManager:
    """Tracks large language model call usage for a session."""

    max_calls: int = 25

    def __post_init__(self) -> None:
        if self.max_calls <= 0:
            raise ValueError("max_calls must be greater than zero")
        self._call_count: int = 0

    @property
    def call_count(self) -> int:
        """Number of calls already registered."""

        return self._call_count

    @property
    def remaining(self) -> int:
        """Number of calls remaining before hitting the budget."""

        return max(self.max_calls - self._call_count, 0)

    def register_call(self) -> None:
        """Record a model invocation and enforce the budget."""

        if self._call_count >= self.max_calls:
            raise BudgetExceededError(
                "Session budget exceeded. Restart the session to continue."
            )
        self._call_count += 1

    def reset(self) -> None:
        """Reset the tracked call count to zero."""

        self._call_count = 0