"""Session budget management utilities.

This module implements cost control for the RAG chatbot by limiting the
number of LLM calls per session (default: 25 as per PRD).
"""

from __future__ import annotations

from dataclasses import dataclass


class BudgetExceededError(RuntimeError):
    """Raised when the call budget for the session has been exhausted.

    Signals that the user should restart the session if they want to
    continue asking questions.
    """


@dataclass
class BudgetManager:
    """Tracks large language model call usage for a session.

    Enforces a maximum number of LLM invocations (PRD specifies 25)
    to keep costs predictable. The budget is checked before each
    non-cached query.

    Attributes:
        max_calls: Maximum LLM invocations allowed (default 25)
    """

    max_calls: int = 25

    def __post_init__(self) -> None:
        """Validate configuration and initialize call counter.

        Raises:
            ValueError: If max_calls is not positive
        """
        if self.max_calls <= 0:
            raise ValueError("max_calls must be greater than zero")
        # Track actual call count (starts at zero)
        self._call_count: int = 0

    @property
    def call_count(self) -> int:
        """Number of calls already registered.

        Returns:
            int: Current call count
        """
        return self._call_count

    @property
    def remaining(self) -> int:
        """Number of calls remaining before hitting the budget.

        Returns:
            int: Remaining calls (never negative)
        """
        return max(self.max_calls - self._call_count, 0)

    def register_call(self) -> None:
        """Record a model invocation and enforce the budget.

        Increments the internal counter and raises an exception if
        the budget has already been exhausted.

        Raises:
            BudgetExceededError: If max_calls has been reached
        """
        if self._call_count >= self.max_calls:
            raise BudgetExceededError(
                "Session budget exceeded. Restart the session to continue."
            )
        self._call_count += 1

    def reset(self) -> None:
        """Reset the tracked call count to zero.

        Useful for testing or if you want to manually reset the session
        without restarting the application.
        """
        self._call_count = 0