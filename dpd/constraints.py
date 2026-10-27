"""Constraint checking and feasibility validation for budget-constrained decoding."""

from __future__ import annotations

from dpd.cost_models import BaseCostModel
from dpd.types import PrefixCandidate


class BudgetConstraintValidator:
    """Evaluates and enforces hard and soft resource budget constraints."""

    def __init__(self, budget: float, cost_model: BaseCostModel, eps: float = 1e-9):
        self.budget = budget
        self.cost_model = cost_model
        self.eps = eps

    def is_prefix_feasible(self, candidate: PrefixCandidate) -> bool:
        """Checks if the prefix itself has not exceeded the total budget."""
        return candidate.accumulated_cost <= self.budget + self.eps

    def can_complete_feasibly(self, candidate: PrefixCandidate) -> bool:
        """Admissible check: whether any feasible completion to EOS can exist.
        
        If candidate is already terminal, returns is_prefix_feasible.
        Otherwise, checks whether accumulated_cost + min_cost_to_eos <= budget.
        """
        if candidate.is_terminal:
            return self.is_prefix_feasible(candidate)

        last_token = candidate.token_ids[-1] if candidate.token_ids else -1
        min_future_cost = self.cost_model.min_cost_to_eos(last_token)
        return (candidate.accumulated_cost + min_future_cost) <= self.budget + self.eps

    def compute_violation(self, candidate: PrefixCandidate) -> float:
        """Returns the non-negative budget violation max(0, Cost - Budget)."""
        return max(0.0, candidate.accumulated_cost - self.budget)

    def filter_feasible_terminals(
        self,
        candidates: list[PrefixCandidate],
    ) -> list[PrefixCandidate]:
        """Filters a list of candidates to only those that are terminal and strictly within budget."""
        return [
            c for c in candidates
            if c.is_terminal and self.is_prefix_feasible(c)
        ]
