"""Admissible upper bounds on future completion log-probabilities and cost bounds."""

from __future__ import annotations

import math

from dpd.cost_models import BaseCostModel
from dpd.types import PrefixCandidate


class BaseCompletionBound:
    """Base class for admissible future completion bounds."""

    def upper_bound_score(
        self,
        candidate: PrefixCandidate,
        max_steps: int,
        budget: float,
    ) -> float:
        """Returns an admissible upper bound U(u) such that S*(u) <= S(u) + U(u)."""
        raise NotImplementedError


class TrivialAdmissibleBound(BaseCompletionBound):
    """Universal admissible upper bound for normalized token log-probabilities.
    
    Since for any valid language model token, 0 < p_theta <= 1, log(p_theta) <= 0.
    Therefore, any completion tau has score S(tau) <= 0.
    Thus, U(u) = 0.0 is universally admissible.
    """

    def upper_bound_score(
        self,
        candidate: PrefixCandidate,
        max_steps: int,
        budget: float,
    ) -> float:
        return 0.0


class BudgetConditionedBound(BaseCompletionBound):
    """Admissible completion bound incorporating remaining resource budget.
    
    If the minimum cost of any non-EOS token is c_step > 0, then with remaining
    budget R(u) = B - C(u), at most floor(R(u) / c_step) tokens can be generated.
    If an upper bound on single-token log probability is max_log_prob <= 0,
    then U(u) <= floor(R(u) / c_step) * max_log_prob (or 0.0).
    """

    def __init__(self, cost_model: BaseCostModel, max_log_prob_upper_bound: float = 0.0):
        self.cost_model = cost_model
        # max_log_prob_upper_bound should be <= 0
        self.max_log_prob_upper_bound = min(0.0, max_log_prob_upper_bound)

    def upper_bound_score(
        self,
        candidate: PrefixCandidate,
        max_steps: int,
        budget: float,
    ) -> float:
        if candidate.is_terminal:
            return 0.0

        remaining_budget = budget - candidate.accumulated_cost
        if remaining_budget < 0.0:
            return -float("inf")

        min_step = self.cost_model.minimum_step_cost()
        if min_step > 0.0:
            max_possible_tokens = math.floor(remaining_budget / min_step)
            steps_left = max(0, max_steps - candidate.step)
            horizon = min(max_possible_tokens, steps_left)
            # If horizon is 0 and not terminal, must emit EOS (if min_cost_to_eos allows)
            if horizon == 0:
                return 0.0
            return horizon * self.max_log_prob_upper_bound
        return 0.0


class BudgetAwareBound(BaseCompletionBound):
    """Admissible completion bound incorporating resource constraints and step penalties.

    Given remaining resource budget R(u) = B - C(u) and minimum non-EOS step cost c_min > 0,
    the maximum number of non-EOS tokens that can be emitted before exhausting budget is:
        H(u) = floor(R(u) / c_min).
    If an upper bound on single-token log probability is alpha <= 0 (e.g. maximum log probability
    of valid non-EOS transitions), then any completion tau has score bounded by:
        S(tau) <= S(u) + H(u) * alpha.

    When an incumbent solution with score S_inc is known, candidate u can be safely pruned if:
        S(u) + H(u) * alpha <= S_inc.

    Because alpha <= 0, U(u) = H(u) * alpha <= 0 = U_trivial(u), providing a strictly tighter
    admissible bound that triggers pruning earlier than the zero bound.
    """

    def __init__(
        self,
        cost_model: BaseCostModel,
        max_log_prob_per_step: float = -0.05,
    ) -> None:
        self.cost_model = cost_model
        # max_log_prob_per_step must be non-positive to guarantee admissibility
        self.max_log_prob_per_step = min(0.0, float(max_log_prob_per_step))

    def upper_bound_score(
        self,
        candidate: PrefixCandidate,
        max_steps: int,
        budget: float,
    ) -> float:
        """Returns admissible completion upper bound U(u) <= 0.0."""
        if candidate.is_terminal:
            return 0.0

        remaining_budget = budget - candidate.accumulated_cost
        if remaining_budget < -1e-9:
            return -float("inf")

        min_step = self.cost_model.minimum_step_cost()
        if min_step <= 0.0:
            return 0.0

        max_possible_tokens = math.floor(max(0.0, remaining_budget) / min_step)
        steps_left = max(0, max_steps - candidate.step)
        horizon = min(max_possible_tokens, steps_left)

        if horizon <= 0:
            return 0.0

        return horizon * self.max_log_prob_per_step

