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
    
    Admissibility requirement:
        For general autoregressive sequence generation, log p(y_t | y_<t) <= 0.0.
        Because highly confident transitions can have log p near 0.0 (e.g., -0.001),
        the only universally certified upper bound per step is alpha = 0.0.
        Any negative per-step bound alpha < 0 is heuristic unless mathematically
        guaranteed across all reachable future tokens in the domain.
    """

    def __init__(
        self,
        cost_model: BaseCostModel,
        max_log_prob_upper_bound: float = 0.0,
        allow_heuristic_tighter_bound: bool = False,
    ) -> None:
        self.cost_model = cost_model
        val = min(0.0, float(max_log_prob_upper_bound))
        if val < 0.0 and not allow_heuristic_tighter_bound:
            raise ValueError(
                f"max_log_prob_upper_bound ({val}) < 0.0 is inadmissible for general "
                "language model decoding without a domain proof that all future transitions "
                f"have log p <= {val}. Use 0.0 for certified admissible pruning, or set "
                "allow_heuristic_tighter_bound=True for heuristic search."
            )
        self.max_log_prob_upper_bound = val
        self.allow_heuristic_tighter_bound = allow_heuristic_tighter_bound

    @property
    def is_admissible(self) -> bool:
        """Returns True if the bound is guaranteed admissible in general LM decoding."""
        return self.max_log_prob_upper_bound >= 0.0

    def upper_bound_score(
        self,
        candidate: PrefixCandidate,
        max_steps: int,
        budget: float,
    ) -> float:
        if candidate.is_terminal:
            return 0.0

        remaining_budget = budget - candidate.accumulated_cost
        if remaining_budget < -1e-9:
            return -float("inf")

        min_step = self.cost_model.minimum_step_cost()
        if min_step > 0.0:
            max_possible_tokens = math.floor(max(0.0, remaining_budget) / min_step)
            steps_left = max(0, max_steps - candidate.step)
            horizon = min(max_possible_tokens, steps_left)
            if horizon == 0:
                return 0.0
            return horizon * self.max_log_prob_upper_bound
        return 0.0


class BudgetAwareBound(BaseCompletionBound):
    """Admissible completion bound incorporating resource constraints and step penalties.

    Given remaining resource budget R(u) = B - C(u) and minimum non-EOS step cost c_min > 0,
    the maximum number of non-EOS tokens that can be emitted before exhausting budget is:
        H(u) = floor(R(u) / c_min).
    
    Admissibility theorem:
        For any completion tau of u, S(tau) = S(u) + sum_{t} log p(y_t).
        Since log p(y_t) <= 0, S(tau) <= S(u) + H(u) * alpha is an upper bound if and only if
        alpha >= max_t log p(y_t).
        In general causal language models, max_t log p(y_t) can be arbitrary close to 0.0
        (e.g., confident next tokens with p = 0.999 => log p = -0.001 > -0.05).
        Therefore, alpha = 0.0 is the unique universally admissible default (yielding U(u) = 0.0).
        Setting alpha < 0 is an uncertified heuristic that can underestimate the true completion
        score and prune optimal sequences (see Counterexample 3).
    """

    def __init__(
        self,
        cost_model: BaseCostModel,
        max_log_prob_per_step: float = 0.0,
        allow_heuristic_tighter_bound: bool = False,
    ) -> None:
        self.cost_model = cost_model
        val = min(0.0, float(max_log_prob_per_step))
        if val < 0.0 and not allow_heuristic_tighter_bound:
            raise ValueError(
                f"max_log_prob_per_step ({val}) < 0.0 is inadmissible for general "
                "language model decoding without a domain proof that all future transitions "
                f"have log p <= {val}. Use 0.0 for certified admissible pruning, or set "
                "allow_heuristic_tighter_bound=True for heuristic search."
            )
        self.max_log_prob_per_step = val
        self.allow_heuristic_tighter_bound = allow_heuristic_tighter_bound

    @property
    def is_admissible(self) -> bool:
        """Returns True if the bound is guaranteed admissible in general LM decoding."""
        return self.max_log_prob_per_step >= 0.0

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

