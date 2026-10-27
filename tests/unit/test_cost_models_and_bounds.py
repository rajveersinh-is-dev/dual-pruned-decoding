"""Unit tests for cost models and admissible completion bounds."""

import pytest

from dpd.bounds import (
    BudgetAwareBound,
    BudgetConditionedBound,
    TrivialAdmissibleBound,
)
from dpd.cost_models import (
    CallableCostModel,
    CharacterCostModel,
    MonetaryCostModel,
    TokenCountCostModel,
)
from dpd.dominance import filter_by_incumbent_bound
from dpd.types import PrefixCandidate


def test_token_count_cost_model():
    """TokenCountCostModel assigns cost 1.0 to ordinary tokens and 0.0 to zero-cost tokens."""
    cm = TokenCountCostModel(eos_token_id=2, zero_cost_tokens={0, 2})
    assert cm.step_cost("prompt", [], 5) == 1.0
    assert cm.step_cost("prompt", [5], 0) == 0.0
    assert cm.step_cost("prompt", [5], 2) == 0.0
    assert cm.min_cost_to_eos(5) == 0.0  # eos is in zero_cost_tokens


def test_character_cost_model():
    """CharacterCostModel maps tokens to character lengths."""
    char_map = {10: 3, 20: 7, 2: 0}
    cm = CharacterCostModel(token_to_char_len=char_map, eos_token_id=2, default_len=5)
    assert cm.step_cost("p", [], 10) == 3.0
    assert cm.step_cost("p", [10], 20) == 7.0
    assert cm.step_cost("p", [10], 99) == 5.0  # default


def test_monetary_cost_model():
    """MonetaryCostModel applies fixed dollar cost per token."""
    cm = MonetaryCostModel(cost_per_token=0.002)
    assert cm.step_cost("p", [], 10) == 0.002
    assert cm.minimum_step_cost() == 0.002


def test_trivial_admissible_bound():
    """Trivial bound always returns 0.0."""
    b = TrivialAdmissibleBound()
    cand = PrefixCandidate(token_ids=[1, 2], accumulated_score=-3.5, accumulated_cost=2.0, step=2)
    assert b.upper_bound_score(cand, max_steps=10, budget=5.0) == 0.0


def test_budget_conditioned_bound():
    """Budget conditioned bound returns -inf if budget already exceeded."""
    cm = TokenCountCostModel(eos_token_id=2)
    b = BudgetConditionedBound(cost_model=cm, max_log_prob_upper_bound=-0.1)

    cand_over = PrefixCandidate(token_ids=[1], accumulated_score=-1.0, accumulated_cost=10.0, step=1)
    # Budget is 5.0, so cost 10.0 is over budget
    assert b.upper_bound_score(cand_over, max_steps=10, budget=5.0) == -float("inf")

    # Feasible candidate with 3 steps remaining and budget 5 (2 tokens left)
    cand_ok = PrefixCandidate(token_ids=[1], accumulated_score=-1.0, accumulated_cost=3.0, step=1)
    ub = b.upper_bound_score(cand_ok, max_steps=10, budget=5.0)
    # Remaining budget = 2.0, min step = 1.0 => max 2 tokens. Horizon = min(2, 9) = 2.
    # 2 * (-0.1) = -0.2
    assert ub == pytest.approx(-0.2)


def test_budget_aware_bound_tighter_than_trivial():
    """BudgetAwareBound returns strictly negative upper bounds that dominate TrivialAdmissibleBound."""
    cm = TokenCountCostModel(eos_token_id=2)
    b_trivial = TrivialAdmissibleBound()
    b_aware = BudgetAwareBound(cost_model=cm, max_log_prob_per_step=-0.05)

    cand = PrefixCandidate(token_ids=[1], accumulated_score=-1.5, accumulated_cost=2.0, step=1)
    ub_trivial = b_trivial.upper_bound_score(cand, max_steps=10, budget=5.0)
    ub_aware = b_aware.upper_bound_score(cand, max_steps=10, budget=5.0)

    # Remaining budget is 3.0, horizon is min(3, 9) = 3. Upper bound is 3 * (-0.05) = -0.15
    assert ub_trivial == 0.0
    assert ub_aware == pytest.approx(-0.15)
    assert ub_aware < ub_trivial  # Strictly tighter!

    # Terminal candidates have 0.0 remaining bound
    cand_term = PrefixCandidate(token_ids=[1, 2], accumulated_score=-1.5, accumulated_cost=2.0, step=2, is_terminal=True)
    assert b_aware.upper_bound_score(cand_term, max_steps=10, budget=5.0) == 0.0

    # Over budget candidate gets -inf
    cand_over = PrefixCandidate(token_ids=[1], accumulated_score=-1.5, accumulated_cost=6.0, step=1)
    assert b_aware.upper_bound_score(cand_over, max_steps=10, budget=5.0) == -float("inf")


def test_budget_aware_bound_pruning_dominance():
    """Verifies that BudgetAwareBound prunes candidates that TrivialAdmissibleBound misses."""
    cm = TokenCountCostModel(eos_token_id=2)
    b_trivial = TrivialAdmissibleBound()
    b_aware = BudgetAwareBound(cost_model=cm, max_log_prob_per_step=-0.1)

    # Candidate with S = -2.0, cost = 2.0. Budget = 4.0.
    # Remaining budget = 2.0 => horizon = 2 tokens.
    # Under trivial bound: S + U = -2.0 + 0 = -2.0
    # Under aware bound: S + U = -2.0 + 2*(-0.1) = -2.2
    cand = PrefixCandidate(token_ids=[1], accumulated_score=-2.0, accumulated_cost=2.0, step=1)

    # Incumbent with S_inc = -2.1
    survivors_trivial, pruned_trivial = filter_by_incumbent_bound([cand], incumbent_score=-2.1, bound_manager=b_trivial, max_steps=5, budget=4.0)
    survivors_aware, pruned_aware = filter_by_incumbent_bound([cand], incumbent_score=-2.1, bound_manager=b_aware, max_steps=5, budget=4.0)

    assert len(survivors_trivial) == 1
    assert pruned_trivial == 0  # Trivial bound failed to prune candidate

    assert len(survivors_aware) == 0
    assert pruned_aware == 1  # BudgetAware bound successfully pruned candidate!


def test_callable_cost_model():
    """CallableCostModel evaluates arbitrary cost functions and validates non-negativity."""
    cm = CallableCostModel(cost_fn=lambda prompt, prefix, tok: float(tok * 0.5), min_step=0.5)
    assert cm.step_cost(None, [], 4) == 2.0
    assert cm.minimum_step_cost() == 0.5

    # Negative cost should raise ValueError
    bad_cm = CallableCostModel(cost_fn=lambda prompt, prefix, tok: -1.0)
    with pytest.raises(ValueError, match="must be non-negative"):
        bad_cm.step_cost(None, [], 1)


def test_character_cost_model_minimum_cost():
    """CharacterCostModel correctly extracts minimum step cost."""
    cm_empty = CharacterCostModel(token_to_char_len={}, eos_token_id=0)
    assert cm_empty.minimum_step_cost() == 1.0

    cm_filled = CharacterCostModel(token_to_char_len={1: 4, 2: 2, 3: 8}, eos_token_id=0)
    assert cm_filled.minimum_step_cost() == 2.0


