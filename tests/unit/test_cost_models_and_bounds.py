"""Unit tests for cost models and admissible completion bounds."""

import pytest

from dpd.bounds import BudgetConditionedBound, TrivialAdmissibleBound
from dpd.cost_models import (
    CharacterCostModel,
    MonetaryCostModel,
    TokenCountCostModel,
)
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
