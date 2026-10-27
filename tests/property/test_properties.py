"""Property-based randomized invariant tests for DPD."""

import random

import numpy as np

from dpd.config import DPDConfig
from dpd.cost_models import TokenCountCostModel
from dpd.decoding import dual_pruned_decoding
from dpd.hull import compute_upper_left_hull, get_hull_slopes
from dpd.types import CostScorePoint, PruningMode


def test_property_hull_strictly_decreasing_slopes():
    """Property: For ANY randomized collection of 2D points, the upper-left
    convex hull vertices always have strictly decreasing transition slopes."""
    rng = np.random.default_rng(12345)
    for _ in range(50):
        n = rng.integers(3, 100)
        costs = rng.uniform(0.0, 50.0, size=n)
        scores = rng.uniform(-100.0, 0.0, size=n)
        points = [
            CostScorePoint(cost=float(c), score=float(s), candidate_id=i)
            for i, (c, s) in enumerate(zip(costs, scores))
        ]
        hull = compute_upper_left_hull(points)
        slopes = get_hull_slopes(hull)
        for i in range(len(slopes) - 1):
            assert slopes[i] > slopes[i + 1] - 1e-9, (
                f"Slopes not strictly decreasing: {slopes[i]} <= {slopes[i+1]}"
            )


def test_property_hard_budget_invariant():
    """Property: If is_feasible is True, the returned sequence cost NEVER exceeds the budget."""
    rng = random.Random(42)

    def random_step_fn(prompt, prefix):
        if len(prefix) >= 8:
            return [(0, 0.0, "eos")]
        # Generate 4 random tokens with random log-probs
        toks = []
        for tok_id in [1, 2, 3, 0]:
            lp = -rng.uniform(0.1, 3.0)
            toks.append((tok_id, lp, f"state_{tok_id}"))
        return toks

    cost_model = TokenCountCostModel(eos_token_id=0)

    for budget in [2.0, 4.0, 6.0, 10.0]:
        cfg = DPDConfig(
            beam_width=4,
            max_steps=10,
            budget=budget,
            eos_token_id=0,
            pruning_mode=PruningMode.HYBRID_HULL_AND_BOUND,
        )
        res = dual_pruned_decoding("prompt", cfg, cost_model, random_step_fn)
        if res.is_feasible:
            assert res.best_cost <= budget + 1e-9, (
                f"Hard budget violation: cost {res.best_cost} > budget {budget}"
            )


def test_property_deterministic_reproducibility():
    """Property: DPD is 100% deterministic given identical inputs."""
    def step_fn(prompt, prefix):
        if len(prefix) >= 5:
            return [(0, 0.0, "eos")]
        return [(1, -0.5, "s1"), (2, -1.2, "s2"), (0, -0.8, "eos")]

    cost_model = TokenCountCostModel(eos_token_id=0)
    cfg = DPDConfig(beam_width=3, max_steps=6, budget=3.0, eos_token_id=0)

    res1 = dual_pruned_decoding("test", cfg, cost_model, step_fn)
    res2 = dual_pruned_decoding("test", cfg, cost_model, step_fn)

    assert res1.best_sequence == res2.best_sequence
    assert res1.best_score == res2.best_score
    assert res1.best_cost == res2.best_cost
    assert res1.is_feasible == res2.is_feasible
