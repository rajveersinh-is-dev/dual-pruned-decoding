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


def test_property_dual_certificate_bounds_incumbent():
    """Property (Weak Duality): For any feasible solution, d* >= S(y_inc) - 1e-6."""
    for seed in range(200, 220):
        def step_fn(prompt, prefix):
            if len(prefix) >= 5:
                return [(0, 0.0, "eos")]
            prefix_seed = seed * 100003 + hash(tuple(prefix)) % 1000003
            local_rng = random.Random(prefix_seed)
            branches = []
            for tok in [1, 2, 0]:
                lp = -local_rng.uniform(0.1, 2.0)
                branches.append((tok, lp, f"s_{tok}"))
            return branches

        cost_model = TokenCountCostModel(eos_token_id=0)
        cfg = DPDConfig(
            beam_width=4,
            max_steps=6,
            budget=3.0,
            eos_token_id=0,
            pruning_mode=PruningMode.HYBRID_HULL_AND_BOUND,
        )
        res = dual_pruned_decoding("test", cfg, cost_model, step_fn)
        if res.is_feasible and res.certificate is not None:
            assert res.certificate.dual_upper_bound >= res.best_score - 1e-6, (
                f"Weak duality violated at seed {seed}: d*={res.certificate.dual_upper_bound} < S={res.best_score}"
            )
            assert res.certificate.suboptimality_gap >= -1e-6
            if res.certificate.certified_optimal:
                assert res.certificate.suboptimality_gap <= 1e-4


def test_property_certified_optimal_rate_on_small_exhaustive_lattice():
    """Property: On small finite lattices where DPD beam covers the active space,
    DPD identifies the true global constrained optimum and achieves >= 50%
    certified optimality rate across diverse random seeds."""
    from dpd.baselines import exact_exhaustive_search

    certified_count = 0
    total_instances = 20
    cost_model = TokenCountCostModel(eos_token_id=0)

    for seed in range(300, 300 + total_instances):
        def step_fn(prompt, prefix):
            if len(prefix) >= 3:
                return [(0, 0.0, "eos")]
            prefix_seed = seed * 100003 + hash(tuple(prefix)) % 1000003
            local_rng = random.Random(prefix_seed)
            branches = []
            for tok in [1, 2, 0]:
                lp = -local_rng.uniform(0.2, 1.5)
                branches.append((tok, lp, f"s_{tok}"))
            return branches

        cfg = DPDConfig(
            beam_width=12,
            max_steps=5,
            budget=2.0,
            eos_token_id=0,
            pruning_mode=PruningMode.HYBRID_HULL_AND_BOUND,
        )
        res = dual_pruned_decoding("test", cfg, cost_model, step_fn)
        true_opt, _ = exact_exhaustive_search("test", cfg, cost_model, step_fn)

        if true_opt is not None and res.is_feasible:
            assert abs(res.best_score - true_opt.accumulated_score) < 1e-6, (
                f"DPD failed to find true constrained optimum at seed {seed}"
            )
        if res.certificate is not None and res.certificate.certified_optimal:
            certified_count += 1

    rate = certified_count / total_instances
    assert rate >= 0.50, f"Expected >= 50% certified optimal rate on small lattices, got {rate * 100:.1f}%"


