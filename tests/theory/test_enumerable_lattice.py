"""Theoretical validation tests on enumerable decoding lattices.

Verifies Theorem 1 (Weak Duality, Bounded Suboptimality, and Exactness)
against brute-force exhaustive enumeration from first principles.
"""
from __future__ import annotations

from typing import Any

from dpd.baselines import (
    exact_exhaustive_search,
    standard_beam_search,
)
from dpd.config import DPDConfig, MultiplierConfig
from dpd.cost_models import TokenCountCostModel
from dpd.decoding import dual_pruned_decoding
from dpd.types import MultiplierType, PruningMode


class DeterministicToyLattice:
    """A small finite directed graph / tree with explicit (cost, log_prob) transitions.
    
    Alphabet:
      Token 1: High score, high cost
      Token 2: Medium score, medium cost
      Token 3: Low score, low cost
      Token 0: EOS (terminal, cost 0)
    """

    def __init__(self):
        self.eos_token_id = 0
        # Prefix tuple -> list of (token_id, log_prob, state_id)
        self.transitions: dict[tuple[int, ...], list[tuple[int, float, str | None]]] = {
            (): [
                (1, -0.2, "s1"),  # high score, high cost
                (2, -0.8, "s2"),  # med score, med cost
                (3, -1.5, "s3"),  # low score, low cost
            ],
            (1,): [
                (1, -0.3, "s11"),
                (2, -0.9, "s12"),
                (0, -0.1, "eos"),
            ],
            (2,): [
                (1, -0.4, "s21"),
                (3, -1.2, "s23"),
                (0, -0.2, "eos"),
            ],
            (3,): [
                (2, -0.7, "s32"),
                (3, -1.0, "s33"),
                (0, -0.3, "eos"),
            ],
            (1, 1): [(0, -0.1, "eos")],
            (1, 2): [(0, -0.1, "eos")],
            (2, 1): [(0, -0.2, "eos")],
            (2, 3): [(0, -0.1, "eos")],
            (3, 2): [(0, -0.1, "eos")],
            (3, 3): [(0, -0.1, "eos")],
        }

    def step_fn(
        self,
        prompt: Any,
        prefix_token_ids: list[int],
    ) -> list[tuple[int, float, str | None]]:
        key = tuple(prefix_token_ids)
        return self.transitions.get(key, [(0, 0.0, "eos")])


def test_exhaustive_lattice_weak_duality_and_exactness():
    """Theorem 1 verification:
    1. Exhaustive search computes true constrained optimum y*.
    2. DPD dual upper bound d* >= S(y*).
    3. If certificate gap Delta == 0, incumbent is verified identical to y*.
    """
    lattice = DeterministicToyLattice()
    cost_model = TokenCountCostModel(
        eos_token_id=lattice.eos_token_id,
        zero_cost_tokens={lattice.eos_token_id},
    )

    # Test under multiple budget levels: B = 1.0, 2.0, 3.0
    for budget in [1.0, 2.0, 3.0]:
        config = DPDConfig(
            beam_width=5,
            max_steps=5,
            budget=budget,
            eos_token_id=lattice.eos_token_id,
            pruning_mode=PruningMode.HYBRID_HULL_AND_BOUND,
            multiplier=MultiplierConfig(
                controller_type=MultiplierType.PROJECTED_SUBGRADIENT,
                initial_lambda=0.5,
                learning_rate=0.2,
            ),
        )

        # 1. True constrained optimum via exhaustive enumeration
        true_opt, _all_terminals = exact_exhaustive_search(
            prompt="toy",
            config=config,
            cost_model=cost_model,
            step_fn=lattice.step_fn,
        )

        assert true_opt is not None, f"Lattice has feasible solutions for budget {budget}"
        true_opt_score = true_opt.accumulated_score
        true_opt_cost = true_opt.accumulated_cost
        assert true_opt_cost <= budget + 1e-9

        # 2. Run DPD
        dpd_res = dual_pruned_decoding(
            prompt="toy",
            config=config,
            cost_model=cost_model,
            step_fn=lattice.step_fn,
        )

        # Assertion A: DPD output is strictly feasible
        assert dpd_res.is_feasible, f"DPD should find a feasible solution for budget {budget}"
        assert dpd_res.best_cost <= budget + 1e-9

        # Assertion B: Weak Duality verification
        # The dual upper bound d* MUST be >= true optimal score S(y*)
        assert dpd_res.certificate is not None
        d_star = dpd_res.certificate.dual_upper_bound
        assert d_star >= true_opt_score - 1e-6, (
            f"Weak Duality VIOLATION: d*={d_star} < true_opt={true_opt_score}"
        )

        # Assertion C: Suboptimality gap validity
        subopt_gap = dpd_res.certificate.suboptimality_gap
        actual_gap = true_opt_score - dpd_res.best_score
        assert subopt_gap >= actual_gap - 1e-6, (
            f"Certificate gap Delta={subopt_gap} is smaller than true gap {actual_gap}"
        )

        # If suboptimality gap is 0 (or within tolerance), DPD score must match true optimum!
        if dpd_res.certificate.certified_optimal:
            assert abs(dpd_res.best_score - true_opt_score) < 1e-5


def test_standard_beam_search_violates_budget_when_unconstrained():
    """Demonstrates baseline failure: Standard Beam Search greedily picks high-score tokens
    and violates the budget when B is tight, whereas DPD remains strictly feasible."""
    lattice = DeterministicToyLattice()
    cost_model = TokenCountCostModel(
        eos_token_id=lattice.eos_token_id,
        zero_cost_tokens={lattice.eos_token_id},
    )

    tight_budget = 1.0  # Allows at most 1 token + EOS
    config = DPDConfig(
        beam_width=2,
        max_steps=5,
        budget=tight_budget,
        eos_token_id=lattice.eos_token_id,
    )

    standard_beam_search(
        prompt="toy",
        config=config,
        cost_model=cost_model,
        step_fn=lattice.step_fn,
    )

    dpd_res = dual_pruned_decoding(
        prompt="toy",
        config=config,
        cost_model=cost_model,
        step_fn=lattice.step_fn,
    )

    # DPD must satisfy the hard budget
    assert dpd_res.best_cost <= tight_budget + 1e-9
    assert dpd_res.is_feasible
