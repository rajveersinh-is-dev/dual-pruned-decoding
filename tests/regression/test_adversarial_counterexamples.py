"""Adversarial regression tests covering failure modes and counterexamples."""
from __future__ import annotations

from typing import Any

import pytest

from dpd.config import DPDConfig
from dpd.cost_models import TokenCountCostModel
from dpd.decoding import dual_pruned_decoding
from dpd.types import PruningMode


class Counterexample1Lattice:
    """Implements Counterexample 1 from research/counterexamples.md:
    At step 1:
      A: (Cost 2, Score -1.0)
      B: (Cost 0, Score -2.0)
      C: (Cost 1, Score -1.8) -> strictly below segment AB in (C, S) space!
    Completions:
      A -> Cost +2, Score 0 -> Total Cost 4 (Infeasible for B=3)
      B -> Cost +1, Score -10 -> Total Cost 1, Total Score -12.0
      C -> Cost +2, Score -0.1 -> Total Cost 3, Total Score -1.9 (Optimal feasible!)
    """

    def __init__(self):
        self.eos_token_id = 0

    def step_fn(
        self,
        prompt: Any,
        prefix_token_ids: list[int],
    ) -> list[tuple[int, float, str | None]]:
        if not prefix_token_ids:
            return [
                (1, -1.0, "A"),  # token 1 = A
                (2, -2.0, "B"),  # token 2 = B
                (3, -1.8, "C"),  # token 3 = C
            ]
        first = prefix_token_ids[0]
        if len(prefix_token_ids) == 1:
            if first == 1:  # from A
                return [(10, 0.0, "A_term")]
            elif first == 2:  # from B
                return [(20, -10.0, "B_term")]
            elif first == 3:  # from C
                return [(30, -0.1, "C_term")]
        # At step 2, emit EOS
        return [(0, 0.0, "eos")]


class Counterexample1CostModel(TokenCountCostModel):
    """Custom step costs for Counterexample 1:
    Step 1:
      token 1 (A) costs 2.0
      token 2 (B) costs 0.0
      token 3 (C) costs 1.0
    Step 2:
      token 10 (A_term) costs 2.0
      token 20 (B_term) costs 1.0
      token 30 (C_term) costs 2.0
    EOS costs 0.0
    """

    def __init__(self, eos_token_id: int = 0):
        super().__init__(eos_token_id=eos_token_id, zero_cost_tokens={eos_token_id})

    def step_cost(self, prompt: Any, prefix_token_ids: list[int], next_token_id: int) -> float:
        if next_token_id == 0:
            return 0.0
        if not prefix_token_ids:
            if next_token_id == 1:
                return 2.0
            elif next_token_id == 2:
                return 0.0
            elif next_token_id == 3:
                return 1.0
        else:
            if next_token_id == 10:
                return 2.0
            elif next_token_id == 20:
                return 1.0
            elif next_token_id == 30:
                return 2.0
        return 1.0


def test_counterexample_1_hull_pruning_vs_hybrid():
    """Verifies that naive static hull pruning discards candidate C,
    while HYBRID_HULL_AND_BOUND or EXACT_BOUND_AND_DOMINANCE retains candidate C."""
    lattice = Counterexample1Lattice()
    cost_model = Counterexample1CostModel(eos_token_id=0)

    # 1. Test HULL_EXTREMES_ONLY with beam_width = 2
    # At step 1, candidates are A(2, -1.0), B(0, -2.0), C(1, -1.8).
    # Upper-left hull vertices are only A and B. C is interior.
    # Therefore HULL_EXTREMES_ONLY drops C and picks B (score -12.0).
    cfg_hull_only = DPDConfig(
        beam_width=2,
        max_steps=5,
        budget=3.0,
        eos_token_id=0,
        pruning_mode=PruningMode.HULL_EXTREMES_ONLY,
    )
    res_hull_only = dual_pruned_decoding("test", cfg_hull_only, cost_model, lattice.step_fn)
    # The feasible output is B (score -12.0)
    assert res_hull_only.is_feasible
    assert res_hull_only.best_score == pytest.approx(-12.0)

    # 2. Test HYBRID_HULL_AND_BOUND with beam_width = 3
    # Beam width accommodates C as well, finding the true global optimum C (score -1.9).
    cfg_hybrid = DPDConfig(
        beam_width=3,
        max_steps=5,
        budget=3.0,
        eos_token_id=0,
        pruning_mode=PruningMode.HYBRID_HULL_AND_BOUND,
    )
    res_hybrid = dual_pruned_decoding("test", cfg_hybrid, cost_model, lattice.step_fn)
    assert res_hybrid.is_feasible
    assert res_hybrid.best_score == pytest.approx(-1.9)
    assert res_hybrid.best_cost == pytest.approx(3.0)


def test_zero_budget_boundary():
    """Zero budget B=0: only immediate zero-cost terminations are feasible."""
    class ZeroBudgetLattice:
        def __init__(self):
            self.eos_token_id = 0

        def step_fn(self, prompt, prefix):
            if not prefix:
                return [(0, -0.5, "eos"), (1, -0.1, "tok1")]
            return [(0, 0.0, "eos")]

    lattice = ZeroBudgetLattice()
    cost_model = TokenCountCostModel(eos_token_id=0, zero_cost_tokens={0})  # EOS costs 0, tok1 costs 1

    cfg = DPDConfig(beam_width=2, max_steps=3, budget=0.0, eos_token_id=0)
    res = dual_pruned_decoding("test", cfg, cost_model, lattice.step_fn)
    assert res.is_feasible
    assert res.best_cost == 0.0
    assert res.best_sequence == [0]


def test_infeasible_lattice_reports_infeasibility_cleanly():
    """When no path in the lattice satisfies the budget, DPD must declare infeasible."""
    class InfeasibleLattice:
        def __init__(self):
            self.eos_token_id = 0

        def step_fn(self, prompt, prefix):
            if not prefix:
                return [(1, -0.1, "tok1")]  # Cost 10.0
            return [(0, 0.0, "eos")]

    class ExpensiveCostModel(TokenCountCostModel):
        def step_cost(self, prompt, prefix, token):
            return 10.0

    lattice = InfeasibleLattice()
    cost_model = ExpensiveCostModel(eos_token_id=0)

    cfg = DPDConfig(beam_width=2, max_steps=3, budget=5.0, eos_token_id=0)
    res = dual_pruned_decoding("test", cfg, cost_model, lattice.step_fn)

    # Must report is_feasible = False
    assert not res.is_feasible
    assert res.termination_reason in ("infeasibility_pruned", "infeasible_fallback")
    assert res.certificate is not None
    assert not res.certificate.is_feasible


class Counterexample3Lattice:
    """Counterexample 3: Inadmissible per-step log-prob bound prunes optimal completion.

    Step 1:
      Branch v (token 0, EOS): S = -1.05, cost = 1.0, is_terminal = True.
        Becomes incumbent immediately at step 1: S_inc = -1.05, C = 1.0 <= 3.0.
      Branch u (token 1): S = -1.0, cost = 1.0, is_terminal = False.
        Completion: token 11 (cost 1.0, log p = -0.01) -> token 0 (EOS, cost 1.0, log p = -0.01).
        Total score S* = -1.0 - 0.01 - 0.01 = -1.02.
        Total cost C = 1.0 + 1.0 + 1.0 = 3.0 <= 3.0 (Feasible).
        Optimal feasible sequence! S* = -1.02 > S_inc = -1.05.

    If alpha = -0.05 is assumed:
      At step 1, remaining budget for u is 3.0 - 1.0 = 2.0.
      Horizon H(u) = floor(2.0 / 1.0) = 2 steps.
      Claimed upper bound U(u) = 2 * (-0.05) = -0.10.
      S(u) + U(u) = -1.0 - 0.10 = -1.10 <= S_inc = -1.05.
      Candidate u is falsely pruned at step 1!
    """

    def __init__(self):
        self.eos_token_id = 0

    def step_fn(self, prompt: Any, prefix: list[int]) -> list[tuple[int, float, str | None]]:
        if not prefix:
            # Step 1: branch v (token 0, EOS) and branch u (token 1)
            return [
                (0, -1.05, "eos"),  # v: immediate terminal incumbent
                (1, -1.0, "u"),     # u: active prefix to optimal sequence
            ]
        if prefix == [1]:
            # u continues with high-confidence token (log p = -0.01 > -0.05)
            return [(11, -0.01, "u1")]
        if prefix == [1, 11]:
            # u completes with EOS (log p = -0.01 > -0.05)
            return [(0, -0.01, "eos")]
        return [(0, 0.0, "eos")]


def test_counterexample_3_inadmissible_bound_discards_optimum():
    """Verifies that alpha=-0.05 prunes the optimal path, while alpha=0.0 preserves it."""
    from dpd.bounds import BudgetAwareBound

    lattice = Counterexample3Lattice()
    cost_model = TokenCountCostModel(eos_token_id=0)  # Each token costs 1.0
    budget = 3.0

    # 1. Run with inadmissible heuristic bound alpha = -0.05
    b_inadmissible = BudgetAwareBound(
        cost_model, max_log_prob_per_step=-0.05, allow_heuristic_tighter_bound=True
    )
    cfg = DPDConfig(
        beam_width=5,
        max_steps=5,
        budget=budget,
        eos_token_id=0,
        pruning_mode=PruningMode.EXACT_BOUND_AND_DOMINANCE,
    )
    res_inadmissible = dual_pruned_decoding(
        "test", cfg, cost_model, lattice.step_fn, bound_manager=b_inadmissible
    )

    # Inadmissible bound pruned u at step 1 and settled for v (score -1.05)
    assert res_inadmissible.best_sequence == [0]
    assert res_inadmissible.best_score == pytest.approx(-1.05)

    # 2. Run with certified admissible default bound alpha = 0.0
    b_admissible = BudgetAwareBound(cost_model, max_log_prob_per_step=0.0)
    res_admissible = dual_pruned_decoding(
        "test", cfg, cost_model, lattice.step_fn, bound_manager=b_admissible
    )

    # Admissible bound correctly finds the true global optimum u (score -1.02)
    assert res_admissible.best_sequence == [1, 11, 0]
    assert res_admissible.best_score == pytest.approx(-1.02)
    assert res_admissible.best_score > res_inadmissible.best_score
