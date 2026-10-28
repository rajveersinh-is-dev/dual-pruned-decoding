"""Adversarial regression tests covering failure modes and counterexamples."""
from __future__ import annotations

from typing import Any

import pytest

from dpd.config import DPDConfig
from dpd.cost_models import TokenCountCostModel
from dpd.decoding import dual_pruned_decoding
from dpd.types import CertificateScope, PrefixCandidate, PruningMode, SuccessorCompleteness


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


def test_counterexample_4_top_k_truncation_discards_optimum():
    """Counterexample 4: Restricting step_fn to top-k (e.g. k=1) discards the true optimum.

    Search space:
      From root:
        token 1 (rank 1): log p = -0.1, step cost = 5.0 (exceeds budget B=3.0)
        token 2 (rank 2): log p = -0.2, step cost = 1.0 (completion: tok 0, EOS, cost 1.0, log p = -0.1 -> total cost 2.0 <= 3.0, score -0.3)
      When top_k=1 is used, token 2 is omitted entirely.
      The top-k search fails to find a feasible solution or returns suboptimal fallback,
      and its certificate MUST have is_global=False and certificate_scope=RESTRICTED_SUBSET.
    """
    class TopKLattice:
        def step_fn_full(self, prompt, prefix):
            if not prefix:
                return [(1, -0.1, "tok1"), (2, -0.2, "tok2")]
            if prefix == [1]:
                return [(0, 0.0, "eos")]
            if prefix == [2]:
                return [(0, -0.1, "eos")]
            return [(0, 0.0, "eos")]

        def step_fn_top1(self, prompt, prefix):
            # Only returns top 1 successor by immediate log prob
            full = self.step_fn_full(prompt, prefix)
            full.sort(key=lambda x: x[1], reverse=True)
            return full[:1]

    class CustomCostModel(TokenCountCostModel):
        def step_cost(self, prompt, prefix, token):
            if not prefix and token == 1:
                return 5.0  # token 1 is infeasible under budget 3.0
            return 1.0

    lattice = TopKLattice()
    cost_model = CustomCostModel(eos_token_id=0)

    # Top-1 restricted search
    cfg_top1 = DPDConfig(
        beam_width=5,
        max_steps=3,
        budget=3.0,
        eos_token_id=0,
        successor_completeness=SuccessorCompleteness.RESTRICTED_TOP_K,
    )
    res_top1 = dual_pruned_decoding("test", cfg_top1, cost_model, lattice.step_fn_top1)
    assert not res_top1.is_feasible
    assert res_top1.certificate.is_global is False
    assert res_top1.certificate.certified_globally_optimal is False
    assert res_top1.certificate.certificate_scope == CertificateScope.RESTRICTED_SUBSET

    # Full search
    cfg_full = DPDConfig(
        beam_width=5,
        max_steps=3,
        budget=3.0,
        eos_token_id=0,
        successor_completeness=SuccessorCompleteness.EXHAUSTIVE_FINITE_DOMAIN,
    )
    res_full = dual_pruned_decoding("test", cfg_full, cost_model, lattice.step_fn_full)
    assert res_full.is_feasible
    assert res_full.best_sequence == [2, 0]
    assert res_full.best_score == pytest.approx(-0.3)
    assert res_full.certificate.is_global is True
    assert res_full.certificate.certified_globally_optimal is True


def test_counterexample_5_tied_scores_and_multiplier_stability():
    """Counterexample 5: Multiple candidates with identical cost and score (degeneracy).
    Verifies that ties in score and cost do not cause division by zero in slope calculations,
    multiplier oscillation, or non-deterministic crashes.
    """
    from dpd.certificates import compute_dual_certificate

    candidates = [
        PrefixCandidate(step=1, token_ids=[1], accumulated_score=-2.0, accumulated_cost=2.0, is_terminal=True),
        PrefixCandidate(step=1, token_ids=[2], accumulated_score=-2.0, accumulated_cost=2.0, is_terminal=True),
        PrefixCandidate(step=1, token_ids=[3], accumulated_score=-2.0, accumulated_cost=2.0, is_terminal=True),
    ]
    cert = compute_dual_certificate(
        all_terminal_candidates=candidates,
        budget=2.0,
        incumbent=candidates[0],
        tolerance=1e-4,
        is_global=True,
    )
    assert cert.is_feasible is True
    assert cert.dual_upper_bound == pytest.approx(-2.0)
    assert cert.suboptimality_gap == pytest.approx(0.0)
    assert cert.certified_globally_optimal is True


def test_counterexample_6_empty_feasible_set_all_candidates_infeasible():
    """Counterexample 6: All candidate completions strictly exceed the resource budget.
    Asserts graceful infeasibility handling, finite fallback sequence, and certificate marking.
    """
    class InfeasibleBranchLattice:
        def step_fn(self, prompt, prefix):
            if not prefix:
                return [(1, -0.5, "b1"), (2, -0.8, "b2")]
            return [(0, 0.0, "eos")]

    class HighCostModel(TokenCountCostModel):
        def step_cost(self, prompt, prefix, token):
            return 10.0

    lattice = InfeasibleBranchLattice()
    cost_model = HighCostModel(eos_token_id=0)
    cfg = DPDConfig(beam_width=4, max_steps=3, budget=5.0, eos_token_id=0)
    res = dual_pruned_decoding("test", cfg, cost_model, lattice.step_fn)

    assert res.is_feasible is False
    assert res.certificate.is_feasible is False
    assert res.certificate.suboptimality_gap == float("inf")
    assert res.certificate.certified_optimal is False
    assert res.certificate.certified_globally_optimal is False


def test_counterexample_7_numerical_tolerance_boundary_shift():
    """Counterexample 7: Floating-point boundary behavior in incumbent bound filtering.
    A candidate with score exactly S_inc - 1e-12 should be pruned if strictly <= incumbent_score,
    while S_inc + 1e-12 must not be pruned.
    """
    from dpd.bounds import TrivialAdmissibleBound
    from dpd.dominance import filter_by_incumbent_bound

    bounds = TrivialAdmissibleBound()
    incumbent_score = -5.0

    c_strictly_below = PrefixCandidate(
        step=1, token_ids=[1], accumulated_score=-5.1, accumulated_cost=1.0
    )
    c_strictly_above = PrefixCandidate(
        step=1, token_ids=[2], accumulated_score=-4.9, accumulated_cost=1.0
    )
    c_exact_tie = PrefixCandidate(
        step=1, token_ids=[3], accumulated_score=-5.0, accumulated_cost=1.0
    )
    c_boundary_survivor = PrefixCandidate(
        step=1, token_ids=[4], accumulated_score=-5.0 + 1e-8, accumulated_cost=1.0
    )

    # Under TrivialAdmissibleBound (bound = 0.0), S(c) + U(c) = S(c).
    # filter_by_incumbent_bound keeps candidate if S(c) + U(c) > incumbent_score + eps (eps=1e-9).
    survivors, pruned_count = filter_by_incumbent_bound(
        [c_strictly_below, c_strictly_above, c_exact_tie, c_boundary_survivor],
        incumbent_score=incumbent_score,
        bound_manager=bounds,
        max_steps=5,
        budget=10.0,
    )
    assert c_strictly_above in survivors
    assert c_boundary_survivor in survivors
    assert c_strictly_below not in survivors
    assert c_exact_tie not in survivors  # Cannot strictly improve upon existing feasible incumbent
    assert pruned_count == 2


def test_counterexample_8_scope_mismatch_prevents_global_certificate():
    """Counterexample 8: Scope mismatch prevention.
    A restricted search cannot produce a GLOBAL_EXHAUSTIVE certificate,
    even if the user or caller passes is_global=True with scope=RESTRICTED_SUBSET.
    """
    from dpd.certificates import compute_dual_certificate

    cand = PrefixCandidate(step=1, token_ids=[1, 0], accumulated_score=-1.0, accumulated_cost=1.0, is_terminal=True)
    
    # Caller attempts to force is_global=True while declaring scope=RESTRICTED_SUBSET
    cert = compute_dual_certificate(
        all_terminal_candidates=[cand],
        budget=2.0,
        incumbent=cand,
        scope=CertificateScope.RESTRICTED_SUBSET,
        is_global=True,  # Conflicting / misleading argument
    )
    # The certificate engine must sanitize this: is_global forced to False
    assert cert.is_global is False
    assert cert.certified_globally_optimal is False
    assert cert.certificate_scope == CertificateScope.RESTRICTED_SUBSET
    assert cert.certified_optimal is False
    assert cert.certified_optimal_within_scope is True
