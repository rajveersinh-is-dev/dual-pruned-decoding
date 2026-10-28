"""Dual upper bound computation and mathematical certificates of suboptimality."""

from __future__ import annotations

from dpd.hull import CostScorePoint, compute_upper_left_hull
from dpd.types import CertificateScope, DualCertificate, PrefixCandidate


def compute_dual_certificate(
    all_terminal_candidates: list[PrefixCandidate],
    budget: float,
    incumbent: PrefixCandidate | None,
    num_multipliers: int = 100,
    max_multiplier: float = 20.0,
    tolerance: float = 1e-4,
    scope: CertificateScope = CertificateScope.RESTRICTED_SUBSET,
    is_global: bool = False,
    open_frontier_upper_bound: float | None = None,
) -> DualCertificate:
    """Computes the Lagrangian dual upper bound and suboptimality certificate.
    
    Theorem 1: For any lambda >= 0 and any set of candidates Y,
        d_Y(lambda) = lambda * B + max_{y in Y} (S(y) - lambda * C(y))
    is an upper bound on max_{y in Y, C(y) <= B} S(y).
    
    Critical Scope Distinction:
        - If is_global is False (default for heuristic beam searches):
          The upper bound d* holds ONLY over the explored candidate set Y_term.
          It does NOT bound sequences in unexpanded branches truncated by the beam.
          In this case, certified_optimal_within_scope may be True, but
          certified_globally_optimal and certified_optimal are set to False.
        - If is_global is True (exhaustive tree or branch-and-bound with closed frontier):
          d* is a genuine upper bound on the global optimal score S(y*).
          If suboptimality_gap <= tolerance, certified_globally_optimal and
          certified_optimal are True.
    
    Args:
        all_terminal_candidates: Evaluated terminal sequences.
        budget: The resource budget B.
        incumbent: The best feasible solution found, or None if infeasible.
        num_multipliers: Number of multiplier evaluation points.
        max_multiplier: Upper limit for multiplier sweep.
        tolerance: Numerical tolerance for exact optimality (Delta <= tolerance).
        scope: Explicit scope of the evaluated sequence set.
        is_global: Whether the candidate set covers all completions in the optimization domain.
        open_frontier_upper_bound: Optional upper bound on unexplored open frontier branches.
        
    Returns:
        DualCertificate with bounding gap and full scope details.
    """
    if not all_terminal_candidates:
        return DualCertificate(
            is_feasible=False,
            incumbent_score=-float("inf"),
            incumbent_cost=0.0,
            dual_upper_bound=float("inf"),
            suboptimality_gap=float("inf"),
            certified_optimal=False,
            certificate_scope=scope,
            is_global=is_global,
            certified_globally_optimal=False,
            certified_optimal_within_scope=False,
            details={"error": "No terminal candidates evaluated.", "is_global": is_global},
        )

    is_feasible = (incumbent is not None) and (incumbent.accumulated_cost <= budget + 1e-9)
    incumbent_score = incumbent.accumulated_score if is_feasible and incumbent else -float("inf")
    incumbent_cost = incumbent.accumulated_cost if is_feasible and incumbent else 0.0

    # Analytical slope breakpoints from upper-left hull
    candidate_lambdas: set[float] = {0.0}
    if len(all_terminal_candidates) >= 2:
        hull_pts = [
            CostScorePoint(cost=c.accumulated_cost, score=c.accumulated_score, candidate_id=i)
            for i, c in enumerate(all_terminal_candidates)
        ]
        hull_vertices = compute_upper_left_hull(hull_pts)
        for i in range(len(hull_vertices) - 1):
            v1, v2 = hull_vertices[i], hull_vertices[i + 1]
            dc = v2.cost - v1.cost
            if dc > 1e-9:
                slope = (v2.score - v1.score) / dc
                if 0.0 <= slope <= max_multiplier:
                    candidate_lambdas.add(slope)

    # Multiplier grid search combined with exact hull breakpoints
    grid_multipliers = [i * (max_multiplier / max(1, num_multipliers - 1)) for i in range(num_multipliers)]
    multipliers = sorted(candidate_lambdas.union(grid_multipliers))
    dual_values: list[float] = []

    for lam in multipliers:
        # Compute d(lambda) = lambda * B + max_{y} (S(y) - lambda * C(y))
        max_scalarized = max(
            c.accumulated_score - lam * c.accumulated_cost
            for c in all_terminal_candidates
        )
        if open_frontier_upper_bound is not None:
            max_scalarized = max(max_scalarized, open_frontier_upper_bound)

        d_val = lam * budget + max_scalarized
        dual_values.append(d_val)

    best_dual_idx = int(min(range(len(dual_values)), key=lambda i: dual_values[i]))
    best_dual_ub = dual_values[best_dual_idx]
    best_lam = multipliers[best_dual_idx]

    if is_feasible:
        # Suboptimality gap Delta = d* - S_inc
        suboptimality_gap = max(0.0, best_dual_ub - incumbent_score)
        certified_optimal_within_scope = suboptimality_gap <= tolerance
        certified_globally_optimal = is_global and certified_optimal_within_scope
        certified_optimal = certified_globally_optimal
    else:
        suboptimality_gap = float("inf")
        certified_optimal_within_scope = False
        certified_globally_optimal = False
        certified_optimal = False

    effective_scope = CertificateScope.GLOBAL_EXHAUSTIVE if is_global else scope

    return DualCertificate(
        is_feasible=is_feasible,
        incumbent_score=incumbent_score,
        incumbent_cost=incumbent_cost,
        dual_upper_bound=best_dual_ub,
        suboptimality_gap=suboptimality_gap,
        multipliers_evaluated=multipliers,
        dual_values=dual_values,
        best_multiplier=best_lam,
        certified_optimal=certified_optimal,
        certificate_scope=effective_scope,
        is_global=is_global,
        certified_globally_optimal=certified_globally_optimal,
        certified_optimal_within_scope=certified_optimal_within_scope,
        details={
            "num_candidates_evaluated": len(all_terminal_candidates),
            "best_multiplier": best_lam,
            "min_dual_value": best_dual_ub,
            "certificate_scope": effective_scope.value,
            "is_global": is_global,
            "scope_description": (
                "Globally valid upper bound over the complete discrete sequence domain."
                if is_global else
                "Restricted dual upper bound valid strictly over the evaluated terminal candidate subset."
            ),
            "certified_globally_optimal": certified_globally_optimal,
            "certified_optimal_within_scope": certified_optimal_within_scope,
        },
    )
