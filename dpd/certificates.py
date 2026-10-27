"""Dual upper bound computation and mathematical certificates of suboptimality."""

from __future__ import annotations

from dpd.types import DualCertificate, PrefixCandidate


def compute_dual_certificate(
    all_terminal_candidates: list[PrefixCandidate],
    budget: float,
    incumbent: PrefixCandidate | None,
    num_multipliers: int = 100,
    max_multiplier: float = 20.0,
    tolerance: float = 1e-4,
) -> DualCertificate:
    """Computes the Lagrangian dual upper bound and suboptimality certificate.
    
    Theorem 1: For any lambda >= 0, d(lambda) = lambda * B + max_y (S(y) - lambda * C(y))
    is an upper bound on S(y*).
    Taking d* = min_lambda d(lambda) yields the tightest evaluated Lagrangian upper bound.
    The gap Delta = d* - S(y_inc) >= 0 certifies the maximum suboptimality of the incumbent.
    
    Args:
        all_terminal_candidates: All complete sequences explored in the search lattice.
        budget: The resource budget B.
        incumbent: The best feasible solution found, or None if infeasible.
        num_multipliers: Number of multiplier evaluation points.
        max_multiplier: Upper limit for multiplier sweep.
        tolerance: Numerical tolerance for exact optimality (Delta <= tolerance).
        
    Returns:
        DualCertificate with bounding gap and full evaluation details.
    """
    if not all_terminal_candidates:
        return DualCertificate(
            is_feasible=False,
            incumbent_score=-float("inf"),
            incumbent_cost=0.0,
            dual_upper_bound=float("inf"),
            suboptimality_gap=float("inf"),
            certified_optimal=False,
            details={"error": "No terminal candidates evaluated."},
        )

    is_feasible = (incumbent is not None) and (incumbent.accumulated_cost <= budget + 1e-9)
    incumbent_score = incumbent.accumulated_score if is_feasible and incumbent else -float("inf")
    incumbent_cost = incumbent.accumulated_cost if is_feasible and incumbent else 0.0

    # Multiplier grid search over [0, max_multiplier]
    multipliers = [i * (max_multiplier / max(1, num_multipliers - 1)) for i in range(num_multipliers)]
    dual_values: list[float] = []

    for lam in multipliers:
        # Compute d(lambda) = lambda * B + max_{y} (S(y) - lambda * C(y))
        max_scalarized = max(
            c.accumulated_score - lam * c.accumulated_cost
            for c in all_terminal_candidates
        )
        d_val = lam * budget + max_scalarized
        dual_values.append(d_val)

    best_dual_idx = int(min(range(len(dual_values)), key=lambda i: dual_values[i]))
    best_dual_ub = dual_values[best_dual_idx]
    best_lam = multipliers[best_dual_idx]

    if is_feasible:
        # Suboptimality gap Delta = d* - S_inc
        suboptimality_gap = max(0.0, best_dual_ub - incumbent_score)
        certified_optimal = suboptimality_gap <= tolerance
    else:
        suboptimality_gap = float("inf")
        certified_optimal = False

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
        details={
            "num_candidates_evaluated": len(all_terminal_candidates),
            "best_multiplier": best_lam,
            "min_dual_value": best_dual_ub,
        },
    )
