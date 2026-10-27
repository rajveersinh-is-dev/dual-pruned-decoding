"""Upper-Left 2D Convex Hull algorithms for Lagrangian candidate pruning."""

from __future__ import annotations

from dpd.config import HullConfig
from dpd.kernel import compute_upper_left_hull_rust
from dpd.types import CostScorePoint


def cross_product_orientation(
    p1: CostScorePoint,
    p2: CostScorePoint,
    p3: CostScorePoint,
) -> float:
    """Computes the cross product (P2 - P1) x (P3 - P2) in the (Cost, Score) plane.
    
    A positive cross product indicates a left turn (convex boundary for upper hull),
    while a non-positive value indicates collinearity or a concave bend.
    """
    dx1 = p2.cost - p1.cost
    dy1 = p2.score - p1.score
    dx2 = p3.cost - p2.cost
    dy2 = p3.score - p2.score
    # In standard 2D geometry, cross product of (dx1, dy1) and (dx2, dy2) is dx1*dy2 - dy1*dx2
    return dx1 * dy2 - dy1 * dx2


def compute_upper_left_hull(
    points: list[CostScorePoint],
    config: HullConfig | None = None,
) -> list[CostScorePoint]:
    """Computes the extreme vertices of the upper-left convex hull in the (Cost, Score) plane.
    
    In budget-constrained generation, we seek small Cost C and large Score S.
    The upper-left convex hull consists of the extreme points that maximize the
    Lagrangian scalarization (S - lambda * C) for some multiplier lambda >= 0.
    
    Properties of the returned hull:
    1. Points are sorted strictly by ascending Cost (C_1 < C_2 < ... < C_H).
    2. Scores are strictly ascending (S_1 < S_2 < ... < S_H).
    3. Slopes between consecutive vertices are strictly positive and strictly decreasing:
       (S_2 - S_1)/(C_2 - C_1) > (S_3 - S_2)/(C_3 - C_2) > ... >= 0.
    4. Handles all geometric degeneracies: identical costs, identical scores,
       identical coordinates, vertical lines, and collinear points.
       
    Args:
        points: List of CostScorePoint candidates.
        config: Optional HullConfig settings (tolerance, collinear retention).
        
    Returns:
        List of CostScorePoint vertices forming the upper-left convex hull.
    """
    if not points:
        return []
    if len(points) == 1:
        return [points[0]]

    cfg = config or HullConfig()
    eps = cfg.eps

    if cfg.use_rust_kernel and not cfg.retain_collinear:
        rust_hull = compute_upper_left_hull_rust(points)
        if rust_hull is not None:
            if cfg.max_hull_vertices is not None and len(rust_hull) > cfg.max_hull_vertices:
                return rust_hull[:cfg.max_hull_vertices]
            return rust_hull

    # Step 1: Filter vertical and horizontal dominances
    # Group points by unique cost, keeping the one with maximum score.
    # If two points have identical cost and identical score, retain canonical lowest candidate_id.
    cost_map: dict[float, CostScorePoint] = {}
    for pt in points:
        # Round cost slightly to avoid floating-point representation jitter
        key = round(pt.cost, 12)
        if key not in cost_map:
            cost_map[key] = pt
        else:
            existing = cost_map[key]
            if pt.score > existing.score + eps or abs(pt.score - existing.score) <= eps and pt.candidate_id < existing.candidate_id:
                cost_map[key] = pt

    sorted_candidates = sorted(cost_map.values(), key=lambda p: (p.cost, -p.score))
    if not sorted_candidates:
        return []

    # Step 2: Pareto filter (Score must strictly increase as Cost increases)
    # If a candidate has higher cost but lower or equal score than an earlier candidate,
    # it is strictly dominated for ALL non-negative lambda >= 0.
    pareto_candidates: list[CostScorePoint] = []
    max_score_so_far = -float("inf")
    for pt in sorted_candidates:
        if pt.score > max_score_so_far + eps:
            pareto_candidates.append(pt)
            max_score_so_far = pt.score

    if len(pareto_candidates) <= 2:
        return pareto_candidates

    # Step 3: Andrew's Monotone Chain for Upper-Left Convex Hull
    # For points P1, P2, P3 with C1 < C2 < C3 and S1 < S2 < S3:
    # Slope(P1, P2) = (S2 - S1)/(C2 - C1)
    # Slope(P2, P3) = (S3 - S2)/(C3 - C2)
    # For upper-left convexity, slopes must be strictly decreasing:
    # Slope(P1, P2) > Slope(P2, P3)
    # <=> (S2 - S1)*(C3 - C2) > (S3 - S2)*(C2 - C1)
    # <=> (S2 - S1)*(C3 - C2) - (S3 - S2)*(C2 - C1) > 0
    hull: list[CostScorePoint] = []
    for pt in pareto_candidates:
        while len(hull) >= 2:
            p1 = hull[-2]
            p2 = hull[-1]
            slope_diff = (p2.score - p1.score) * (pt.cost - p2.cost) - (pt.score - p2.score) * (p2.cost - p1.cost)
            
            if cfg.retain_collinear:
                if slope_diff < -eps:
                    hull.pop()
                else:
                    break
            else:
                if slope_diff <= eps:
                    hull.pop()
                else:
                    break
        hull.append(pt)

    if cfg.max_hull_vertices is not None and len(hull) > cfg.max_hull_vertices:
        hull = hull[:cfg.max_hull_vertices]

    return hull


def find_optimal_vertex_for_multiplier(
    hull_vertices: list[CostScorePoint],
    multiplier: float,
) -> CostScorePoint | None:
    """Finds the vertex on the upper-left convex hull that maximizes S - lambda * C.
    
    Since hull_vertices are sorted by increasing cost with decreasing slopes,
    this can be found in O(H) or O(log H). For typical beam sizes (H <= 50),
    a vectorized linear scan is exact and fast.
    """
    if not hull_vertices:
        return None
    if len(hull_vertices) == 1:
        return hull_vertices[0]

    best_pt = hull_vertices[0]
    best_val = best_pt.scalarized_value(multiplier)
    for pt in hull_vertices[1:]:
        val = pt.scalarized_value(multiplier)
        if val > best_val:
            best_val = val
            best_pt = pt
    return best_pt


def get_hull_slopes(hull_vertices: list[CostScorePoint]) -> list[float]:
    """Computes the transition slopes between adjacent hull vertices.
    
    For vertices P_1, P_2, ..., P_H:
    Returns [m_1, m_2, ..., m_{H-1}] where m_i = (S_{i+1} - S_i) / (C_{i+1} - C_i).
    These slopes represent the critical threshold multipliers where the optimal
    hull vertex shifts from P_i to P_{i+1}.
    """
    if len(hull_vertices) < 2:
        return []
    slopes: list[float] = []
    for i in range(len(hull_vertices) - 1):
        p1 = hull_vertices[i]
        p2 = hull_vertices[i + 1]
        dc = p2.cost - p1.cost
        ds = p2.score - p1.score
        slopes.append(ds / dc if dc > 0 else 0.0)
    return slopes
