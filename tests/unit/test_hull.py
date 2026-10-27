"""Unit tests for 2D upper-left convex hull and Lagrangian extreme point identification."""

import numpy as np

from dpd.config import HullConfig
from dpd.hull import (
    compute_upper_left_hull,
    find_optimal_vertex_for_multiplier,
    get_hull_slopes,
)
from dpd.types import CostScorePoint


def test_empty_and_single_point():
    """Empty list returns empty; single point returns single point."""
    assert compute_upper_left_hull([]) == []

    pt = CostScorePoint(cost=1.0, score=-2.0, candidate_id=0)
    hull = compute_upper_left_hull([pt])
    assert len(hull) == 1
    assert hull[0] == pt


def test_vertical_points_keep_maximum_score():
    """For candidates with identical cost, the upper-left hull must keep only the max score."""
    pts = [
        CostScorePoint(cost=2.0, score=-5.0, candidate_id=1),
        CostScorePoint(cost=2.0, score=-1.0, candidate_id=2),  # Highest score
        CostScorePoint(cost=2.0, score=-3.0, candidate_id=3),
    ]
    hull = compute_upper_left_hull(pts)
    assert len(hull) == 1
    assert hull[0].candidate_id == 2
    assert hull[0].score == -1.0


def test_horizontal_points_keep_minimum_cost():
    """For candidates with identical score, cheaper cost dominates."""
    pts = [
        CostScorePoint(cost=4.0, score=-2.0, candidate_id=1),
        CostScorePoint(cost=1.0, score=-2.0, candidate_id=2),  # Lowest cost
        CostScorePoint(cost=3.0, score=-2.0, candidate_id=3),
    ]
    hull = compute_upper_left_hull(pts)
    assert len(hull) == 1
    assert hull[0].candidate_id == 2
    assert hull[0].cost == 1.0


def test_simple_convex_hull():
    """Points forming a classic upper-left convex boundary."""
    # (0, -4.0), (1, -2.0), (3, -1.0), (6, -0.5)
    # Slopes:
    # (1 - 0): (-2.0 - (-4.0)) / (1 - 0) = 2.0
    # (3 - 1): (-1.0 - (-2.0)) / (3 - 1) = 0.5
    # (6 - 3): (-0.5 - (-1.0)) / (6 - 3) = 0.1667
    # Slopes are strictly decreasing: 2.0 > 0.5 > 0.1667 (Convex!)
    pts = [
        CostScorePoint(cost=0.0, score=-4.0, candidate_id=0),
        CostScorePoint(cost=1.0, score=-2.0, candidate_id=1),
        CostScorePoint(cost=3.0, score=-1.0, candidate_id=2),
        CostScorePoint(cost=6.0, score=-0.5, candidate_id=3),
    ]
    hull = compute_upper_left_hull(pts)
    assert len(hull) == 4
    assert [p.candidate_id for p in hull] == [0, 1, 2, 3]

    slopes = get_hull_slopes(hull)
    assert len(slopes) == 3
    assert slopes[0] > slopes[1] > slopes[2]


def test_interior_point_is_pruned():
    """A point strictly below the chord between two extreme points must be pruned."""
    # Chord between (0, -2.0) and (2, -1.0) has mid-point at cost 1.0 with score -1.5.
    # Point at (1.0, -1.8) is strictly below chord and must be pruned.
    pts = [
        CostScorePoint(cost=0.0, score=-2.0, candidate_id=0),
        CostScorePoint(cost=1.0, score=-1.8, candidate_id=1),  # Interior / concave
        CostScorePoint(cost=2.0, score=-1.0, candidate_id=2),
    ]
    hull = compute_upper_left_hull(pts)
    assert len(hull) == 2
    assert [p.candidate_id for p in hull] == [0, 2]


def test_hull_invariance_theorem_2():
    """Theorem 2 check: for any lambda >= 0, the maximum of (S - lambda * C)
    over all candidates is achieved by a vertex of the upper-left convex hull."""
    np.random.seed(42)
    # Generate 50 random (Cost, Score) points with Cost in [0, 20] and Score in [-30, 0]
    points = [
        CostScorePoint(
            cost=float(np.random.uniform(0.0, 20.0)),
            score=float(np.random.uniform(-30.0, 0.0)),
            candidate_id=i,
        )
        for i in range(50)
    ]
    hull = compute_upper_left_hull(points)
    assert len(hull) >= 1

    # Test across a dense grid of 100 multipliers from 0.0 to 10.0
    lambdas = np.linspace(0.0, 10.0, 100)
    for lam in lambdas:
        # Brute force maximum over all 50 points
        brute_force_max = max(p.scalarized_value(lam) for p in points)
        # Optimal vertex on the hull
        opt_vertex = find_optimal_vertex_for_multiplier(hull, lam)
        assert opt_vertex is not None
        hull_max = opt_vertex.scalarized_value(lam)

        # Must match exactly up to floating-point epsilon
        assert abs(brute_force_max - hull_max) < 1e-8, (
            f"Theorem 2 violation at lambda={lam}: brute={brute_force_max}, hull={hull_max}"
        )


def test_collinear_points():
    """Collinear points on the boundary."""
    # (0, 0), (1, 1), (2, 2)
    pts = [
        CostScorePoint(cost=0.0, score=0.0, candidate_id=0),
        CostScorePoint(cost=1.0, score=1.0, candidate_id=1),
        CostScorePoint(cost=2.0, score=2.0, candidate_id=2),
    ]
    # Without retain_collinear: only extreme endpoints [0, 2]
    hull_strict = compute_upper_left_hull(pts, HullConfig(retain_collinear=False))
    assert len(hull_strict) == 2
    assert [p.candidate_id for p in hull_strict] == [0, 2]

    # With retain_collinear: all 3 points
    hull_collinear = compute_upper_left_hull(pts, HullConfig(retain_collinear=True))
    assert len(hull_collinear) == 3
    assert [p.candidate_id for p in hull_collinear] == [0, 1, 2]
