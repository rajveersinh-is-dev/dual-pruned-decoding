"""Property and differential testing for the native Rust pruning kernel.

Verifies:
1. Native Rust kernel compilation and FFI loading.
2. 100% agreement between Rust kernel and Python reference on degenerate, collinear,
   adversarial, and randomized inputs.
"""

from __future__ import annotations

import random

import pytest

from dpd.hull import compute_upper_left_hull
from dpd.kernel import compute_upper_left_hull_rust, is_rust_kernel_available
from dpd.types import CostScorePoint


def test_rust_kernel_is_available():
    assert is_rust_kernel_available(), "Native Rust kernel (dpd_kernel) should be compiled and loadable."


def test_rust_kernel_matches_python_on_counterexample_1():
    # A(2.0, -1.0), B(0.0, -2.0), C(1.0, -1.8)
    pts = [
        CostScorePoint(2.0, -1.0, candidate_id=0),
        CostScorePoint(0.0, -2.0, candidate_id=1),
        CostScorePoint(1.0, -1.8, candidate_id=2),
    ]

    py_hull = compute_upper_left_hull(pts)
    rust_hull = compute_upper_left_hull_rust(pts)

    assert rust_hull is not None
    assert [p.candidate_id for p in py_hull] == [p.candidate_id for p in rust_hull]
    assert [p.candidate_id for p in rust_hull] == [1, 0]


def test_rust_kernel_matches_python_on_collinear_and_duplicates():
    pts = [
        CostScorePoint(1.0, -5.0, candidate_id=0),
        CostScorePoint(1.0, -2.0, candidate_id=1),  # Higher score at same cost
        CostScorePoint(1.0, -4.0, candidate_id=2),
        CostScorePoint(2.0, -1.0, candidate_id=3),
        CostScorePoint(3.0, 0.0, candidate_id=4),   # Collinear with (1,-2) and (2,-1)
    ]

    py_hull = compute_upper_left_hull(pts)
    rust_hull = compute_upper_left_hull_rust(pts)

    assert rust_hull is not None
    py_indices = [p.candidate_id for p in py_hull]
    rust_indices = [p.candidate_id for p in rust_hull]
    assert py_indices == rust_indices


@pytest.mark.parametrize("seed", list(range(42, 62)))
def test_rust_kernel_randomized_fuzzing_equivalence(seed: int):
    rng = random.Random(seed)
    n_points = rng.randint(5, 50)
    pts = []
    for i in range(n_points):
        # Generate some duplicate costs and collinear patterns to stress boundary logic
        c = round(rng.uniform(0.0, 10.0), 1)
        s = round(rng.uniform(-20.0, 0.0), 2)
        pts.append(CostScorePoint(cost=c, score=s, candidate_id=i))

    py_hull = compute_upper_left_hull(pts)
    rust_hull = compute_upper_left_hull_rust(pts)

    assert rust_hull is not None
    # Compare vertices
    py_coords = [(p.cost, p.score) for p in py_hull]
    rust_coords = [(p.cost, p.score) for p in rust_hull]

    assert len(py_coords) == len(rust_coords), f"Length mismatch on seed {seed}: py={py_coords}, rust={rust_coords}"
    for (c1, s1), (c2, s2) in zip(py_coords, rust_coords):
        assert abs(c1 - c2) < 1e-6, f"Cost mismatch on seed {seed}"
        assert abs(s1 - s2) < 1e-6, f"Score mismatch on seed {seed}"
