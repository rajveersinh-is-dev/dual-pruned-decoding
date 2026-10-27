"""Theoretical tests for Lagrangian dual upper bounds and duality gaps.

Verifies Theorem 1 on known analytical instances, including Counterexample 2
where the duality gap is strictly positive (non-zero due to discrete non-convexity).
"""

from __future__ import annotations

import pytest

from dpd.certificates import compute_dual_certificate
from dpd.types import PrefixCandidate


@pytest.mark.parametrize(
    ("instance_name", "candidates", "budget", "incumbent_idx", "expected_d_star", "expected_gap"),
    [
        (
            "counterexample_2_positive_gap",
            [
                PrefixCandidate(step=1, token_ids=[1], accumulated_score=-5.0, accumulated_cost=0.0, is_terminal=True),
                PrefixCandidate(step=1, token_ids=[2], accumulated_score=-2.0, accumulated_cost=2.0, is_terminal=True),
                PrefixCandidate(step=1, token_ids=[3], accumulated_score=0.0, accumulated_cost=5.0, is_terminal=True),
            ],
            3.0,
            1,  # y2 is incumbent: score = -2.0, cost = 2.0
            -4.0 / 3.0,
            2.0 / 3.0,
        ),
        (
            "collinear_zero_duality_gap",
            [
                PrefixCandidate(step=1, token_ids=[1], accumulated_score=-4.0, accumulated_cost=1.0, is_terminal=True),
                PrefixCandidate(step=1, token_ids=[2], accumulated_score=-1.0, accumulated_cost=3.0, is_terminal=True),
                PrefixCandidate(step=1, token_ids=[3], accumulated_score=2.0, accumulated_cost=5.0, is_terminal=True),
            ],
            3.0,
            1,  # y2 is incumbent: score = -1.0, cost = 3.0
            -1.0,
            0.0,
        ),
        (
            "two_point_tradeoff_positive_gap",
            [
                PrefixCandidate(step=1, token_ids=[1], accumulated_score=-2.0, accumulated_cost=1.0, is_terminal=True),
                PrefixCandidate(step=1, token_ids=[2], accumulated_score=0.0, accumulated_cost=4.0, is_terminal=True),
            ],
            2.0,
            0,  # y1 is incumbent: score = -2.0, cost = 1.0
            -4.0 / 3.0,
            2.0 / 3.0,
        ),
    ],
)
def test_analytical_duality_gaps(
    instance_name: str,
    candidates: list[PrefixCandidate],
    budget: float,
    incumbent_idx: int,
    expected_d_star: float,
    expected_gap: float,
) -> None:
    """Asserts that compute_dual_certificate matches exact analytical values within numerical tolerance."""
    incumbent = candidates[incumbent_idx]
    cert = compute_dual_certificate(
        all_terminal_candidates=candidates,
        budget=budget,
        incumbent=incumbent,
        tolerance=1e-4,
    )

    assert cert.is_feasible is True
    assert abs(cert.dual_upper_bound - expected_d_star) < 1e-3, (
        f"Failed on {instance_name}: d* = {cert.dual_upper_bound}, expected {expected_d_star}"
    )
    assert abs(cert.suboptimality_gap - expected_gap) < 1e-3, (
        f"Failed on {instance_name}: gap = {cert.suboptimality_gap}, expected {expected_gap}"
    )
    if expected_gap <= 1e-4:
        assert cert.certified_optimal is True
    else:
        assert cert.certified_optimal is False


def test_empty_candidates_certificate() -> None:
    """Verifies that an empty candidate list produces an infeasible certificate with infinite gap."""
    cert = compute_dual_certificate(
        all_terminal_candidates=[],
        budget=10.0,
        incumbent=None,
    )
    assert cert.is_feasible is False
    assert cert.certified_optimal is False
    assert cert.suboptimality_gap == float("inf")
    assert cert.dual_upper_bound == float("inf")
