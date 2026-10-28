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
        is_global=True,
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
        assert cert.certified_globally_optimal is True
        assert cert.certified_optimal_within_scope is True
    else:
        assert cert.certified_optimal is False
        assert cert.certified_globally_optimal is False
        assert cert.certified_optimal_within_scope is False


def test_restricted_subset_scope_does_not_claim_global_optimality() -> None:
    """Verifies that non-exhaustive candidate evaluations never claim global optimality."""
    from dpd.types import CertificateScope

    # Collinear instance where duality gap is zero within the subset
    candidates = [
        PrefixCandidate(step=1, token_ids=[1], accumulated_score=-4.0, accumulated_cost=1.0, is_terminal=True),
        PrefixCandidate(step=1, token_ids=[2], accumulated_score=-1.0, accumulated_cost=3.0, is_terminal=True),
    ]
    cert_restricted = compute_dual_certificate(
        all_terminal_candidates=candidates,
        budget=3.0,
        incumbent=candidates[1],
        tolerance=1e-4,
        is_global=False,  # Heuristic / truncated search
    )
    assert cert_restricted.is_feasible is True
    assert cert_restricted.suboptimality_gap == pytest.approx(0.0)
    # Crucial correctness guarantee: certified_optimal MUST be False for restricted subset
    assert cert_restricted.certified_optimal is False
    assert cert_restricted.certified_globally_optimal is False
    assert cert_restricted.certified_optimal_within_scope is True
    assert cert_restricted.certificate_scope == CertificateScope.RESTRICTED_SUBSET


def test_empty_candidates_certificate() -> None:
    """Verifies that an empty candidate list produces an infeasible certificate with infinite gap."""
    cert = compute_dual_certificate(
        all_terminal_candidates=[],
        budget=10.0,
        incumbent=None,
    )
    assert cert.is_feasible is False
    assert cert.certified_optimal is False
    assert cert.certified_globally_optimal is False
    assert cert.suboptimality_gap == float("inf")
    assert cert.dual_upper_bound == float("inf")

