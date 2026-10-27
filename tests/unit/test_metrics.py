"""Unit tests for metrics and summary aggregation."""

from dpd.metrics import compute_evaluation_summary
from dpd.types import DecodingResult, DecodingTelemetry, DualCertificate


def test_empty_results_summary():
    """Empty list returns valid zeroed summary."""
    summary = compute_evaluation_summary([])
    assert summary.num_instances == 0
    assert summary.feasible_rate == 0.0


def test_metrics_aggregation():
    """Aggregates metrics correctly across results."""
    t1 = DecodingTelemetry(total_candidates_generated=10, total_candidates_pruned=4, total_wall_clock_time_sec=0.01)
    cert1 = DualCertificate(is_feasible=True, incumbent_score=-2.0, incumbent_cost=3.0, dual_upper_bound=-1.5, suboptimality_gap=0.5, certified_optimal=False)
    r1 = DecodingResult(best_sequence=[1, 2, 0], best_score=-2.0, best_cost=3.0, is_feasible=True, budget=5.0, certificate=cert1, telemetry=t1)

    t2 = DecodingTelemetry(total_candidates_generated=15, total_candidates_pruned=8, total_wall_clock_time_sec=0.02)
    cert2 = DualCertificate(is_feasible=False, incumbent_score=-10.0, incumbent_cost=6.0, dual_upper_bound=-5.0, suboptimality_gap=5.0, certified_optimal=False)
    r2 = DecodingResult(best_sequence=[1, 3, 4, 0], best_score=-10.0, best_cost=6.0, is_feasible=False, budget=5.0, certificate=cert2, telemetry=t2)

    summary = compute_evaluation_summary([r1, r2])
    assert summary.num_instances == 2
    assert summary.feasible_rate == 0.5
    assert summary.violation_rate == 0.5
    assert summary.average_score == -6.0
    assert summary.average_cost == 4.5
    assert summary.average_tokens == 3.5
    assert summary.average_expansions == 12.5
    assert summary.average_pruned_candidates == 6.0
    assert summary.average_latency_sec == 0.015
    assert summary.average_dual_gap == 0.5  # Only finite feasible certificates counted
