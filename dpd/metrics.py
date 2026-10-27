"""Evaluation metrics for constrained sequence generation and search efficiency."""

from __future__ import annotations

import math
from dataclasses import dataclass

from dpd.types import DecodingResult


@dataclass
class EvaluationSummary:
    """Aggregated evaluation metrics across a dataset or test suite."""
    num_instances: int
    feasible_rate: float
    violation_rate: float
    average_score: float
    average_cost: float
    average_tokens: float
    average_latency_sec: float
    median_latency_sec: float
    average_expansions: float
    average_pruned_candidates: float
    average_dual_gap: float
    certified_optimal_rate: float


def compute_evaluation_summary(results: list[DecodingResult]) -> EvaluationSummary:
    """Computes comprehensive summary metrics over a collection of DecodingResults."""
    if not results:
        return EvaluationSummary(
            num_instances=0,
            feasible_rate=0.0,
            violation_rate=0.0,
            average_score=0.0,
            average_cost=0.0,
            average_tokens=0.0,
            average_latency_sec=0.0,
            median_latency_sec=0.0,
            average_expansions=0.0,
            average_pruned_candidates=0.0,
            average_dual_gap=0.0,
            certified_optimal_rate=0.0,
        )

    n = len(results)
    num_feasible = sum(1 for r in results if r.is_feasible)
    feasible_rate = num_feasible / n
    violation_rate = 1.0 - feasible_rate

    avg_score = sum(r.best_score for r in results) / n
    avg_cost = sum(r.best_cost for r in results) / n
    avg_tokens = sum(len(r.best_sequence) for r in results) / n

    latencies = [r.telemetry.total_wall_clock_time_sec for r in results]
    latencies.sort()
    avg_lat = sum(latencies) / n
    med_lat = latencies[n // 2] if n % 2 == 1 else (latencies[n // 2 - 1] + latencies[n // 2]) / 2.0

    avg_expansions = sum(r.telemetry.total_candidates_generated for r in results) / n
    avg_pruned = sum(r.telemetry.total_candidates_pruned for r in results) / n

    # Dual certificates
    dual_gaps: list[float] = []
    certified_count = 0
    for r in results:
        if r.certificate is not None:
            if r.certificate.is_feasible and math.isfinite(r.certificate.suboptimality_gap):
                dual_gaps.append(r.certificate.suboptimality_gap)
            if r.certificate.certified_optimal:
                certified_count += 1

    avg_gap = (sum(dual_gaps) / len(dual_gaps)) if dual_gaps else 0.0
    certified_rate = certified_count / n

    return EvaluationSummary(
        num_instances=n,
        feasible_rate=feasible_rate,
        violation_rate=violation_rate,
        average_score=avg_score,
        average_cost=avg_cost,
        average_tokens=avg_tokens,
        average_latency_sec=avg_lat,
        median_latency_sec=med_lat,
        average_expansions=avg_expansions,
        average_pruned_candidates=avg_pruned,
        average_dual_gap=avg_gap,
        certified_optimal_rate=certified_rate,
    )
