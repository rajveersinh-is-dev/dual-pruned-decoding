"""Core data types and interfaces for Dual-Pruned Decoding (DPD)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class PruningMode(str, Enum):
    """Pruning mode for DPD search."""
    EXACT_BOUND_AND_DOMINANCE = "exact_bound_and_dominance"
    HULL_EXTREMES_ONLY = "hull_extremes_only"
    HYBRID_HULL_AND_BOUND = "hybrid_hull_and_bound"
    NO_PRUNING = "no_pruning"


class MultiplierType(str, Enum):
    """Type of Lagrangian multiplier controller."""
    PROJECTED_SUBGRADIENT = "projected_subgradient"
    PROPORTIONAL_INTEGRAL = "proportional_integral"
    EXPONENTIAL_MOVING_AVERAGE = "exponential_moving_average"
    FIXED = "fixed"


@dataclass(frozen=True)
class CostScorePoint:
    """Represents a 2D point in the (Cost, Score) coordinate plane."""
    cost: float
    score: float
    candidate_id: int

    def scalarized_value(self, multiplier: float) -> float:
        """Computes the Lagrangian scalarized value S - lambda * C."""
        return self.score - multiplier * self.cost


@dataclass
class PrefixCandidate:
    """Represents an active candidate sequence during decoding."""
    token_ids: list[int]
    accumulated_score: float  # S(u) <= 0
    accumulated_cost: float   # C(u) >= 0
    step: int
    is_terminal: bool = False
    state_id: str | None = None  # Optional state equivalence identifier
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def remaining_budget(self) -> float:
        """Remaining budget assuming an external total budget B in metadata."""
        budget = float(self.metadata.get("budget", 0.0))
        return budget - self.accumulated_cost

    def extend(
        self,
        token_id: int,
        step_score: float,
        step_cost: float,
        is_terminal: bool = False,
        state_id: str | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> PrefixCandidate:
        """Extends this prefix with a new token."""
        new_meta = dict(self.metadata)
        if extra_metadata:
            new_meta.update(extra_metadata)
        return PrefixCandidate(
            token_ids=self.token_ids + [token_id],
            accumulated_score=self.accumulated_score + step_score,
            accumulated_cost=self.accumulated_cost + step_cost,
            step=self.step + 1,
            is_terminal=is_terminal,
            state_id=state_id,
            metadata=new_meta,
        )


@dataclass
class DualCertificate:
    """Mathematical certificate of dual upper bounds and suboptimality gap."""
    is_feasible: bool
    incumbent_score: float
    incumbent_cost: float
    dual_upper_bound: float
    suboptimality_gap: float  # Delta = dual_upper_bound - incumbent_score >= 0
    multipliers_evaluated: list[float] = field(default_factory=list)
    dual_values: list[float] = field(default_factory=list)
    best_multiplier: float = 0.0
    certified_optimal: bool = False  # True if Delta <= epsilon (within tolerance)
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class DecodingTelemetry:
    """Detailed telemetry and instrumentation recorded during decoding."""
    step_candidates_expanded: list[int] = field(default_factory=list)
    step_candidates_pruned_hull: list[int] = field(default_factory=list)
    step_candidates_pruned_infeasibility: list[int] = field(default_factory=list)
    step_candidates_pruned_bound: list[int] = field(default_factory=list)
    step_candidates_pruned_dominance: list[int] = field(default_factory=list)
    step_multipliers: list[float] = field(default_factory=list)
    step_average_costs: list[float] = field(default_factory=list)
    step_durations_ns: list[int] = field(default_factory=list)
    total_forward_passes: int = 0
    total_candidates_generated: int = 0
    total_candidates_pruned: int = 0
    total_wall_clock_time_sec: float = 0.0


@dataclass
class DecodingResult:
    """The complete result returned by a decoding search algorithm."""
    best_sequence: list[int]
    best_score: float
    best_cost: float
    is_feasible: bool
    budget: float
    certificate: DualCertificate | None = None
    all_feasible_sequences: list[PrefixCandidate] = field(default_factory=list)
    telemetry: DecodingTelemetry = field(default_factory=DecodingTelemetry)
    termination_reason: str = "eos"
