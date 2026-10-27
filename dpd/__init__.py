"""Dual-Pruned Decoding (DPD): Certified Resource-Aware Search for Autoregressive Generation."""

from __future__ import annotations

from dpd.bounds import (
    BaseCompletionBound,
    BudgetConditionedBound,
    TrivialAdmissibleBound,
)
from dpd.certificates import compute_dual_certificate
from dpd.config import (
    DPDConfig,
    HullConfig,
    MultiplierConfig,
)
from dpd.constraints import BudgetConstraintValidator
from dpd.cost_models import (
    BaseCostModel,
    CallableCostModel,
    CharacterCostModel,
    MonetaryCostModel,
    TokenCountCostModel,
)
from dpd.decoding import dual_pruned_decoding
from dpd.dominance import (
    filter_by_incumbent_bound,
    filter_state_matched_dominance,
)
from dpd.hull import (
    compute_upper_left_hull,
    find_optimal_vertex_for_multiplier,
    get_hull_slopes,
)
from dpd.instrumentation import HighResolutionProfiler
from dpd.metrics import EvaluationSummary, compute_evaluation_summary
from dpd.multipliers import (
    BaseMultiplierController,
    FixedMultiplierController,
    ProjectedSubgradientController,
    ProportionalIntegralController,
    create_multiplier_controller,
)
from dpd.recovery import RecoveryManager
from dpd.types import (
    CostScorePoint,
    DecodingResult,
    DecodingTelemetry,
    DualCertificate,
    MultiplierType,
    PrefixCandidate,
    PruningMode,
)

__version__ = "0.1.0"

__all__ = [
    "BaseCompletionBound",
    "BaseCostModel",
    "BaseMultiplierController",
    "BudgetConditionedBound",
    "BudgetConstraintValidator",
    "CallableCostModel",
    "CharacterCostModel",
    "CostScorePoint",
    "DPDConfig",
    "DecodingResult",
    "DecodingTelemetry",
    "DualCertificate",
    "EvaluationSummary",
    "FixedMultiplierController",
    "HighResolutionProfiler",
    "HullConfig",
    "MonetaryCostModel",
    "MultiplierConfig",
    "MultiplierType",
    "PrefixCandidate",
    "ProjectedSubgradientController",
    "ProportionalIntegralController",
    "PruningMode",
    "RecoveryManager",
    "TokenCountCostModel",
    "TrivialAdmissibleBound",
    "compute_dual_certificate",
    "compute_evaluation_summary",
    "compute_upper_left_hull",
    "create_multiplier_controller",
    "dual_pruned_decoding",
    "filter_by_incumbent_bound",
    "filter_state_matched_dominance",
    "find_optimal_vertex_for_multiplier",
    "get_hull_slopes",
]
