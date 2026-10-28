"""Configuration dataclasses for Dual-Pruned Decoding."""

from __future__ import annotations

from dataclasses import dataclass, field

from dpd.types import MultiplierType, PruningMode, SuccessorCompleteness


@dataclass
class MultiplierConfig:
    """Configuration for adaptive Lagrangian multiplier pricing."""
    controller_type: MultiplierType = MultiplierType.PROJECTED_SUBGRADIENT
    initial_lambda: float = 0.5
    learning_rate: float = 0.1
    min_lambda: float = 0.0
    max_lambda: float = 50.0
    kp: float = 0.2  # Proportional gain for PI controller
    ki: float = 0.05  # Integral gain for PI controller
    momentum: float = 0.9  # Momentum for EMA controller

    def validate(self) -> None:
        """Validates multiplier configuration values."""
        if self.initial_lambda < 0.0:
            raise ValueError(f"initial_lambda must be non-negative, got {self.initial_lambda}")
        if self.min_lambda < 0.0:
            raise ValueError(f"min_lambda must be non-negative, got {self.min_lambda}")
        if self.max_lambda < self.min_lambda:
            raise ValueError(f"max_lambda ({self.max_lambda}) cannot be smaller than min_lambda ({self.min_lambda})")
        if self.learning_rate <= 0.0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")


@dataclass
class HullConfig:
    """Configuration for convex hull candidate maintenance."""
    eps: float = 1e-9  # Numerical tolerance for cross-product and slope comparisons
    retain_collinear: bool = False  # If False, keep only extreme hull vertices
    max_hull_vertices: int | None = None  # Optional cap on retained hull vertices
    use_rust_kernel: bool = True  # Use high-performance Rust FFI kernel when available


@dataclass
class DPDConfig:
    """Master configuration for Dual-Pruned Decoding."""
    beam_width: int = 5
    max_steps: int = 64
    budget: float = 20.0
    pruning_mode: PruningMode = PruningMode.HYBRID_HULL_AND_BOUND
    multiplier: MultiplierConfig = field(default_factory=MultiplierConfig)
    hull: HullConfig = field(default_factory=HullConfig)
    eos_token_id: int = 2
    pad_token_id: int = 0
    allow_empty: bool = False
    lookahead_steps: int = 0  # Number of lookahead steps for bound evaluation
    optimality_tolerance: float = 1e-4  # Epsilon for dual certificate exactness
    successor_completeness: SuccessorCompleteness = SuccessorCompleteness.RESTRICTED_TOP_K

    def validate(self) -> None:
        """Validates the configuration parameters."""
        if self.beam_width < 1:
            raise ValueError(f"beam_width must be >= 1, got {self.beam_width}")
        if self.max_steps < 1:
            raise ValueError(f"max_steps must be >= 1, got {self.max_steps}")
        if self.budget < 0.0:
            raise ValueError(f"budget must be non-negative, got {self.budget}")
        self.multiplier.validate()
