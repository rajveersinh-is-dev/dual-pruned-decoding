"""Unit tests for DPD and multiplier configuration validation."""

from __future__ import annotations

import pytest

from dpd.config import DPDConfig, HullConfig, MultiplierConfig
from dpd.types import MultiplierType, PruningMode


def test_multiplier_config_valid():
    """Valid configuration passes validation without exceptions."""
    cfg = MultiplierConfig(
        controller_type=MultiplierType.PROJECTED_SUBGRADIENT,
        initial_lambda=1.0,
        learning_rate=0.1,
        min_lambda=0.0,
        max_lambda=10.0,
    )
    cfg.validate()


def test_multiplier_config_validation_failures():
    """MultiplierConfig detects negative lambdas, invalid limits, and non-positive learning rates."""
    with pytest.raises(ValueError, match="initial_lambda must be non-negative"):
        MultiplierConfig(initial_lambda=-0.1).validate()

    with pytest.raises(ValueError, match="min_lambda must be non-negative"):
        MultiplierConfig(min_lambda=-1.0).validate()

    with pytest.raises(ValueError, match="cannot be smaller than min_lambda"):
        MultiplierConfig(min_lambda=5.0, max_lambda=2.0).validate()

    with pytest.raises(ValueError, match="learning_rate must be positive"):
        MultiplierConfig(learning_rate=0.0).validate()

    with pytest.raises(ValueError, match="learning_rate must be positive"):
        MultiplierConfig(learning_rate=-0.5).validate()


def test_dpd_config_valid():
    """Valid master DPDConfig passes validation."""
    cfg = DPDConfig(
        beam_width=5,
        max_steps=32,
        budget=10.0,
        pruning_mode=PruningMode.HYBRID_HULL_AND_BOUND,
    )
    cfg.validate()


def test_dpd_config_validation_failures():
    """DPDConfig detects invalid beam width, max steps, and negative budget."""
    with pytest.raises(ValueError, match="beam_width must be >= 1"):
        DPDConfig(beam_width=0).validate()

    with pytest.raises(ValueError, match="max_steps must be >= 1"):
        DPDConfig(max_steps=0).validate()

    with pytest.raises(ValueError, match="budget must be non-negative"):
        DPDConfig(budget=-1.0).validate()


def test_hull_config_defaults():
    """HullConfig instantiates with expected default parameters."""
    cfg = HullConfig()
    assert cfg.eps == 1e-9
    assert cfg.retain_collinear is False
    assert cfg.max_hull_vertices is None
    assert cfg.use_rust_kernel is True

