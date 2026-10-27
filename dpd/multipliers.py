"""Adaptive Lagrangian multiplier controllers for budget-constrained decoding."""

from __future__ import annotations

import math

from dpd.config import MultiplierConfig
from dpd.types import MultiplierType


class BaseMultiplierController:
    """Base class for Lagrangian multiplier price controllers."""

    def __init__(self, config: MultiplierConfig | None = None):
        self.config = config or MultiplierConfig()
        self.config.validate()
        self.current_lambda = self.config.initial_lambda
        self.step_history: list[float] = [self.current_lambda]

    def update(
        self,
        current_step: int,
        max_steps: int,
        total_budget: float,
        current_average_cost: float,
    ) -> float:
        """Computes the updated multiplier price for the next decoding step."""
        raise NotImplementedError

    def reset(self) -> None:
        """Resets controller state to initial configuration."""
        self.current_lambda = self.config.initial_lambda
        self.step_history = [self.current_lambda]


class FixedMultiplierController(BaseMultiplierController):
    """Fixed price multiplier (no adaptation)."""

    def update(
        self,
        current_step: int,
        max_steps: int,
        total_budget: float,
        current_average_cost: float,
    ) -> float:
        return self.current_lambda


class ProjectedSubgradientController(BaseMultiplierController):
    """Projected subgradient multiplier update against linear budget trajectory."""

    def update(
        self,
        current_step: int,
        max_steps: int,
        total_budget: float,
        current_average_cost: float,
    ) -> float:
        if max_steps <= 0:
            return self.current_lambda

        # Nominal budget trajectory at step t: B_t = B * (t / max_steps)
        nominal_target = total_budget * (current_step / max_steps)
        # Resource violation signal: g_t = average_cost - nominal_target
        violation = current_average_cost - nominal_target

        # Diminishing or constant learning rate: eta_t = lr / sqrt(1 + t)
        lr = self.config.learning_rate / math.sqrt(1.0 + current_step)

        new_lambda = self.current_lambda + lr * violation
        # Project into [min_lambda, max_lambda]
        new_lambda = max(self.config.min_lambda, min(self.config.max_lambda, new_lambda))

        self.current_lambda = new_lambda
        self.step_history.append(new_lambda)
        return self.current_lambda


class ProportionalIntegralController(BaseMultiplierController):
    """Proportional-Integral (PI) multiplier controller for smooth zero-offset tracking."""

    def __init__(self, config: MultiplierConfig | None = None):
        super().__init__(config)
        self.integral_error: float = 0.0

    def update(
        self,
        current_step: int,
        max_steps: int,
        total_budget: float,
        current_average_cost: float,
    ) -> float:
        if max_steps <= 0:
            return self.current_lambda

        nominal_target = total_budget * (current_step / max_steps)
        error = current_average_cost - nominal_target
        self.integral_error += error

        p_term = self.config.kp * error
        i_term = self.config.ki * self.integral_error

        new_lambda = self.config.initial_lambda + p_term + i_term
        new_lambda = max(self.config.min_lambda, min(self.config.max_lambda, new_lambda))

        self.current_lambda = new_lambda
        self.step_history.append(new_lambda)
        return self.current_lambda

    def reset(self) -> None:
        super().reset()
        self.integral_error = 0.0


class ExponentialMovingAverageController(BaseMultiplierController):
    """Exponential Moving Average (EMA) multiplier controller for smooth price tracking."""

    def update(
        self,
        current_step: int,
        max_steps: int,
        total_budget: float,
        current_average_cost: float,
    ) -> float:
        """Updates lambda using exponential moving average of violation signals."""
        if max_steps <= 0:
            return self.current_lambda

        nominal_target = total_budget * (current_step / max_steps)
        violation = current_average_cost - nominal_target

        # Update target signal and apply momentum smoothing
        target_signal = self.config.initial_lambda + self.config.learning_rate * violation
        ema = self.config.momentum * self.current_lambda + (1.0 - self.config.momentum) * target_signal

        new_lambda = max(self.config.min_lambda, min(self.config.max_lambda, ema))
        self.current_lambda = new_lambda
        self.step_history.append(new_lambda)
        return self.current_lambda


def create_multiplier_controller(config: MultiplierConfig) -> BaseMultiplierController:
    """Factory function for instantiating multiplier controllers."""
    if config.controller_type == MultiplierType.FIXED:
        return FixedMultiplierController(config)
    elif config.controller_type == MultiplierType.PROJECTED_SUBGRADIENT:
        return ProjectedSubgradientController(config)
    elif config.controller_type == MultiplierType.PROPORTIONAL_INTEGRAL:
        return ProportionalIntegralController(config)
    elif config.controller_type == MultiplierType.EXPONENTIAL_MOVING_AVERAGE:
        return ExponentialMovingAverageController(config)
    else:
        return ProjectedSubgradientController(config)
