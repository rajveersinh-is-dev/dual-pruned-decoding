"""Unit tests for adaptive Lagrangian multiplier controllers."""


from dpd.config import MultiplierConfig
from dpd.multipliers import (
    create_multiplier_controller,
)
from dpd.types import MultiplierType


def test_fixed_multiplier():
    """Fixed controller never modifies lambda."""
    cfg = MultiplierConfig(controller_type=MultiplierType.FIXED, initial_lambda=1.5)
    ctrl = create_multiplier_controller(cfg)
    assert ctrl.update(1, 10, 20.0, 50.0) == 1.5
    assert ctrl.update(2, 10, 20.0, 5.0) == 1.5


def test_projected_subgradient_increases_when_over_budget():
    """When current average cost exceeds nominal target, lambda increases."""
    cfg = MultiplierConfig(
        controller_type=MultiplierType.PROJECTED_SUBGRADIENT,
        initial_lambda=1.0,
        learning_rate=0.5,
        min_lambda=0.0,
        max_lambda=10.0,
    )
    ctrl = create_multiplier_controller(cfg)
    # At step 5 of 10, budget is 20.0, so nominal target is 10.0.
    # Current average cost is 15.0 > 10.0 => over budget
    new_lam = ctrl.update(current_step=5, max_steps=10, total_budget=20.0, current_average_cost=15.0)
    assert new_lam > 1.0


def test_projected_subgradient_decreases_when_under_budget():
    """When current average cost is below nominal target, lambda decreases."""
    cfg = MultiplierConfig(
        controller_type=MultiplierType.PROJECTED_SUBGRADIENT,
        initial_lambda=1.0,
        learning_rate=0.5,
        min_lambda=0.0,
        max_lambda=10.0,
    )
    ctrl = create_multiplier_controller(cfg)
    # At step 5 of 10, target is 10.0. Cost is 2.0 < 10.0 => under budget
    new_lam = ctrl.update(current_step=5, max_steps=10, total_budget=20.0, current_average_cost=2.0)
    assert new_lam < 1.0
    assert new_lam >= 0.0


def test_projected_subgradient_projection_clamping():
    """Lambda is clamped strictly between min_lambda and max_lambda."""
    cfg = MultiplierConfig(
        controller_type=MultiplierType.PROJECTED_SUBGRADIENT,
        initial_lambda=0.5,
        learning_rate=100.0,  # Huge step
        min_lambda=0.1,
        max_lambda=5.0,
    )
    ctrl = create_multiplier_controller(cfg)
    # Huge violation upwards
    high_lam = ctrl.update(5, 10, 10.0, 1000.0)
    assert high_lam == 5.0

    # Huge violation downwards
    low_lam = ctrl.update(5, 10, 1000.0, 0.0)
    assert low_lam == 0.1


def test_proportional_integral_tracking():
    """PI controller integrates errors over steps."""
    cfg = MultiplierConfig(
        controller_type=MultiplierType.PROPORTIONAL_INTEGRAL,
        initial_lambda=1.0,
        kp=0.1,
        ki=0.05,
        min_lambda=0.0,
        max_lambda=10.0,
    )
    ctrl = create_multiplier_controller(cfg)
    lam1 = ctrl.update(1, 10, 20.0, 5.0)  # target 2.0, error +3.0
    lam2 = ctrl.update(2, 10, 20.0, 7.0)  # target 4.0, error +3.0
    assert lam2 > lam1


def test_exponential_moving_average_controller_updates():
    """EMA controller smoothly tracks violation signals with momentum."""
    cfg = MultiplierConfig(
        controller_type=MultiplierType.EXPONENTIAL_MOVING_AVERAGE,
        initial_lambda=1.0,
        learning_rate=0.5,
        momentum=0.8,
        min_lambda=0.0,
        max_lambda=10.0,
    )
    ctrl = create_multiplier_controller(cfg)
    assert ctrl.current_lambda == 1.0

    # Step 5 of 10, target 10.0. Cost 14.0 => violation = +4.0
    # target_signal = 1.0 + 0.5 * 4.0 = 3.0
    # expected ema = 0.8 * 1.0 + 0.2 * 3.0 = 0.8 + 0.6 = 1.4
    lam1 = ctrl.update(current_step=5, max_steps=10, total_budget=20.0, current_average_cost=14.0)
    assert abs(lam1 - 1.4) < 1e-6
    assert ctrl.current_lambda == lam1

    # Next step under budget: target 12.0. Cost 6.0 => violation = -6.0
    # target_signal = 1.0 + 0.5 * (-6.0) = -2.0
    # expected ema = 0.8 * 1.4 + 0.2 * (-2.0) = 1.12 - 0.4 = 0.72
    lam2 = ctrl.update(current_step=6, max_steps=10, total_budget=20.0, current_average_cost=6.0)
    assert abs(lam2 - 0.72) < 1e-6
    assert lam2 < lam1


def test_exponential_moving_average_clamping_and_reset():
    """EMA controller respects min/max projection and resets properly."""
    cfg = MultiplierConfig(
        controller_type=MultiplierType.EXPONENTIAL_MOVING_AVERAGE,
        initial_lambda=1.0,
        learning_rate=10.0,
        momentum=0.5,
        min_lambda=0.2,
        max_lambda=4.0,
    )
    ctrl = create_multiplier_controller(cfg)

    # Massive violation upwards
    high_lam = ctrl.update(1, 10, 10.0, 500.0)
    assert high_lam == 4.0

    # Massive violation downwards
    low_lam = ctrl.update(9, 10, 1000.0, 0.0)
    assert low_lam == 0.2

    # Reset
    ctrl.reset()
    assert ctrl.current_lambda == 1.0
    assert ctrl.step_history == [1.0]

