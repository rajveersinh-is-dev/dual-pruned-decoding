"""Unit tests for baseline decoding algorithms and profiler."""

from dpd.baselines import greedy_decoding, length_penalty_beam_search
from dpd.config import DPDConfig
from dpd.cost_models import TokenCountCostModel
from dpd.instrumentation import HighResolutionProfiler


def test_greedy_decoding():
    """Greedy decoding generates top-1 sequence."""
    def step_fn(prompt, prefix):
        if len(prefix) >= 3:
            return [(0, 0.0, "eos")]
        return [(1, -0.1, "s1"), (2, -1.0, "s2")]

    cfg = DPDConfig(budget=5.0, max_steps=5, eos_token_id=0)
    cm = TokenCountCostModel(eos_token_id=0, zero_cost_tokens={0})
    res = greedy_decoding("prompt", cfg, cm, step_fn)
    assert res.is_feasible
    assert res.best_sequence == [1, 1, 1, 0]


def test_length_penalty_beam_search():
    """Length-penalty beam search runs and returns feasible candidate."""
    def step_fn(prompt, prefix):
        if len(prefix) >= 2:
            return [(0, 0.0, "eos")]
        return [(1, -0.5, "s1"), (2, -0.8, "s2")]

    cfg = DPDConfig(beam_width=2, budget=5.0, max_steps=4, eos_token_id=0)
    cm = TokenCountCostModel(eos_token_id=0, zero_cost_tokens={0})
    res = length_penalty_beam_search("prompt", cfg, cm, step_fn, alpha=0.6)
    assert res.is_feasible
    assert len(res.best_sequence) >= 1


def test_high_resolution_profiler():
    """HighResolutionProfiler executes warmup and timing correctly."""
    counter = 0

    def work():
        nonlocal counter
        counter += 1

    mean_t, _std_t, runs = HighResolutionProfiler.benchmark_callable(
        work,
        warmup_runs=5,
        benchmark_runs=10,
    )
    assert counter == 15  # 5 warmup + 10 runs
    assert mean_t >= 0.0
    assert len(runs) == 10
