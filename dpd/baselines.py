"""Standard and constrained decoding baselines for rigorous benchmarking."""

from __future__ import annotations

from typing import Any

from dpd.config import DPDConfig
from dpd.constraints import BudgetConstraintValidator
from dpd.cost_models import BaseCostModel
from dpd.decoding import StepFunction
from dpd.instrumentation import HighResolutionProfiler
from dpd.types import (
    DecodingResult,
    DecodingTelemetry,
    PrefixCandidate,
)


def standard_beam_search(
    prompt: Any,
    config: DPDConfig,
    cost_model: BaseCostModel,
    step_fn: StepFunction,
    initial_tokens: list[int] | None = None,
) -> DecodingResult:
    """Standard Beam Search: expands top-K candidates strictly by model log-probability."""
    telemetry = DecodingTelemetry()
    validator = BudgetConstraintValidator(config.budget, cost_model)
    t_start = HighResolutionProfiler.now_ns()

    init_toks = initial_tokens or []
    root = PrefixCandidate(
        token_ids=init_toks,
        accumulated_score=0.0,
        accumulated_cost=0.0,
        step=0,
        is_terminal=(len(init_toks) > 0 and init_toks[-1] == config.eos_token_id),
    )

    beam = [root]
    all_terminals: list[PrefixCandidate] = []
    all_explored: list[PrefixCandidate] = [root]

    for step in range(1, config.max_steps + 1):
        step_t0 = HighResolutionProfiler.now_ns()
        active = [c for c in beam if not c.is_terminal]
        if not active:
            break

        expanded: list[PrefixCandidate] = []
        for parent in active:
            branches = step_fn(prompt, parent.token_ids)
            telemetry.total_forward_passes += 1
            for token_id, log_prob, state_id in branches:
                c_step = cost_model.step_cost(prompt, parent.token_ids, token_id)
                child = parent.extend(
                    token_id=token_id,
                    step_score=log_prob,
                    step_cost=c_step,
                    is_terminal=(token_id == config.eos_token_id),
                    state_id=state_id,
                )
                expanded.append(child)
                all_explored.append(child)

        telemetry.total_candidates_generated += len(expanded)
        current_terms = [c for c in expanded if c.is_terminal]
        all_terminals.extend(current_terms)

        # Standard beam search ranks strictly by accumulated log-prob S(u)
        expanded.sort(key=lambda c: c.accumulated_score, reverse=True)
        beam = expanded[: config.beam_width]

        step_t1 = HighResolutionProfiler.now_ns()
        telemetry.step_durations_ns.append(step_t1 - step_t0)

    t_end = HighResolutionProfiler.now_ns()
    telemetry.total_wall_clock_time_sec = (t_end - t_start) * 1e-9

    feasible_terms = [c for c in all_terminals if validator.is_prefix_feasible(c)]
    if feasible_terms:
        best_cand = max(feasible_terms, key=lambda c: c.accumulated_score)
        is_feasible = True
    elif all_terminals:
        # If no feasible terminal, pick best score terminal even if violated
        best_cand = max(all_terminals, key=lambda c: c.accumulated_score)
        is_feasible = validator.is_prefix_feasible(best_cand)
    else:
        best_cand = beam[0] if beam else root
        is_feasible = validator.is_prefix_feasible(best_cand)

    return DecodingResult(
        best_sequence=best_cand.token_ids,
        best_score=best_cand.accumulated_score,
        best_cost=best_cand.accumulated_cost,
        is_feasible=is_feasible,
        budget=config.budget,
        all_feasible_sequences=feasible_terms,
        telemetry=telemetry,
        termination_reason="standard_beam",
    )


def length_penalty_beam_search(
    prompt: Any,
    config: DPDConfig,
    cost_model: BaseCostModel,
    step_fn: StepFunction,
    alpha: float = 0.6,
    initial_tokens: list[int] | None = None,
) -> DecodingResult:
    """Length-Penalty Beam Search (Wu et al. 2016): S_norm = S / ((5 + L)^alpha / 6^alpha)."""
    telemetry = DecodingTelemetry()
    validator = BudgetConstraintValidator(config.budget, cost_model)
    t_start = HighResolutionProfiler.now_ns()

    def norm_score(c: PrefixCandidate) -> float:
        length = max(1, len(c.token_ids))
        lp = ((5.0 + length) / 6.0) ** alpha
        return c.accumulated_score / lp

    init_toks = initial_tokens or []
    root = PrefixCandidate(
        token_ids=init_toks,
        accumulated_score=0.0,
        accumulated_cost=0.0,
        step=0,
        is_terminal=(len(init_toks) > 0 and init_toks[-1] == config.eos_token_id),
    )

    beam = [root]
    all_terminals: list[PrefixCandidate] = []

    for step in range(1, config.max_steps + 1):
        step_t0 = HighResolutionProfiler.now_ns()
        active = [c for c in beam if not c.is_terminal]
        if not active:
            break

        expanded: list[PrefixCandidate] = []
        for parent in active:
            branches = step_fn(prompt, parent.token_ids)
            telemetry.total_forward_passes += 1
            for token_id, log_prob, state_id in branches:
                c_step = cost_model.step_cost(prompt, parent.token_ids, token_id)
                child = parent.extend(
                    token_id=token_id,
                    step_score=log_prob,
                    step_cost=c_step,
                    is_terminal=(token_id == config.eos_token_id),
                    state_id=state_id,
                )
                expanded.append(child)

        telemetry.total_candidates_generated += len(expanded)
        current_terms = [c for c in expanded if c.is_terminal]
        all_terminals.extend(current_terms)

        expanded.sort(key=norm_score, reverse=True)
        beam = expanded[: config.beam_width]

        step_t1 = HighResolutionProfiler.now_ns()
        telemetry.step_durations_ns.append(step_t1 - step_t0)

    t_end = HighResolutionProfiler.now_ns()
    telemetry.total_wall_clock_time_sec = (t_end - t_start) * 1e-9

    feasible_terms = [c for c in all_terminals if validator.is_prefix_feasible(c)]
    if feasible_terms:
        best_cand = max(feasible_terms, key=norm_score)
        is_feasible = True
    elif all_terminals:
        best_cand = max(all_terminals, key=norm_score)
        is_feasible = validator.is_prefix_feasible(best_cand)
    else:
        best_cand = beam[0] if beam else root
        is_feasible = validator.is_prefix_feasible(best_cand)

    return DecodingResult(
        best_sequence=best_cand.token_ids,
        best_score=best_cand.accumulated_score,
        best_cost=best_cand.accumulated_cost,
        is_feasible=is_feasible,
        budget=config.budget,
        all_feasible_sequences=feasible_terms,
        telemetry=telemetry,
        termination_reason="length_penalty_beam",
    )


def greedy_decoding(
    prompt: Any,
    config: DPDConfig,
    cost_model: BaseCostModel,
    step_fn: StepFunction,
    initial_tokens: list[int] | None = None,
) -> DecodingResult:
    """Greedy Decoding: top-1 next token at each step."""
    cfg = DPDConfig(
        beam_width=1,
        max_steps=config.max_steps,
        budget=config.budget,
        eos_token_id=config.eos_token_id,
    )
    return standard_beam_search(prompt, cfg, cost_model, step_fn, initial_tokens)


def exact_exhaustive_search(
    prompt: Any,
    config: DPDConfig,
    cost_model: BaseCostModel,
    step_fn: StepFunction,
    initial_tokens: list[int] | None = None,
) -> tuple[PrefixCandidate | None, list[PrefixCandidate]]:
    """Exact exhaustive depth-first search for small test lattices.
    
    Used strictly in tests/theory to verify the true global constrained optimum
    y* = argmax_{y in F_B} S(y) from first principles.
    """
    validator = BudgetConstraintValidator(config.budget, cost_model)
    all_terminals: list[PrefixCandidate] = []

    init_toks = initial_tokens or []
    root = PrefixCandidate(
        token_ids=init_toks,
        accumulated_score=0.0,
        accumulated_cost=0.0,
        step=0,
        is_terminal=(len(init_toks) > 0 and init_toks[-1] == config.eos_token_id),
    )

    stack = [root]
    while stack:
        curr = stack.pop()
        if curr.is_terminal or curr.step >= config.max_steps:
            all_terminals.append(curr)
            continue

        branches = step_fn(prompt, curr.token_ids)
        for token_id, log_prob, state_id in branches:
            c_step = cost_model.step_cost(prompt, curr.token_ids, token_id)
            child = curr.extend(
                token_id=token_id,
                step_score=log_prob,
                step_cost=c_step,
                is_terminal=(token_id == config.eos_token_id),
                state_id=state_id,
            )
            stack.append(child)

    feasible = [c for c in all_terminals if validator.is_prefix_feasible(c)]
    if feasible:
        true_opt = max(feasible, key=lambda c: c.accumulated_score)
        return true_opt, all_terminals
    return None, all_terminals
