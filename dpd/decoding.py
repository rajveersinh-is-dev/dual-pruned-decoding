"""Core Dual-Pruned Decoding (DPD) search engine."""

from __future__ import annotations

from typing import Any, Callable, Optional

from dpd.bounds import BaseCompletionBound, TrivialAdmissibleBound
from dpd.certificates import compute_dual_certificate
from dpd.config import DPDConfig
from dpd.constraints import BudgetConstraintValidator
from dpd.cost_models import BaseCostModel
from dpd.dominance import filter_by_incumbent_bound, filter_state_matched_dominance
from dpd.hull import compute_upper_left_hull
from dpd.instrumentation import HighResolutionProfiler
from dpd.multipliers import create_multiplier_controller
from dpd.recovery import RecoveryManager
from dpd.types import (
    CostScorePoint,
    DecodingResult,
    DecodingTelemetry,
    PrefixCandidate,
    PruningMode,
)

# Step function signature: (prompt, prefix_token_ids) -> List[Tuple[token_id, log_prob, optional_state_id]]
StepFunction = Callable[
    [Any, list[int]],
    list[tuple[int, float, Optional[str]]],
]


def dual_pruned_decoding(
    prompt: Any,
    config: DPDConfig,
    cost_model: BaseCostModel,
    step_fn: StepFunction,
    bound_manager: BaseCompletionBound | None = None,
    initial_tokens: list[int] | None = None,
) -> DecodingResult:
    """Executes Dual-Pruned Decoding (DPD) for budget-constrained autoregressive generation.
    
    Args:
        prompt: Task prompt or prompt token representation.
        config: Master DPD configuration.
        cost_model: Resource cost model C(x, y_<t, y_t).
        step_fn: Autoregressive candidate generator returning (token_id, log_prob, state_id).
        bound_manager: Admissible future completion bound provider.
        initial_tokens: Optional prefix tokens to start generation from.
        
    Returns:
        DecodingResult containing the best sequence, feasibility flag, dual certificate,
        and full instrumentation telemetry.
    """
    config.validate()
    bounds = bound_manager or TrivialAdmissibleBound()
    validator = BudgetConstraintValidator(config.budget, cost_model)
    recovery_mgr = RecoveryManager(config.budget)
    controller = create_multiplier_controller(config.multiplier)
    telemetry = DecodingTelemetry()

    t_start = HighResolutionProfiler.now_ns()

    # Initial candidate
    init_toks = initial_tokens or []
    root = PrefixCandidate(
        token_ids=init_toks,
        accumulated_score=0.0,
        accumulated_cost=0.0,
        step=0,
        is_terminal=(len(init_toks) > 0 and init_toks[-1] == config.eos_token_id),
        metadata={"budget": config.budget},
    )

    beam: list[PrefixCandidate] = [root]
    incumbent: PrefixCandidate | None = None
    all_terminal_candidates: list[PrefixCandidate] = []
    all_explored_candidates: list[PrefixCandidate] = [root]

    # Check if initial root is already a feasible terminal
    if root.is_terminal:
        all_terminal_candidates.append(root)
        if validator.is_prefix_feasible(root):
            incumbent = root

    termination_reason = "max_steps"

    for step in range(1, config.max_steps + 1):
        step_t0 = HighResolutionProfiler.now_ns()

        # Check if all candidates in beam are terminal
        active_candidates = [c for c in beam if not c.is_terminal]
        if not active_candidates:
            termination_reason = "all_terminal"
            break

        # Step 1: Candidate Expansion
        expanded_candidates: list[PrefixCandidate] = []
        for parent in active_candidates:
            branches = step_fn(prompt, parent.token_ids)
            telemetry.total_forward_passes += 1
            for token_id, log_prob, state_id in branches:
                c_step = cost_model.step_cost(prompt, parent.token_ids, token_id)
                is_term = (token_id == config.eos_token_id)
                child = parent.extend(
                    token_id=token_id,
                    step_score=log_prob,
                    step_cost=c_step,
                    is_terminal=is_term,
                    state_id=state_id,
                )
                expanded_candidates.append(child)
                all_explored_candidates.append(child)

        telemetry.total_candidates_generated += len(expanded_candidates)
        telemetry.step_candidates_expanded.append(len(expanded_candidates))

        # Separate terminals and update incumbent
        current_terminals = [c for c in expanded_candidates if c.is_terminal]
        all_terminal_candidates.extend(current_terminals)

        for term in current_terminals:
            if validator.is_prefix_feasible(term) and (
                incumbent is None or term.accumulated_score > incumbent.accumulated_score
            ):
                incumbent = term

        # Retain active (non-terminal) candidates for pruning and filtering
        active_expanded = [c for c in expanded_candidates if not c.is_terminal]
        if not active_expanded:
            # Only terminals generated in this step
            beam = [c for c in expanded_candidates if c.is_terminal]
            step_t1 = HighResolutionProfiler.now_ns()
            telemetry.step_durations_ns.append(step_t1 - step_t0)
            continue

        # Step 2: Pruning by Infeasibility
        feasible_lookahead: list[PrefixCandidate] = []
        infeasible_count = 0
        for c in active_expanded:
            if validator.can_complete_feasibly(c):
                feasible_lookahead.append(c)
            else:
                infeasible_count += 1

        telemetry.step_candidates_pruned_infeasibility.append(infeasible_count)
        telemetry.total_candidates_pruned += infeasible_count

        # Step 3: Pruning by Incumbent Bound
        incumbent_score = incumbent.accumulated_score if incumbent else -float("inf")
        bound_survivors, bound_pruned_count = filter_by_incumbent_bound(
            feasible_lookahead,
            incumbent_score,
            bounds,
            config.max_steps,
            config.budget,
        )
        telemetry.step_candidates_pruned_bound.append(bound_pruned_count)
        telemetry.total_candidates_pruned += bound_pruned_count

        # Step 4: State-Matched Dominance Pruning
        dom_survivors, dom_pruned_count = filter_state_matched_dominance(bound_survivors)
        telemetry.step_candidates_pruned_dominance.append(dom_pruned_count)
        telemetry.total_candidates_pruned += dom_pruned_count

        # Recovery check if all candidates were eliminated
        if not dom_survivors:
            recovered, should_terminate = recovery_mgr.select_recovery_candidates(
                active_expanded, incumbent, config.beam_width
            )
            if should_terminate or not recovered:
                termination_reason = "infeasibility_pruned"
                break
            dom_survivors = recovered

        # Step 5: Multiplier Update
        avg_cost = sum(c.accumulated_cost for c in dom_survivors) / len(dom_survivors)
        current_lambda = controller.update(step, config.max_steps, config.budget, avg_cost)
        telemetry.step_multipliers.append(current_lambda)
        telemetry.step_average_costs.append(avg_cost)

        # Step 6: Geometric Convex Hull and Beam Selection
        points = [
            CostScorePoint(
                cost=c.accumulated_cost,
                score=c.accumulated_score,
                candidate_id=idx,
            )
            for idx, c in enumerate(dom_survivors)
        ]

        hull_vertices = compute_upper_left_hull(points, config.hull)
        hull_indices = {pt.candidate_id for pt in hull_vertices}

        hull_pruned_count = len(points) - len(hull_indices)
        telemetry.step_candidates_pruned_hull.append(hull_pruned_count)

        if config.pruning_mode == PruningMode.HULL_EXTREMES_ONLY:
            # Retain only hull vertices
            selected_candidates = [dom_survivors[idx] for idx in hull_indices]
            # If hull vertices exceed beam width, rank by Lagrangian scalarization S - lambda * C
            if len(selected_candidates) > config.beam_width:
                selected_candidates.sort(
                    key=lambda c: c.accumulated_score - current_lambda * c.accumulated_cost,
                    reverse=True,
                )
                selected_candidates = selected_candidates[: config.beam_width]

        elif config.pruning_mode == PruningMode.HYBRID_HULL_AND_BOUND:
            # Prioritize hull vertices, then fill remaining slots with highest scalarized non-hull candidates
            hull_candidates = [dom_survivors[idx] for idx in hull_indices]
            hull_candidates.sort(
                key=lambda c: c.accumulated_score - current_lambda * c.accumulated_cost,
                reverse=True,
            )
            selected_candidates = hull_candidates[: config.beam_width]

            if len(selected_candidates) < config.beam_width:
                remaining_slots = config.beam_width - len(selected_candidates)
                non_hull_candidates = [
                    dom_survivors[idx] for idx in range(len(dom_survivors))
                    if idx not in hull_indices
                ]
                non_hull_candidates.sort(
                    key=lambda c: c.accumulated_score - current_lambda * c.accumulated_cost,
                    reverse=True,
                )
                selected_candidates.extend(non_hull_candidates[:remaining_slots])

        elif config.pruning_mode == PruningMode.EXACT_BOUND_AND_DOMINANCE:
            # Rank strictly by scalarized score without geometric hull filtering
            dom_survivors.sort(
                key=lambda c: c.accumulated_score - current_lambda * c.accumulated_cost,
                reverse=True,
            )
            selected_candidates = dom_survivors[: config.beam_width]

        else:  # NO_PRUNING / plain ranking
            dom_survivors.sort(key=lambda c: c.accumulated_score, reverse=True)
            selected_candidates = dom_survivors[: config.beam_width]

        beam = selected_candidates
        step_t1 = HighResolutionProfiler.now_ns()
        telemetry.step_durations_ns.append(step_t1 - step_t0)

    t_end = HighResolutionProfiler.now_ns()
    telemetry.total_wall_clock_time_sec = (t_end - t_start) * 1e-9

    # Determine final output and compute certificate
    all_feasible = [
        c for c in all_terminal_candidates
        if validator.is_prefix_feasible(c)
    ]

    if incumbent is not None:
        best_cand = incumbent
        is_feasible = True
    elif all_feasible:
        best_cand = max(all_feasible, key=lambda c: c.accumulated_score)
        is_feasible = True
    else:
        # Fallback to least-violating candidate
        fallback = recovery_mgr.get_best_fallback(all_explored_candidates)
        best_cand = fallback if fallback else root
        is_feasible = False
        termination_reason = "infeasible_fallback"

    certificate = compute_dual_certificate(
        all_terminal_candidates=all_terminal_candidates,
        budget=config.budget,
        incumbent=best_cand if is_feasible else None,
        tolerance=config.optimality_tolerance,
    )

    return DecodingResult(
        best_sequence=best_cand.token_ids,
        best_score=best_cand.accumulated_score,
        best_cost=best_cand.accumulated_cost,
        is_feasible=is_feasible,
        budget=config.budget,
        certificate=certificate,
        all_feasible_sequences=all_feasible,
        telemetry=telemetry,
        termination_reason=termination_reason,
    )
