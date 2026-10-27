"""Dominance filtering and incumbent-based branch-and-bound pruning."""

from __future__ import annotations

from dpd.bounds import BaseCompletionBound
from dpd.types import PrefixCandidate


def filter_state_matched_dominance(
    candidates: list[PrefixCandidate],
    eps: float = 1e-9,
) -> tuple[list[PrefixCandidate], int]:
    """Prunes candidates that are strictly dominated by another candidate with identical forward state.
    
    Theorem 3 (State Dominance): If candidate u1 and u2 have identical state_id (u1 ~ u2),
    and C(u1) <= C(u2) and S(u1) >= S(u2), then every feasible completion of u2 is matched
    or exceeded by u1. Thus u2 can be pruned without loss of optimality.
    
    Returns:
        Tuple of (surviving_candidates, number_pruned).
    """
    if len(candidates) <= 1:
        return candidates, 0

    # Group by state_id (if state_id is None, no state-merging assumption is made)
    grouped: dict[str, list[PrefixCandidate]] = {}
    ungrouped: list[PrefixCandidate] = []

    for c in candidates:
        if c.state_id is not None:
            grouped.setdefault(c.state_id, []).append(c)
        else:
            ungrouped.append(c)

    survivors: list[PrefixCandidate] = list(ungrouped)
    pruned_count = 0

    for group in grouped.values():
        if len(group) == 1:
            survivors.append(group[0])
            continue

        # Sort by Cost ascending, then Score descending
        sorted_group = sorted(
            group,
            key=lambda c: (c.accumulated_cost, -c.accumulated_score),
        )
        non_dominated: list[PrefixCandidate] = []
        best_score_seen = -float("inf")

        for c in sorted_group:
            # For c to survive, it must have a strictly higher score than all earlier
            # (cheaper or equal cost) candidates in this state.
            if c.accumulated_score > best_score_seen + eps:
                non_dominated.append(c)
                best_score_seen = c.accumulated_score
            else:
                pruned_count += 1

        survivors.extend(non_dominated)

    return survivors, pruned_count


def filter_by_incumbent_bound(
    candidates: list[PrefixCandidate],
    incumbent_score: float,
    bound_manager: BaseCompletionBound,
    max_steps: int,
    budget: float,
    eps: float = 1e-9,
) -> tuple[list[PrefixCandidate], int]:
    """Prunes candidates whose admissible upper bound cannot beat the current feasible incumbent.
    
    Theorem 3 (Bound Condition): If S(u) + U(u) <= S_inc, no feasible completion of u
    can strictly improve upon the incumbent.
    
    Returns:
        Tuple of (surviving_candidates, number_pruned).
    """
    if incumbent_score == -float("inf") or not candidates:
        return candidates, 0

    survivors: list[PrefixCandidate] = []
    pruned_count = 0

    for c in candidates:
        ub = bound_manager.upper_bound_score(c, max_steps, budget)
        max_possible_score = c.accumulated_score + ub
        if max_possible_score > incumbent_score + eps:
            survivors.append(c)
        else:
            pruned_count += 1

    return survivors, pruned_count
