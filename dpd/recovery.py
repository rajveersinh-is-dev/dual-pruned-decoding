"""Recovery mechanisms and infeasibility handling for budget-constrained decoding."""

from __future__ import annotations

from dpd.types import PrefixCandidate


class RecoveryManager:
    """Manages fallback and recovery when candidate beams approach infeasibility."""

    def __init__(self, budget: float, eps: float = 1e-9):
        self.budget = budget
        self.eps = eps

    def select_recovery_candidates(
        self,
        current_beam: list[PrefixCandidate],
        incumbent: PrefixCandidate | None,
        beam_width: int,
    ) -> tuple[list[PrefixCandidate], bool]:
        """Selects candidates to continue when standard pruning eliminates all candidates.
        
        Policy:
        1. If a feasible incumbent already exists, return empty list to trigger immediate
           graceful termination (since no current candidate can complete feasibly).
        2. If NO feasible incumbent exists, select the candidate(s) that minimize total
           budget violation max(0, Cost - Budget) to provide the least-infeasible fallback.
           
        Returns:
            Tuple of (recovered_candidates, should_terminate).
        """
        if incumbent is not None and incumbent.accumulated_cost <= self.budget + self.eps:
            # We already have a valid feasible solution; safe to terminate
            return [], True

        if not current_beam:
            return [], True

        # Sort candidates by minimum violation first, then highest score
        def violation_score_key(c: PrefixCandidate) -> tuple[float, float]:
            violation = max(0.0, c.accumulated_cost - self.budget)
            return (violation, -c.accumulated_score)

        sorted_candidates = sorted(current_beam, key=violation_score_key)
        recovered = sorted_candidates[:beam_width]
        return recovered, False

    def get_best_fallback(
        self,
        all_explored_candidates: list[PrefixCandidate],
    ) -> PrefixCandidate | None:
        """Returns the least-violating candidate from all explored candidates."""
        if not all_explored_candidates:
            return None

        def key_fn(c: PrefixCandidate) -> tuple[float, float]:
            violation = max(0.0, c.accumulated_cost - self.budget)
            return (violation, -c.accumulated_score)

        return min(all_explored_candidates, key=key_fn)
