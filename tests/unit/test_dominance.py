"""Unit tests for dominance filtering routines."""

from dpd.bounds import TrivialAdmissibleBound
from dpd.dominance import filter_by_incumbent_bound, filter_state_matched_dominance
from dpd.types import PrefixCandidate


def test_state_matched_dominance():
    """State-matched candidates: lower cost and higher score dominates."""
    c1 = PrefixCandidate(token_ids=[1], accumulated_score=-1.0, accumulated_cost=2.0, step=1, state_id="sA")
    c2 = PrefixCandidate(token_ids=[2], accumulated_score=-2.5, accumulated_cost=3.0, step=1, state_id="sA")  # Dominated by c1
    c3 = PrefixCandidate(token_ids=[3], accumulated_score=-0.5, accumulated_cost=5.0, step=1, state_id="sA")  # Pareto non-dominated (higher cost, higher score)
    c4 = PrefixCandidate(token_ids=[4], accumulated_score=-3.0, accumulated_cost=1.0, step=1, state_id="sB")  # Different state, kept

    survivors, pruned_count = filter_state_matched_dominance([c1, c2, c3, c4])
    assert pruned_count == 1
    assert len(survivors) == 3
    token_ids = [c.token_ids[0] for c in survivors]
    assert 2 not in token_ids  # c2 was pruned
    assert 1 in token_ids
    assert 3 in token_ids
    assert 4 in token_ids


def test_filter_by_incumbent_bound():
    """Candidates whose score + U(u) <= incumbent_score are pruned."""
    bounds = TrivialAdmissibleBound()  # U(u) = 0.0
    c1 = PrefixCandidate(token_ids=[1], accumulated_score=-2.0, accumulated_cost=1.0, step=1)
    c2 = PrefixCandidate(token_ids=[2], accumulated_score=-6.0, accumulated_cost=1.0, step=1)

    incumbent_score = -4.0
    survivors, pruned_count = filter_by_incumbent_bound(
        candidates=[c1, c2],
        incumbent_score=incumbent_score,
        bound_manager=bounds,
        max_steps=10,
        budget=10.0,
    )
    assert pruned_count == 1
    assert len(survivors) == 1
    assert survivors[0] == c1  # c1 survives since -2.0 + 0.0 > -4.0; c2 pruned since -6.0 <= -4.0
