"""Extensible cost models for resource-constrained sequence generation."""

from __future__ import annotations

from typing import Any, Callable


class BaseCostModel:
    """Base class for step-level resource cost models."""

    def step_cost(
        self,
        prompt: Any,
        prefix_token_ids: list[int],
        next_token_id: int,
    ) -> float:
        """Computes the incremental resource cost of appending next_token_id to prefix."""
        raise NotImplementedError

    def minimum_step_cost(self) -> float:
        """Returns a globally valid lower bound on the cost of any single step."""
        return 0.0

    def min_cost_to_eos(self, next_token_id: int) -> float:
        """Lower bound on additional cost required to reach EOS from next_token_id."""
        return 0.0


class TokenCountCostModel(BaseCostModel):
    """Uniform token budget cost model (each token costs 1 unit)."""

    def __init__(self, eos_token_id: int = 2, zero_cost_tokens: set[int] | None = None):
        self.eos_token_id = eos_token_id
        self.zero_cost_tokens = zero_cost_tokens or set()

    def step_cost(
        self,
        prompt: Any,
        prefix_token_ids: list[int],
        next_token_id: int,
    ) -> float:
        if next_token_id in self.zero_cost_tokens:
            return 0.0
        return 1.0

    def minimum_step_cost(self) -> float:
        return 0.0 if self.zero_cost_tokens else 1.0

    def min_cost_to_eos(self, next_token_id: int) -> float:
        if next_token_id == self.eos_token_id:
            return 0.0
        # To terminate, at least the EOS token must be emitted
        return 0.0 if self.eos_token_id in self.zero_cost_tokens else 1.0


class CharacterCostModel(BaseCostModel):
    """Resource cost proportional to token character length (e.g. byte/char budget)."""

    def __init__(
        self,
        token_to_char_len: dict[int, int],
        eos_token_id: int = 2,
        default_len: int = 4,
    ):
        self.token_to_char_len = token_to_char_len
        self.eos_token_id = eos_token_id
        self.default_len = default_len

    def step_cost(
        self,
        prompt: Any,
        prefix_token_ids: list[int],
        next_token_id: int,
    ) -> float:
        return float(self.token_to_char_len.get(next_token_id, self.default_len))

    def minimum_step_cost(self) -> float:
        if self.token_to_char_len:
            return float(min(self.token_to_char_len.values()))
        return 1.0


class MonetaryCostModel(BaseCostModel):
    """Monetary inference pricing model (e.g., dollars per 1k tokens)."""

    def __init__(
        self,
        cost_per_token: float = 0.00002,  # $0.02 per 1000 tokens
        eos_token_id: int = 2,
    ):
        self.cost_per_token = cost_per_token
        self.eos_token_id = eos_token_id

    def step_cost(
        self,
        prompt: Any,
        prefix_token_ids: list[int],
        next_token_id: int,
    ) -> float:
        return self.cost_per_token

    def minimum_step_cost(self) -> float:
        return self.cost_per_token


class CallableCostModel(BaseCostModel):
    """Wraps an arbitrary user-defined callable as a cost model."""

    def __init__(
        self,
        cost_fn: Callable[[Any, list[int], int], float],
        min_step: float = 0.0,
    ):
        self.cost_fn = cost_fn
        self.min_step = min_step

    def step_cost(
        self,
        prompt: Any,
        prefix_token_ids: list[int],
        next_token_id: int,
    ) -> float:
        val = float(self.cost_fn(prompt, prefix_token_ids, next_token_id))
        if val < 0.0:
            raise ValueError(f"Step cost must be non-negative, got {val}")
        return val

    def minimum_step_cost(self) -> float:
        return self.min_step
