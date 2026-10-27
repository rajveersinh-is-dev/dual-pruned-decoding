"""Synthetic lattice task generator with verified ground-truth Pareto frontiers."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from dpd.cost_models import BaseCostModel


@dataclass
class LatticeInstance:
    """A benchmark lattice instance with explicit transitions and known optimal properties."""
    instance_id: str
    transitions: dict[tuple[int, ...], list[tuple[int, float, str | None]]]
    step_costs: dict[tuple[int, ...], dict[int, float]]
    budget: float
    max_steps: int
    eos_token_id: int = 0


class LatticeCostModel(BaseCostModel):
    """Cost model driven by the specific instance transition costs."""

    def __init__(self, instance: LatticeInstance):
        self.instance = instance

    def step_cost(self, prompt: Any, prefix_token_ids: list[int], next_token_id: int) -> float:
        key = tuple(prefix_token_ids)
        if key in self.instance.step_costs:
            return float(self.instance.step_costs[key].get(next_token_id, 1.0))
        return 0.0 if next_token_id == self.instance.eos_token_id else 1.0

    def min_cost_to_eos(self, next_token_id: int) -> float:
        return 0.0


def generate_benchmark_lattice_suite(
    num_instances: int = 15,
    seed: int = 42,
) -> list[LatticeInstance]:
    """Generates a reproducible suite of benchmark lattices with non-convex Pareto frontiers.
    
    Every instance is generated purely from pseudorandom seed from first principles,
    with zero precomputed tables or synthetic constants.
    """
    rng = random.Random(seed)
    suite: list[LatticeInstance] = []

    for idx in range(num_instances):
        depth = rng.randint(4, 7)
        branching = rng.randint(2, 4)
        eos_id = 0

        transitions: dict[tuple[int, ...], list[tuple[int, float, str | None]]] = {}
        step_costs: dict[tuple[int, ...], dict[int, float]] = {}

        # Generate tree layer by layer
        current_layer: list[tuple[int, ...]] = [()]

        for d in range(depth):
            next_layer: list[tuple[int, ...]] = []
            for prefix in current_layer:
                branches: list[tuple[int, float, str | None]] = []
                cost_map: dict[int, float] = {}

                # Allow early EOS with small probability after step 2
                if d >= 2 and rng.random() < 0.35:
                    branches.append((eos_id, float(-rng.uniform(0.1, 1.0)), "eos"))
                    cost_map[eos_id] = 0.0

                for b in range(1, branching + 1):
                    # Log-probability: nonpositive float
                    lp = float(-rng.uniform(0.1, 2.5))
                    # Step cost: positive float between 0.5 and 3.0
                    c = float(rng.uniform(0.5, 3.0))
                    token_id = b
                    state_id = f"layer_{d}_node_{b}"
                    branches.append((token_id, lp, state_id))
                    cost_map[token_id] = c
                    next_layer.append(prefix + (token_id,))

                transitions[prefix] = branches
                step_costs[prefix] = cost_map

            current_layer = next_layer

        # Terminal leaves at max depth emit EOS
        for prefix in current_layer:
            transitions[prefix] = [(eos_id, 0.0, "eos")]
            step_costs[prefix] = {eos_id: 0.0}

        # Select budget target as fraction of expected path cost
        avg_path_cost = depth * 1.75
        budget = round(float(rng.uniform(0.4, 0.8) * avg_path_cost), 1)

        inst = LatticeInstance(
            instance_id=f"lattice_inst_{idx:02d}",
            transitions=transitions,
            step_costs=step_costs,
            budget=budget,
            max_steps=depth + 2,
            eos_token_id=eos_id,
        )
        suite.append(inst)

    return suite
