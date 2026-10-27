"""Structured code and arithmetic expression generation task with non-uniform resource costs."""

from __future__ import annotations

from typing import Any

from dpd.cost_models import BaseCostModel


class StructuredCodeTask:
    """A grammar-structured code and expression generation environment.
    
    Tokens:
      0: EOS (cost 0.0)
      1: "def" (cost 2.0, function definition)
      2: "f(x):" (cost 1.5, signature)
      3: "return" (cost 1.0, return statement)
      4: "x" (cost 0.5, variable)
      5: "+" (cost 0.5, operator)
      6: "*" (cost 0.8, operator)
      7: "f(x-1)" (cost 3.5, expensive recursive call)
      8: "1" (cost 0.5, base case literal)
      9: "if x <= 1:" (cost 1.5, conditional guard)
    """

    def __init__(self):
        self.eos_token_id = 0
        self.token_names = {
            0: "EOS",
            1: "def",
            2: "f(x):",
            3: "return",
            4: "x",
            5: "+",
            6: "*",
            7: "f(x-1)",
            8: "1",
            9: "if x <= 1:",
        }
        self.costs = {
            0: 0.0,
            1: 2.0,
            2: 1.5,
            3: 1.0,
            4: 0.5,
            5: 0.5,
            6: 0.8,
            7: 3.5,
            8: 0.5,
            9: 1.5,
        }

    def step_fn(
        self,
        prompt: Any,
        prefix_token_ids: list[int],
    ) -> list[tuple[int, float, str | None]]:
        """Returns valid syntactic extensions with log-probabilities."""
        length = len(prefix_token_ids)
        if length == 0:
            # Must start with def
            return [(1, -0.05, "header")]
        if length == 1:
            # def -> f(x):
            return [(2, -0.05, "sig")]
        if length == 2:
            # Can either do guard 'if x <= 1:' (leading to base case) or directly 'return'
            return [
                (9, -0.4, "guard"),    # if x <= 1:
                (3, -1.2, "return"),   # return ...
            ]
        if prefix_token_ids[-1] == 9:
            # After guard -> return 1
            return [(3, -0.1, "ret_base")]
        if prefix_token_ids[-2:] == [9, 3]:
            return [(8, -0.1, "base_lit")]
        if prefix_token_ids[-3:] == [9, 3, 8]:
            # Next line: return x * f(x-1) (recursive) or return x (linear)
            return [(3, -0.2, "ret_rec")]
        if prefix_token_ids[-1] == 3 and prefix_token_ids[-4:] != [9, 3, 8, 3]:
            # Simple return
            return [
                (4, -0.3, "ret_x"),      # return x
                (8, -0.8, "ret_1"),      # return 1
                (7, -1.8, "ret_rec"),    # return f(x-1)
            ]
        if prefix_token_ids[-1] == 4:
            # After x: operator or EOS
            return [
                (5, -0.4, "plus"),
                (6, -0.6, "mult"),
                (0, -0.5, "eos"),
            ]
        if prefix_token_ids[-1] in (5, 6):
            # After operator: x, 1, or recursive call f(x-1)
            return [
                (4, -0.5, "operand_x"),
                (8, -0.5, "operand_1"),
                (7, -0.8, "operand_rec"),  # expensive
            ]
        if prefix_token_ids[-1] == 7:
            # After f(x-1) -> EOS
            return [(0, -0.1, "eos")]
        if prefix_token_ids[-1] == 8:
            return [(0, -0.2, "eos")]

        # Default fallback
        return [(0, 0.0, "eos")]


class StructuredCodeCostModel(BaseCostModel):
    """Cost model assigning non-uniform execution costs to code tokens."""

    def __init__(self, task: StructuredCodeTask):
        self.task = task

    def step_cost(self, prompt: Any, prefix_token_ids: list[int], next_token_id: int) -> float:
        return float(self.task.costs.get(next_token_id, 1.0))

    def min_cost_to_eos(self, next_token_id: int) -> float:
        return 0.0
