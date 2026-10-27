"""Hugging Face Transformers integration for Dual-Pruned Decoding (DPD).

Provides:
1. `DPDLogitsProcessor`: Native Hugging Face `LogitsProcessor` that applies
   adaptive Lagrangian resource penalties and hard-budget feasibility masks
   directly during standard `model.generate()`.
2. `DPDGenerationWrapper`: End-to-end wrapper for CausalLM models providing
   certified suboptimality bounds and 2D convex hull pruning.
"""

from __future__ import annotations

from typing import Any

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]
try:
    from transformers import LogitsProcessor
except ImportError:
    class LogitsProcessor:  # type: ignore[no-redef]
        """Fallback base class when transformers is not installed."""

from dpd.config import DPDConfig, MultiplierConfig
from dpd.cost_models import BaseCostModel, TokenCountCostModel
from dpd.decoding import dual_pruned_decoding
from dpd.multipliers import ProjectedSubgradientController
from dpd.types import DPDOutput


class DPDLogitsProcessor(LogitsProcessor):
    """Hugging Face LogitsProcessor implementing Lagrangian resource penalty and feasibility filtering.
    
    Can be passed directly to `model.generate(logits_processor=[processor])`.
    """

    def __init__(
        self,
        budget: float,
        cost_model: BaseCostModel,
        initial_lambda: float = 0.0,
        learning_rate: float = 0.05,
        max_lambda: float = 10.0,
        eos_token_id: int = 50256,
    ) -> None:
        super().__init__()
        self.budget = float(budget)
        self.cost_model = cost_model
        mult_cfg = MultiplierConfig(
            initial_lambda=initial_lambda,
            learning_rate=learning_rate,
            max_lambda=max_lambda,
        )
        self.controller = ProjectedSubgradientController(mult_cfg)
        self.eos_token_id = eos_token_id
        self.accumulated_costs: dict[int, float] = {}

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Applies Lagrangian price penalty and masks infeasible next tokens."""
        batch_size, vocab_size = scores.shape
        lam = self.controller.current_lambda

        for b in range(batch_size):
            prefix = input_ids[b].tolist()
            curr_cost = self.accumulated_costs.get(b, 0.0)

            # Infeasibility mask & price penalty
            for v in range(vocab_size):
                if v == self.eos_token_id:
                    continue
                step_c = self.cost_model.step_cost(None, prefix, v)
                min_c = self.cost_model.min_cost_to_eos(v)

                if curr_cost + step_c + min_c > self.budget + 1e-9:
                    scores[b, v] = float("-inf")
                elif lam > 0.0:
                    scores[b, v] -= float(lam * step_c)

        return scores


class DPDGenerationWrapper:
    """Wrapper around Hugging Face models providing a unified interface for Dual-Pruned Decoding."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        cost_model: BaseCostModel | None = None,
        config: DPDConfig | None = None,
        branch_top_k: int = 16,
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.cost_model = cost_model or TokenCountCostModel()
        self.config = config or DPDConfig()
        self.branch_top_k = branch_top_k
        self.device = device

    def generate(
        self,
        prompt: str | list[int],
        budget: float | None = None,
    ) -> DPDOutput:
        """Executes Dual-Pruned Decoding on the wrapped model."""
        target_budget = float(budget) if budget is not None else self.config.budget
        target_config = DPDConfig(
            beam_width=self.config.beam_width,
            max_steps=self.config.max_steps,
            budget=target_budget,
            pruning_mode=self.config.pruning_mode,
            multiplier=self.config.multiplier,
            hull=self.config.hull,
            eos_token_id=self.config.eos_token_id,
            pad_token_id=self.config.pad_token_id,
        )

        if isinstance(prompt, str):
            prompt_tokens = self.tokenizer.encode(prompt)
        else:
            prompt_tokens = list(prompt)

        def step_fn(
            _prompt: Any,
            prefix_token_ids: list[int],
        ) -> list[tuple[int, float, str | None]]:
            if torch is None:
                raise ImportError("PyTorch is required for DPDGenerationWrapper.")
            full_ids = prompt_tokens + prefix_token_ids if prefix_token_ids else prompt_tokens
            input_tensor = torch.tensor([full_ids], dtype=torch.long, device=self.device)
            with torch.no_grad():
                outputs = self.model(input_tensor)
                next_token_logits = outputs.logits[0, -1, :]
                log_probs = torch.log_softmax(next_token_logits, dim=-1)
                top_log_probs, top_indices = torch.topk(log_probs, k=self.branch_top_k)

            return [
                (int(idx), float(lp), None)
                for lp, idx in zip(top_log_probs.tolist(), top_indices.tolist())
            ]

        return dual_pruned_decoding(
            prompt=prompt_tokens,
            config=target_config,
            cost_model=self.cost_model,
            step_fn=step_fn,
        )
