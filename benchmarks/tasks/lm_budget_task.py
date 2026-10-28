"""Language model budget-constrained generation task using DistilGPT-2."""

from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from dpd.cost_models import TokenCountCostModel
from dpd.decoding import StepFunction


class LMGenerationEnvironment:
    """Wraps an autoregressive HuggingFace causal LM for step-by-step search decoding."""

    def __init__(
        self,
        model_name: str = "distilbert/distilgpt2",
        branch_top_k: int = 8,
        device: str = "cpu",
    ):
        self.device = device
        self.branch_top_k = branch_top_k
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.model.eval()
        self.eos_token_id = self.tokenizer.eos_token_id or 50256
        self.cost_model = TokenCountCostModel(
            eos_token_id=self.eos_token_id,
            zero_cost_tokens={self.eos_token_id},
        )

    def create_step_fn(self, prompt_token_ids: list[int]) -> StepFunction:
        """Creates a step function bound to a specific prompt prefix."""
        torch.tensor([prompt_token_ids], dtype=torch.long, device=self.device)

        def step_fn(
            prompt: Any,
            prefix_token_ids: list[int],
        ) -> list[tuple[int, float, str | None]]:
            if prefix_token_ids:
                full_ids = prompt_token_ids + prefix_token_ids
            else:
                full_ids = prompt_token_ids

            input_tensor = torch.tensor([full_ids], dtype=torch.long, device=self.device)
            with torch.no_grad():
                outputs = self.model(input_tensor)
                next_token_logits = outputs.logits[0, -1, :]
                log_probs = torch.log_softmax(next_token_logits, dim=-1)

                # Select top-k branches
                top_log_probs, top_indices = torch.topk(log_probs, k=self.branch_top_k)

            branches: list[tuple[int, float, str | None]] = []
            top_set = set(top_indices.tolist())
            for lp, idx in zip(top_log_probs.tolist(), top_indices.tolist()):
                branches.append((int(idx), float(lp), None))

            # Ensure the EOS transition is always accessible with its exact model log-probability,
            # allowing the decoder to legally terminate within budget constraints.
            if self.eos_token_id not in top_set:
                eos_lp = float(log_probs[self.eos_token_id].item())
                branches.append((self.eos_token_id, eos_lp, None))

            return branches

        return step_fn


def get_standard_evaluation_prompts() -> list[str]:
    """Returns a curated suite of diverse evaluation prompts for budget-constrained generation."""
    return [
        "The primary goal of machine learning is to",
        "In quantum physics, the uncertainty principle states that",
        "The algorithm begins by traversing the directed graph and",
        "To optimize the energy efficiency of data centers, engineers",
        "A major challenge in natural language understanding is that",
        "The economic implications of artificial intelligence include",
        "Photosynthesis is the biological process by which plants",
        "In distributed computing systems, the consensus problem requires",
        "The history of renewable energy technologies demonstrates",
        "To ensure robust software security, modern cryptographic protocols",
    ]
