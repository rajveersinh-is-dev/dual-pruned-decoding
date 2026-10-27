from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from dpd.config import DPDConfig
from dpd.cost_models import TokenCountCostModel
from dpd.hf_integration import DPDGenerationWrapper, DPDLogitsProcessor


def test_dpd_logits_processor_penalty_and_mask():
    cost_model = TokenCountCostModel(eos_token_id=0, zero_cost_tokens={0})
    processor = DPDLogitsProcessor(
        budget=2.0,
        cost_model=cost_model,
        initial_lambda=1.5,
        eos_token_id=0,
    )

    batch_size = 1
    vocab_size = 5
    # Token 0 is EOS, Tokens 1..4 have cost 1.0 each
    input_ids = torch.zeros((batch_size, 1), dtype=torch.long)
    scores = torch.zeros((batch_size, vocab_size), dtype=torch.float)

    # Set accumulated cost to 1.5
    processor.accumulated_costs[0] = 1.5

    # If step cost is 1.0, 1.5 + 1.0 = 2.5 > budget 2.0 -> should be masked to -inf
    modified_scores = processor(input_ids, scores.clone())

    # Non-EOS tokens (1..4) exceed budget and should be -inf
    for v in range(1, vocab_size):
        assert torch.isneginf(modified_scores[0, v]), f"Token {v} should be masked to -inf"

    # EOS token (0) should remain 0.0 (unmasked)
    assert not torch.isneginf(modified_scores[0, 0])


def test_dpd_generation_wrapper_init():
    dummy_model = torch.nn.Linear(10, 10)
    dummy_tokenizer = object()
    config = DPDConfig(beam_width=3, budget=12.0)

    wrapper = DPDGenerationWrapper(
        model=dummy_model,
        tokenizer=dummy_tokenizer,
        config=config,
    )
    assert wrapper.config.beam_width == 3
    assert wrapper.config.budget == 12.0
