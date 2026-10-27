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


def test_dpd_logits_processor_multistep_manual_tracking():
    """Verifies that record_selected_token correctly tracks cost and adapts lambda across steps."""
    cost_model = TokenCountCostModel(eos_token_id=0, zero_cost_tokens={0})
    processor = DPDLogitsProcessor(
        budget=2.0,
        cost_model=cost_model,
        initial_lambda=0.0,
        learning_rate=0.5,
        max_steps=4,
        eos_token_id=0,
    )

    batch_size = 1
    vocab_size = 4
    input_ids = torch.zeros((batch_size, 1), dtype=torch.long)
    scores = torch.zeros((batch_size, vocab_size), dtype=torch.float)

    # Step 1: processor called, lambda is initially 0.0
    s1 = processor(input_ids, scores.clone())
    assert processor.controller.current_lambda == 0.0
    # No price penalty at lambda=0.0
    assert torch.allclose(s1, scores)

    # Select token 1 (cost 1.0)
    new_lam = processor.record_selected_token(batch_idx=0, token_id=1)
    assert processor.accumulated_costs[0] == 1.0
    # Expected progress target: budget * (1 / 4) = 0.5. Violation = 1.0 - 0.5 = +0.5
    # Controller should increase lambda
    assert new_lam > 0.0

    # Step 2: processor called with increased lambda -> price penalty applied
    input_ids_step2 = torch.tensor([[0, 1]], dtype=torch.long)
    s2 = processor(input_ids_step2, scores.clone())
    # Non-EOS tokens 1..3 should have negative score due to -lambda * step_c
    for v in range(1, vocab_size):
        assert s2[0, v] < 0.0

    # Select token 2 (cost 1.0) -> accumulated cost reaches 2.0 (budget limit)
    processor.record_selected_token(batch_idx=0, token_id=2)
    assert processor.accumulated_costs[0] == 2.0

    # Step 3: further non-EOS tokens cost 1.0, 2.0 + 1.0 = 3.0 > budget 2.0 -> masked to -inf
    input_ids_step3 = torch.tensor([[0, 1, 2]], dtype=torch.long)
    s3 = processor(input_ids_step3, scores.clone())
    for v in range(1, vocab_size):
        assert torch.isneginf(s3[0, v]), f"Token {v} must be masked to -inf when budget exhausted"
    # EOS token (0) remains unmasked
    assert not torch.isneginf(s3[0, 0])


def test_dpd_logits_processor_multistep_auto_tracking():
    """Verifies that DPDLogitsProcessor automatically detects newly appended tokens in input_ids."""
    cost_model = TokenCountCostModel(eos_token_id=0, zero_cost_tokens={0})
    processor = DPDLogitsProcessor(
        budget=2.0,
        cost_model=cost_model,
        initial_lambda=0.0,
        learning_rate=0.5,
        max_steps=4,
        eos_token_id=0,
    )

    vocab_size = 4
    scores = torch.zeros((1, vocab_size), dtype=torch.float)

    # Step 1: prompt has 1 token
    input_ids_1 = torch.tensor([[100]], dtype=torch.long)
    processor(input_ids_1, scores.clone())
    assert processor.accumulated_costs.get(0, 0.0) == 0.0

    # Step 2: generate() appended token 1 (cost 1.0)
    input_ids_2 = torch.tensor([[100, 1]], dtype=torch.long)
    processor(input_ids_2, scores.clone())
    assert processor.accumulated_costs[0] == 1.0
    assert processor.controller.current_lambda > 0.0

    # Step 3: generate() appended token 2 (cost 1.0)
    input_ids_3 = torch.tensor([[100, 1, 2]], dtype=torch.long)
    s3 = processor(input_ids_3, scores.clone())
    assert processor.accumulated_costs[0] == 2.0
    # Next tokens 1..3 should now be masked to -inf
    for v in range(1, vocab_size):
        assert torch.isneginf(s3[0, v])
    assert not torch.isneginf(s3[0, 0])


def test_dpd_generation_wrapper_generate():
    """Verifies end-to-end execution of DPDGenerationWrapper.generate() with a mock model."""
    class DummyTokenizer:
        def encode(self, text: str) -> list[int]:
            return [10]

    class DummyModelOutput:
        def __init__(self, logits: torch.Tensor):
            self.logits = logits

    class DummyModel(torch.nn.Module):
        def forward(self, input_tensor: torch.Tensor) -> DummyModelOutput:
            batch, seq_len = input_tensor.shape
            vocab_size = 8
            # Construct deterministic logits where token 0 is EOS and token 1 is preferred
            logits = torch.zeros((batch, seq_len, vocab_size), dtype=torch.float)
            if seq_len >= 3:
                logits[:, -1, 0] = 5.0  # Encourage EOS after 2 steps
            else:
                logits[:, -1, 1] = 3.0
                logits[:, -1, 2] = 2.0
            return DummyModelOutput(logits)

    model = DummyModel()
    tokenizer = DummyTokenizer()
    cost_model = TokenCountCostModel(eos_token_id=0, zero_cost_tokens={0})
    config = DPDConfig(beam_width=2, max_steps=4, budget=3.0, eos_token_id=0)

    wrapper = DPDGenerationWrapper(
        model=model,
        tokenizer=tokenizer,
        cost_model=cost_model,
        config=config,
    )

    result = wrapper.generate("test prompt", budget=2.0)
    assert result.is_feasible is True
    assert result.best_cost <= 2.0
    assert result.best_sequence[-1] == 0  # Terminates with EOS

