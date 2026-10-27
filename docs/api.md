# API Reference: Dual-Pruned Decoding (`dpd`)

The `dpd` library provides a modular, typed Python API for certified resource-aware decoding.

---

## 1. Primary Entry Point: `dual_pruned_decoding`

```python
from dpd import dual_pruned_decoding, DPDConfig, TokenCountCostModel

result = dual_pruned_decoding(
    model=model,
    tokenizer=tokenizer,
    prompt="Summarize the core mechanism of Transformers:",
    budget=15.0,
    cost_model=TokenCountCostModel(),
    config=DPDConfig(beam_width=4, max_steps=30),
)

print("Generated text:", result.text)
print("Log-prob score:", result.score)
print("Resource cost:", result.cost)
print("Is feasible:", result.is_feasible)
print("Suboptimality certificate (Delta):", result.certificate.suboptimality_gap)
```

### Signature
```python
def dual_pruned_decoding(
    model: Any,
    tokenizer: Any,
    prompt: str | list[int],
    budget: float,
    cost_model: BaseCostModel,
    config: DPDConfig | None = None,
    upper_bound_model: BaseUpperBoundModel | None = None,
    eos_token_id: int | None = None,
    device: str | None = None,
) -> DPDOutput:
```

### Parameters
- `model`: PyTorch causal language model (e.g. Hugging Face `AutoModelForCausalLM`) or custom graph transition model.
- `tokenizer`: Tokenizer instance with `encode` and `decode` methods.
- `prompt`: String prompt or list of token IDs.
- `budget`: Non-negative float representing the hard resource cap ($B$).
- `cost_model`: Instance of `BaseCostModel` defining step and completion costs.
- `config`: `DPDConfig` instance controlling beam width, pruning strategy, and multipliers.
- `upper_bound_model`: Optional admissible bound model (defaults to `TrivialUpperBound`).
- `eos_token_id`: Optional EOS token ID (inferred from `tokenizer.eos_token_id` if omitted).
- `device`: Target device (e.g., `"cpu"`, `"cuda"`).

### Returns
`DPDOutput` dataclass:
- `tokens`: Generated token IDs (`list[int]`).
- `text`: Decoded string (`str`).
- `score`: Sequence log-probability ($S \le 0$).
- `cost`: Total resource cost ($C \ge 0$).
- `is_feasible`: Boolean indicating whether $C \le B$.
- `certificate`: `DualCertificate` with dual bound $d^\star$ and suboptimality gap $\Delta$.
- `stats`: `DecodingStats` with pruning counts, latency, and step traces.

---

## 2. Configuration: `DPDConfig`

```python
from dpd.config import DPDConfig, PruningMode, MultiplierUpdateRule

config = DPDConfig(
    beam_width=4,
    max_steps=32,
    pruning_mode=PruningMode.HYBRID_HULL_AND_BOUND,
    multiplier_rule=MultiplierUpdateRule.PROJECTED_SUBGRADIENT,
    initial_lambda=0.0,
    learning_rate=0.05,
    max_lambda=5.0,
    enable_dominance_pruning=True,
    enable_bound_pruning=True,
    enable_recovery=True,
)
```

### Enums
- `PruningMode`:
  - `HYBRID_HULL_AND_BOUND`: (Recommended) Combines safe branch-and-bound pruning with upper-left convex hull vertex selection.
  - `EXACT_BOUND_AND_DOMINANCE`: Certified pruning without hull filtering.
  - `STATIC_HULL_ONLY`: Pure static hull filtering (for ablation studies).
  - `NONE`: No pruning (unconstrained beam search).
- `MultiplierUpdateRule`:
  - `PROJECTED_SUBGRADIENT`: First-order subgradient step projected onto $[0, \lambda_{\max}]$.
  - `PROPORTIONAL_INTEGRAL`: PI controller regulating consumption pace.
  - `CONSTANT`: Static fixed multiplier.

---

## 3. Cost Models: `BaseCostModel`

### `TokenCountCostModel`
Counts generated non-EOS tokens:
```python
from dpd.cost_models import TokenCountCostModel
cost_model = TokenCountCostModel(token_cost=1.0)
```

### `LinearTokenCostModel`
Assigns token-specific non-uniform costs (e.g., API pricing, keyword penalties):
```python
from dpd.cost_models import LinearTokenCostModel
cost_model = LinearTokenCostModel(token_costs={50256: 0.0, 1024: 2.5}, default_cost=1.0)
```

### `PrefixSensitiveCostModel`
State-dependent cost functions (e.g., non-linear latency models):
```python
from dpd.cost_models import PrefixSensitiveCostModel
```

---

## 4. Geometric Convex Hull: `dpd.hull`

```python
from dpd.hull import compute_upper_left_hull, is_on_upper_left_hull

points = [(2.0, -1.0), (0.0, -2.0), (1.0, -1.8)]
hull_indices = compute_upper_left_hull(points)
# Returns [1, 0], identifying (0.0, -2.0) and (2.0, -1.0) as extreme vertices
```

---

## 5. Dual Certification: `dpd.certificates`

```python
from dpd.certificates import compute_dual_certificate

cert = compute_dual_certificate(
    evaluated_sequences=completed_candidates,
    budget=15.0,
    evaluated_lambdas=[0.0, 0.5, 1.0, 2.0],
    incumbent_score=-7.8,
)
print("Dual bound d*:", cert.dual_bound)
print("Suboptimality gap Delta:", cert.suboptimality_gap)
print("Is certified optimal:", cert.is_certified_optimal)
```

---

## 6. Hugging Face Transformers Integration

DPD integrates seamlessly with standard Hugging Face pipelines through two interfaces:

### A. `DPDLogitsProcessor`
Drop-in logits processor compatible with `model.generate()`:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from dpd.cost_models import TokenCountCostModel
from dpd.hf_integration import DPDLogitsProcessor

model = AutoModelForCausalLM.from_pretrained("distilgpt2")
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")

logits_processor = DPDLogitsProcessor(
    budget=15.0,
    cost_model=TokenCountCostModel(),
    initial_lambda=0.5,
    learning_rate=0.1,
    eos_token_id=tokenizer.eos_token_id,
)

outputs = model.generate(
    **tokenizer("Write a summary of quantum mechanics:", return_tensors="pt"),
    max_new_tokens=30,
    logits_processor=[logits_processor],
)
```

### B. `DPDGenerationWrapper`
End-to-end wrapper providing full certificate tracking:
```python
from dpd.hf_integration import DPDGenerationWrapper

wrapper = DPDGenerationWrapper(model=model, tokenizer=tokenizer)
result = wrapper.generate(prompt="Explain general relativity:", budget=20.0)
print("Text:", result.text)
print("Certified optimal:", result.certificate.is_certified_optimal)
```

---

## 7. Native Rust Kernel (`dpd_kernel`)

The native Rust acceleration kernel can be configured via `HullConfig`:
```python
from dpd.config import DPDConfig, HullConfig

config = DPDConfig(
    beam_width=4,
    hull=HullConfig(use_rust_kernel=True),  # Automatically loads dpd_kernel.dll/.so
)
```
If the compiled native library is not found or fails to load, DPD gracefully falls back to the pure Python reference implementation with zero interruption.
