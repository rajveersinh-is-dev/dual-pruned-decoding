# Benchmarking Suite and Experimental Protocols

This document specifies the experimental protocols, benchmark tasks, and evaluation baselines used to validate **Dual-Pruned Decoding (DPD)**.

---

## 1. Experimental Integrity Standards

In accordance with scientific integrity guidelines:
- **Zero Preloaded Numbers:** All log-probabilities, costs, convex hulls, and dual bounds are computed dynamically from primary principles.
- **Hardware Performance Counters:** In-process latency measurements use high-resolution hardware counters (`time.perf_counter_ns()`).
- **Warmup Protocol:** Every evaluated method is preceded by at least 5 discarded warmup runs to eliminate OS cold-start and JIT compilation variance.
- **Seeded Evaluation:** All evaluation prompts and synthetic lattices use fixed, documented seeds.

---

## 2. Benchmark Tasks

### Task 1: Real Autoregressive Language Model (DistilGPT-2)
- **Model:** `distilgpt2` (82M parameters, Hugging Face transformers).
- **Task:** Autoregressive completion under strict token expenditure budgets.
- **Budgets Evaluated:** $B \in \{10, 15, 20\}$.
- **Prompts:** 10 diverse NLP prompts covering scientific summarization, factual completion, and dialogue:
  1. `"The fundamental law of thermodynamics states that"`
  2. `"Artificial intelligence systems require careful alignment because"`
  3. `"The primary advantage of quantum computing is"`
  4. `"In modern distributed database architecture, consensus is achieved by"`
  5. `"The history of cryptography transformed during the mid-20th century when"`
  6. `"Photosynthesis is the biological process wherein plants"`
  7. `"To optimize high-throughput streaming systems, engineers should"`
  8. `"Reinforcement learning from human feedback operates by"`
  9. `"The theory of general relativity predicts that gravitational waves"`
  10. `"Compiler optimization passes preserve program semantics while"`

### Task 2: Ground-Truth Synthetic Lattices
- **Lattices:** 15 multi-branch search lattices with non-convex Pareto frontiers and non-uniform branch step costs.
- **Ground Truth:** Exhaustive brute-force enumeration of every valid path in the search space to determine the true constrained global optimum $y^\star$.
- **Budgets Evaluated:** $B \in \{3.0, 5.0, 7.0\}$.

---

## 3. Evaluated Baselines

1. **Standard Beam Search:** Unconstrained beam search prioritizing immediate log-probabilities ($S(u)$).
2. **Length-Penalty Beam Search:** Beam search with length normalization penalty $\frac{(5 + |y|)^\alpha}{(5 + 1)^\alpha}$ with $\alpha = 0.6$.
3. **Greedy Decoding:** $K=1$ unconstrained greedy decoding.
4. **DPD without Hull (`PruningMode.EXACT_BOUND_AND_DOMINANCE`):** Certified bound and dominance pruning with adaptive multiplier, omitting 2D convex hull vertex selection.
5. **DPD with Fixed Multiplier:** Hybrid hull-and-bound pruning with static non-adaptive penalty $\lambda = 0.8$.

---

## 4. Evaluated Metrics

- **Feasible Output Rate (%):** Fraction of completed sequences satisfying $C(y) \le B$.
- **Average Feasible Score:** Mean log-probability among feasible sequences.
- **Hard-Constraint Violation Rate (%):** $100\% - \text{Feasibility Rate}$.
- **Average Cost:** Mean resource expenditure.
- **Wall-Clock Latency (ms):** Mean generation time per prompt.
- **Candidates Pruned per Sequence:** Average count of dead-end branches pruned prior to token expansion.
- **Certified Optimality Rate (%):** Fraction of instances where DPD's reported suboptimality gap $\Delta = d^\star - S(y_{\text{inc}}) \le 10^{-4}$.
- **Average Duality Gap ($\Delta$):** Mean suboptimality certificate.

---

## 5. Running the Benchmarks

```bash
# Execute benchmark suite (approx. 2-3 minutes on standard CPU)
python experiments/scripts/run_experiments.py

# Verify outputs
ls -lh benchmarks/results/benchmark_results.json
```
