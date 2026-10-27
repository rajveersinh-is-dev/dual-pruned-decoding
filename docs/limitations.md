# Limitations and Failure Analysis

This document provides a candid evaluation of the theoretical and engineering limitations of **Dual-Pruned Decoding (DPD)**.

---

## 1. Non-Zero Discrete Duality Gaps

In integer and combinatorial optimization, linear Lagrangian relaxation provides an upper bound that may not be tight when the underlying Pareto frontier is non-convex.

### Unsupported Pareto Points
Consider an instance where the true constrained optimum is an "unsupported Pareto point"—a point that lies strictly below the convex upper envelope of the discrete candidate set.
In such settings:
- No linear price $\lambda \ge 0$ can make this unsupported point the unique maximizer of $S - \lambda C$.
- The dual bound $d^\star = \min_{\lambda \ge 0} d(\lambda)$ strictly exceeds the true constrained optimum $S(y^\star)$, resulting in an irreducible positive duality gap $\Delta^\star > 0$.

In our synthetic non-convex lattice benchmark, an exact zero duality gap was achieved on **80.0%** of instances, while 20.0% exhibited a non-zero duality gap ($\Delta > 0$). In those instances, DPD's certificate remains mathematically valid as an upper bound, but it provides a conservative estimate rather than exact certificate of optimality.

---

## 2. State Merging in Full-Attention Transformers

In classical dynamic programming (e.g., Hidden Markov Models or finite-state automata), two different paths $u_1, u_2$ arriving at the same state $\sigma$ can be merged exactly: the path with lower cost and higher score strictly dominates the other.

However, in modern causal Transformers with full self-attention:
- The hidden state and key-value (KV) cache depend on the *entire* prefix history $y_{<t}$.
- Two distinct prefixes $u_1 \ne u_2$ ending in the same token $y_t$ have different KV representations.
- Therefore, state dominance pruning (`enable_dominance_pruning`) can only be applied when:
  1. Identical token prefixes are compared.
  2. The cost model operates on an external Markov automaton (e.g. lexical CNF state).
  3. An explicit $n$-gram Markov assumption is enforced.

Without state merging, search graph nodes cannot be collapsed, requiring search over a tree rather than a general DAG.

---

## 3. Geometric Overhead on Micro-Instances

Computing the 2D upper-left convex hull via Andrew's Monotone Chain algorithm requires sorting $M$ candidate pairs and scanning them in $O(M \log M)$ time.

- **In Neural Generation:** A standard PyTorch Transformer forward pass takes $10 - 50\,\text{ms}$. Hull computation over 200 candidates takes approximately $30 - 50\,\mu\text{s}$ ($< 0.1\%$ of total time), making the overhead completely negligible compared to GPU inference.
- **In Micro-Instances:** If evaluating transitions takes less than $1\,\mu\text{s}$ (e.g., in a simple table lookup or tiny synthetic graph), the $O(M \log M)$ hull construction can be slower than raw unconstrained expansion.

---

## 4. Sensitivity to Step-Size and Hyperparameters

While the proportional-integral (PI) and projected subgradient rules adaptively price resource consumption, poorly chosen initial multipliers ($\lambda_0$) or excessive learning rates ($\eta$) can cause oscillatory behavior:
- If $\eta$ is too large, $\lambda_t$ spikes rapidly after a single high-cost token, causing the search to prematurely favor degenerate short sequences.
- If $\eta$ is too small, $\lambda_t$ fails to penalize resource consumption before the budget is exhausted.
- We recommend using our calibrated default values (`learning_rate=0.05`, `max_lambda=5.0`, `initial_lambda=0.0`).
