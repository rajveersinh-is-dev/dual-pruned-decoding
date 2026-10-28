# Algorithm Overview: Dual-Pruned Decoding (DPD)

**Dual-Pruned Decoding (DPD)** is a certified, resource-aware search algorithm for autoregressive generation under hard resource constraints (token limits, latency deadlines, financial API budgets, or risk quotas).

---

## 1. Problem Formulation

Let $x$ be an input prompt, and let $y = (y_1, \dots, y_T) \in \mathcal{V}^*$ be a generated token sequence.
The model defines a sequence log-probability score:
$$S(y \mid x) = \sum_{t=1}^T \log p_\theta(y_t \mid x, y_{<t}) \le 0$$

Let $c(x, y_{<t}, y_t) \ge 0$ be a non-negative step cost function, defining the cumulative resource cost:
$$C(y \mid x) = \sum_{t=1}^T c(x, y_{<t}, y_t) \ge 0$$

Given a resource budget $B \ge 0$, the constrained optimization problem is:
$$\max_{y \in \mathcal{Y}(x)} S(y \mid x) \quad \text{subject to} \quad C(y \mid x) \le B$$

---

## 2. Step-by-Step Algorithm Walkthrough

At each decoding step $t = 1, \dots, T_{\max}$:

### Step 1: Candidate Expansion
For each active beam candidate $u$ in beam $\mathcal{B}_{t-1}$:
1. Compute next-token log-probabilities $\log p_\theta(v \mid x, u)$ for all $v \in \mathcal{V}$.
2. Compute step cost $c(x, u, v)$ and minimum cost-to-complete $c_{\min}(u \circ v)$.
3. Compute admissible score upper bound $U(u \circ v)$.
4. Track completed sequences if $v = \langle\text{eos}\rangle$. If $C(u \circ v) \le B$, update the feasible incumbent sequence $y_{\text{inc}}$ whenever $S(u \circ v) > S(y_{\text{inc}})$.

### Step 2: Feasibility & Bound Pruning
Filter the expanded pool $\mathcal{C}_t$:
1. **Infeasibility Filtering:** Discard candidates where $C(u) + c_{\min}(u) > B$.
2. **Incumbent Bound Pruning:** If a feasible incumbent $y_{\text{inc}}$ exists, discard candidates where $S(u) + U(u) \le S(y_{\text{inc}})$.
3. **State Dominance:** If two candidates $u_1, u_2$ share the same Markov model state and $C(u_1) \le C(u_2)$ and $S(u_1) \ge S(u_2)$, discard $u_2$.

### Step 3: Adaptive Resource Price Regulation
Update the Lagrangian multiplier $\lambda_t$:
$$\lambda_{t+1} = \Pi_{[0, \lambda_{\max}]} \left( \lambda_t + \eta_t \cdot g_t \right)$$
where $g_t$ is the subgradient error signal comparing average beam cost $\bar{C}_t$ against the proportional pace target $B \cdot \frac{t}{T_{\max}}$.

### Step 4: Hybrid Hull-and-Bound Selection
To construct the new beam $\mathcal{B}_t$ of size $K$:
1. Project all surviving candidates to $(C_i, S_i)$ in $\mathbb{R}^2$.
2. Compute the upper-left convex hull $\mathcal{H}_{\text{UL}}$ using Andrew's Monotone Chain algorithm.
3. Include all extreme vertices of $\mathcal{H}_{\text{UL}}$ (up to $K$). These represent the Pareto-optimal candidates under *all* linear resource prices $\lambda \ge 0$.
4. Fill any remaining beam slots with top non-hull candidates ranked by their scalarized Lagrangian score $S_i - \lambda_t C_i$.

### Step 5: Dual Certificate Calculation
At termination, evaluate the Lagrangian dual function across explored multipliers $\Lambda$:
$$d(\lambda) = \lambda B + \max_{y \in \mathcal{Y}_{\text{eval}}} [S(y) - \lambda C(y)]$$
$$d^\star = \min_{\lambda \in \Lambda} d(\lambda)$$
Report the certified suboptimality bound:
$$\Delta = d^\star - S(y_{\text{inc}}) \ge 0$$

#### Certificate Scopes:
- **`GLOBAL_EXHAUSTIVE` / `BRANCH_AND_BOUND`:** If beam width was never binding and all pruned branches were safely discarded via admissible bounds ($U(u)$ admissible) or proved infeasible, $\Delta$ certifies global suboptimality over the entire search space. Global optimality (`certified_optimal = True`) is reported if and only if $\Delta \le \epsilon$.
- **`RESTRICTED_SUBSET`:** If beam width truncated branches or heuristic hull pruning dropped non-hull candidates, the dual bound holds strictly relative to the evaluated subset $\mathcal{Y}_{\text{eval}}$. Global optimality is explicitly refused (`certified_optimal = False`).

---

## 3. Complexity Analysis

| Component | Time Complexity (per step) | Space Complexity |
|---|---|---|
| Model Forward Pass | $O(K \cdot |\mathcal{V}|)$ | $O(K \cdot d_{\text{model}})$ |
| Filtering & Bounds | $O(K \cdot |\mathcal{V}|)$ | $O(K \cdot |\mathcal{V}|)$ |
| 2D Convex Hull | $O(M \log M)$, where $M \le K \cdot |\mathcal{V}|$ | $O(M)$ |
| Multiplier Update | $O(K)$ | $O(1)$ |
| **Total per Step** | **$O(K \cdot |\mathcal{V}| + M \log M)$** | **$O(K \cdot |\mathcal{V}|)$** |

Since $M \le K \cdot |\mathcal{V}|$ is typically on the order of 50 to 500 candidates after top-$k$ pre-selection, the 2D Monotone Chain hull computation takes under $50\,\mu\text{s}$, which is less than $0.1\%$ of the GPU forward-pass latency ($20-50\,\text{ms}$).

---

## 4. Native Rust Acceleration Kernel (`dpd_kernel`)

To eliminate Python interpreter overhead in performance-critical deployments, DPD provides a high-performance native Rust implementation in `dpd_kernel/`.
- **Algorithm:** Andrew's Monotone Chain 2D convex hull algorithm in $O(M \log M)$ with pre-hull Pareto dominance filtering.
- **Safety & FFI:** Zero-copy C foreign function interface (`pub unsafe extern "C" fn rust_compute_upper_left_hull`) accessed via Python's `ctypes`.
- **Verification:** Differential property-based fuzzing (`tests/property/test_rust_kernel_equivalence.py`) runs 23 exhaustive randomized, collinear, and degenerate tests verifying 100% equivalence with the pure Python reference implementation.
- **Clippy:** Strictly validated with `cargo clippy --all-targets -- -D warnings` (0 warnings, 0 errors).

---

## 5. Formal Mechanization in Lean 4

All foundational theorems of DPD are mechanized and proven in Lean 4 (`formal/DPDTheory.lean`):
- `weak_duality`: Pointwise weak Lagrangian duality.
- `dual_suboptimality_bound`: Bounded suboptimality certificate $\Delta = d^\star - S(y_{\text{inc}}) \ge 0$.
- `certified_optimal_when_zero_gap`: Exact global optimality certificate when $\Delta = 0$.
- `safe_infeasibility_pruning`: Certified pruning when $C(u) + c_{\min}(u) > B$.
- `safe_incumbent_bound_pruning`: Certified pruning when $S(u) + U(u) \le S_{\text{inc}}$.

Every proof compiles cleanly with 0 errors, 0 warnings, 0 `sorry`, and 0 `axiom` declarations.
