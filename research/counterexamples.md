# Counterexamples & Failure Mode Audit: Dual-Pruned Decoding (DPD)

**Project:** Dual-Pruned Decoding (DPD)  
**Date:** October 2026  
**Status:** Completed Phase 2 Counterexample Audit  

---

## 1. Counterexample 1: Static Prefix Hull Pruning Discards the Global Optimum

This counterexample proves that **pruning a candidate prefix solely because it lies below the upper-left convex hull at step $t$ is NOT exact** for the constrained sequence generation problem when future costs vary across branches.

### Construction
Consider a decoding search with budget $B = 3$.  
At step $t = 1$, three candidate prefixes exist: $A, B, C$.  
Their accumulated costs and log-probabilities at step 1 are:
- Candidate $A$: $(C_A, S_A) = (2, -1.0)$
- Candidate $B$: $(C_B, S_B) = (0, -2.0)$
- Candidate $C$: $(C_C, S_C) = (1, -1.8)$

### Analysis of the Static Convex Hull at Step 1
Let us test whether $C$ is on the upper-left convex hull of $\{A, B, C\}$.  
The line connecting $B(0, -2.0)$ and $A(2, -1.0)$ has slope:
$$m = \frac{S_A - S_B}{C_A - C_B} = \frac{-1.0 - (-2.0)}{2 - 0} = \frac{1.0}{2} = 0.5$$
At $C = 1$, the linear combination between $B$ and $A$ has score:
$$S_{\text{hull}}(1) = -2.0 + 0.5 \times 1 = -1.5$$
Since $S_C = -1.8 < -1.5$, point $C$ lies strictly below the segment connecting $B$ and $A$.

Now test the scalarized objective $f_\lambda(C, S) = S - \lambda C$ for all $\lambda \ge 0$:
- If $\lambda < 0.5$: $S_A - \lambda C_A = -1.0 - 2\lambda$. At $\lambda = 0.2$:
  $f_{0.2}(A) = -1.4$, while $f_{0.2}(C) = -1.8 - 0.2(1) = -2.0 < -1.4$. ($A$ wins).
- If $\lambda > 0.5$: $S_B - \lambda C_B = -2.0 - 0 = -2.0$. At $\lambda = 1.0$:
  $f_{1.0}(B) = -2.0$, while $f_{1.0}(C) = -1.8 - 1.0(1) = -2.8 < -2.0$. ($B$ wins).
- If $\lambda = 0.5$:
  $f_{0.5}(A) = -1.0 - 0.5(2) = -2.0$.
  $f_{0.5}(B) = -2.0 - 0.5(0) = -2.0$.
  $f_{0.5}(C) = -1.8 - 0.5(1) = -2.3 < -2.0$. ($A$ and $B$ tie; $C$ loses).

Therefore, candidate $C$ is strictly dominated under every possible non-negative linear price $\lambda \ge 0$.  
Under a naive hull-pruning rule, **candidate $C$ would be pruned at step 1.**

### Future Completions to EOS at Step 2
Suppose the remaining transitions to reach EOS are deterministic for each branch:
- **From $A$ (current cost 2):** Only continuation costs $\Delta C = 2$ with score $\Delta S = 0$.
  - Terminal sequence $y_A$: Total cost $= 2 + 2 = 4 > B = 3$. **INFEASIBLE.**
- **From $B$ (current cost 0):** Only continuation costs $\Delta C = 1$ with score $\Delta S = -10.0$.
  - Terminal sequence $y_B$: Total cost $= 0 + 1 = 1 \le 3$. Feasible! Total score $= -2.0 + (-10.0) = -12.0$.
- **From $C$ (current cost 1):** Only continuation costs $\Delta C = 2$ with score $\Delta S = -0.1$.
  - Terminal sequence $y_C$: Total cost $= 1 + 2 = 3 \le 3$. Feasible! Total score $= -1.8 + (-0.1) = -1.9$.

### Conclusion
- The true constrained optimal sequence is $y_C$, with score $-1.9$ and cost $3 \le 3$.
- If candidate $C$ was pruned at step 1 because it was off the static prefix hull, the search returns $y_B$ (score $-12.0$), suffering an optimality loss of $10.1$ log-prob units!
- **Scientific Implication:** Static prefix hull pruning cannot be claimed as exact for the original discrete constrained problem. Hull pruning must either be combined with admissible future completion bounds (Theorem 3) or evaluated honestly as a heuristic filter that trades exactness for beam diversity across the multiplier spectrum.

---

## 2. Counterexample 2: Non-Convex Pareto Frontier and Positive Duality Gap

This counterexample proves that **the Lagrangian relaxation of the discrete decoding problem has an inherent duality gap** whenever the optimal feasible solution is an *unsupported Pareto-optimal point*.

### Construction
Consider a single-step decision over three complete sequences:
- Sequence 1 ($y_1$): Cost $C_1 = 0$, Score $S_1 = -5.0$. (Feasible)
- Sequence 2 ($y_2$): Cost $C_2 = 2$, Score $S_2 = -2.0$. (Feasible)
- Sequence 3 ($y_3$): Cost $C_3 = 5$, Score $S_3 = 0.0$. (Infeasible for $B = 3$)

Let budget $B = 3$.
The feasible set is $\{y_1, y_2\}$.
The true constrained optimum is $y_2$ with score $S(y_2) = -2.0$.

### Lagrangian Dual Value
The Lagrangian dual function is:
$$d(\lambda) = \lambda B + \max_{i \in \{1, 2, 3\}} [S_i - \lambda C_i] = 3\lambda + \max\{-5, -2 - 2\lambda, 0 - 5\lambda\}$$

Let us evaluate $d(\lambda)$ for all $\lambda \ge 0$:
1. For $\lambda \in [0, 2/3]$:
   - Sequence 3 ($0 - 5\lambda$) dominates sequence 2 ($-2 - 2\lambda$).
   - The breakpoint where sequence 3 and sequence 2 tie is:
     $$-2 - 2\lambda = -5\lambda \implies 3\lambda = 2 \implies \lambda = \frac{2}{3}$$
   - On this interval, $d(\lambda) = 3\lambda + (-5\lambda) = -2\lambda$, which is strictly decreasing.
   - At $\lambda = 0$: $d(0) = 0$.
   - At $\lambda = 2/3$: $d(2/3) = 3(2/3) + \max\{-5, -10/3, -10/3\} = 2 - 10/3 = -4/3 \approx -1.333$.
2. For $\lambda \in [2/3, 1.5]$:
   - Sequence 2 ($-2 - 2\lambda$) dominates both sequence 1 and sequence 3.
   - On this interval, $d(\lambda) = 3\lambda + (-2 - 2\lambda) = \lambda - 2$, which is strictly increasing.
   - For example, at $\lambda = 1.0$: $d(1.0) = 3(1.0) + (-4) = -1.0$.
3. For $\lambda \ge 1.5$:
   - At $\lambda = 1.5$, sequence 1 ($-5$) ties sequence 2 ($-2 - 2(1.5) = -5$).
   - For $\lambda > 1.5$, sequence 1 dominates: $d(\lambda) = 3\lambda - 5$.
   - As $\lambda \to \infty$, $d(\lambda) \to \infty$.

The minimum dual upper bound occurs at the breakpoint $\lambda^\star = 2/3$ with value:
$$d^\star = \min_{\lambda \ge 0} d(\lambda) = d\left(\frac{2}{3}\right) = -\frac{4}{3} \approx -1.333$$

### The Duality Gap
The true constrained optimum is $S(y^\star) = S(y_2) = -2.0$.  
The dual upper bound is $d^\star = -4/3 \approx -1.333$.  
The duality gap is:
$$\Delta = d^\star - S(y^\star) = -\frac{4}{3} - (-2.0) = \frac{2}{3} \approx 0.667 > 0$$

### Conclusion
- The duality gap is strictly positive ($\Delta = 2/3 \approx 0.667$).
- Weak duality holds: $S(y^\star) = -2.0 \le -4/3 = d^\star$.
- But strong duality does NOT hold. This occurs because the discrete nature of token sequences creates a non-convex Pareto frontier where point $(C_2, S_2) = (2, -2.0)$ lies below the line connecting $(0, -5.0)$ and $(5, 0.0)$.
- **Scientific Implication:** DPD must report the certificate gap $\Delta$ honestly and never claim that the Lagrangian dual is tight in general.

---

## 3. Counterexample 3: Inadmissibility of Negative Per-Step Log-Probability Bounds

This counterexample proves that **configuring a negative per-step log-probability upper bound $\alpha < 0$ in completion bounds (such as `BudgetAwareBound`) is mathematically inadmissible in general autoregressive generation and leads to pruning the globally optimal completion.**

### Construction
Consider a budget $B = 3.0$ under a uniform step cost model where every emitted token incurs cost $c(y_t) = 1.0$, including EOS (token 0).
Suppose a search algorithm uses `BudgetAwareBound` with assumed step upper bound $\alpha = -0.05$.
At step $t = 1$, two candidate branches are expanded from the prompt root:
1. **Branch $v$ (Immediate Terminal Candidate):** Emits token 0 (EOS) with $\log p = -1.05$.
   - Cost: $C(v) = 1.0 \le B$.
   - Score: $S(v) = -1.05$.
   - Because $v$ is terminal and feasible, it immediately establishes an incumbent solution with $S_{\text{inc}} = -1.05$.
2. **Branch $u$ (Active Prefix to Optimal Completion):** Emits token 1 with $\log p = -1.00$.
   - Cost: $C(u) = 1.0$.
   - Score: $S(u) = -1.00$.
   - Remaining budget: $R(u) = B - C(u) = 3.0 - 1.0 = 2.0$.
   - Maximum allowed remaining tokens: $H(u) = \lfloor 2.0 / 1.0 \rfloor = 2$.

### Bound Evaluation at Step 1
Under the assumed per-step parameter $\alpha = -0.05$:
$$U_\alpha(u) = H(u) \times \alpha = 2 \times (-0.05) = -0.10$$
The algorithm computes the upper bound on any completion of $u$:
$$\bar{S}(u) = S(u) + U_\alpha(u) = -1.00 + (-0.10) = -1.10$$
Testing against the current incumbent $S_{\text{inc}} = -1.05$:
$$\bar{S}(u) = -1.10 \le -1.05 = S_{\text{inc}}$$
The condition $\bar{S}(u) \le S_{\text{inc}}$ is satisfied. **Candidate prefix $u$ is pruned at step 1!**

### True Continuation of Candidate $u$
Suppose candidate $u$ continues with high-confidence tokens from the language model:
- Step 2: Emits token 11 with cost $1.0$ and log-probability $\log p(y_2 \mid y_1) = -0.01$ (transition probability $p \approx 0.99 > e^{-0.05} \approx 0.9512$).
- Step 3: Emits token 0 (EOS) with cost $1.0$ and log-probability $\log p(y_3 \mid y_{\le 2}) = -0.01$.
The total sequence $y^* = (1, 11, 0)$ has:
- Total Cost: $C(y^*) = 1.0 + 1.0 + 1.0 = 3.0 \le B = 3.0$ (FEASIBLE).
- Total Score: $S(y^*) = -1.00 - 0.01 - 0.01 = -1.02$.

### Mathematical Violation
Observe that:
$$S(y^*) = -1.02 > -1.05 = S(v) = S_{\text{inc}}$$
Sequence $y^*$ is the unique constrained global optimum.
However, because $\alpha = -0.05 < \log p_2 = -0.01$, the bound underestimated the true future potential:
$$\bar{S}(u) = -1.10 < -1.02 = S(y^*)$$
The bound was **strictly less than the true completion score**, violating the definition of an admissible upper bound:
$$\exists \tau \in \text{Completions}(u) \quad \text{s.t.} \quad S(u \circ \tau) > S(u) + U_\alpha(u)$$

### Scientific Conclusion & Enforcement
1. **Uncertified Heuristic vs. Admissible Bound:** In general causal language models with normalized categorical outputs, $\log p_\theta(y_t \mid y_{<t}) \in (-\infty, 0]$. Since confidence can approach 1.0 arbitrarily closely ($p \to 1.0 \implies \log p \to 0.0$), the **unique universally admissible per-step upper bound is $\alpha = 0.0$** (yielding $U(u) = 0.0$).
2. **Implementation Rule:** In `dpd/bounds.py`, `max_log_prob_per_step` must strictly default to `0.0`. Any negative value must require an explicit opt-in (`allow_heuristic_tighter_bound=True`) and be marked as an uncertified heuristic search feature.

---

## 4. Geometric Degeneracies in Hull Maintenance

When implementing the 2D convex hull algorithm, the following edge cases must be handled without crashing, numerical overflow, or non-deterministic ordering:
1. **Vertical Points (Identical Cost):** Multiple candidates with identical cost $C_i = C_j$ but different scores $S_i > S_j$. The upper-left hull must discard $j$ and retain only the maximum score candidate $i$.
2. **Horizontal Points (Identical Score):** Multiple candidates with identical score $S_i = S_j$ but different costs $C_i < C_j$. The upper-left hull must discard $j$ and retain only the minimum cost candidate $i$.
3. **Coincident Points:** Multiple candidates with identical $(C, S)$. Retain exactly one canonical representative.
4. **Collinear Vertices:** Three points lying on the exact same line with slope $\lambda$. Standard Graham/Andrew algorithms must choose whether to retain collinear boundary points or only extreme vertices. For DPD, extreme vertices suffice to represent the support.

---

## 5. Counterexample 4: Top-$k$ Successor Shortlists Truncate the Feasible Optimum

This counterexample proves that **restricting the transition function `step_fn` to a top-$k$ shortlist (e.g. top-8 tokens by language model logit) invalidates global optimality guarantees, and can discard the unique feasible optimum even when beam width $W$ is arbitrarily large.**

### Construction
Consider budget $B = 3.0$ and vocabulary $\mathcal{V} = \{1, 2, 0\}$ where $0 = \text{EOS}$.
From prompt root $\epsilon$:
- Successor $1$ (rank 1 by immediate logit): $\log p_1 = -0.10$, step cost $c_1 = 5.0$.
  - Any completion of $1$ has total cost $C \ge 5.0 > B = 3.0$. **Every completion is INFEASIBLE.**
- Successor $2$ (rank 2 by immediate logit): $\log p_2 = -0.20$, step cost $c_2 = 1.0$.
  - Continuation: token $0$ (EOS) with $\log p = -0.10$, cost $c = 1.0$.
  - Complete sequence $y^* = (2, 0)$ has total cost $C(y^*) = 1.0 + 1.0 = 2.0 \le 3.0$ (FEASIBLE) and total score $S(y^*) = -0.30$.

### Impact of Top-$k$ Truncation ($k = 1$)
If `step_fn` only generates top-$k$ ($k=1$) successors:
1. Successor $1$ is generated; successor $2$ is never generated.
2. Search expands successor $1$, encounters infeasibility, and has no feasible candidates in the beam.
3. The search must either fail or fall back to an infeasible candidate, completely missing the true global optimum $y^*$.

### Scientific Conclusion & Enforcement
A search over a top-$k$ successor generator is an optimization over a restricted heuristic sub-lattice $\mathcal{L}_{\text{top-}k} \subset \mathcal{V}^*$.  
**Implementation Rule:** When `config.successor_completeness != SuccessorCompleteness.EXHAUSTIVE_FINITE_DOMAIN`, the certificate engine strictly disallows global certificates, forcing `certificate_scope = CertificateScope.RESTRICTED_SUBSET` (or `RESTRICTED_LATTICE`) and `is_global = False`.

---

## 6. Counterexample 5: Multi-Candidate Degeneracy and Tied Scores

This counterexample examines the behavior of DPD when multiple distinct candidates achieve identical costs and scores, $(C_1, S_1) = (C_2, S_2) = (C_3, S_3)$.

### Potential Failure Mode
When two points $v_1, v_2$ have identical cost $C_1 = C_2$, computing analytical slopes:
$$\lambda = \frac{S_2 - S_1}{C_2 - C_1} = \frac{0}{0}$$
produces a `ZeroDivisionError` or `NaN`. Furthermore, under subgradient multiplier updates, identical scalarized values can cause oscillation or non-deterministic ordering between steps.

### Resolution & Verification
1. Andrew's Monotone Chain 2D convex hull algorithm sorts lexicographically by $(C, -S)$ and deduplicates points with identical $(C, S)$ within numerical tolerance $\epsilon = 10^{-9}$.
2. In `dpd/certificates.py`, slope computation enforces `dc = v2.cost - v1.cost > 1e-9`, bypassing vertical degenerate segments.
3. Regression test `test_counterexample_5_tied_scores_and_multiplier_stability` verifies that the certificate calculates the exact dual upper bound without NaN or crash.

---

## 7. Counterexample 6: Empty Feasible Set / Total Budget Infeasibility

This counterexample proves that **when the resource budget $B$ is set strictly below the minimum possible sequence cost in the search space, DPD must cleanly report infeasibility rather than fabricating an invalid certificate.**

### Construction
Let budget $B = 5.0$.  
Suppose every non-trivial sequence has minimum cost $C_{\min} = 10.0 > B$.  
The feasible candidate set $\mathcal{Y}_{\text{feas}} = \{y \in \mathcal{Y} \mid C(y) \le B\} = \emptyset$.

### Protocol & Enforcement
1. `incumbent` remains `None`.
2. When the beam is exhausted, `recovery_mgr.get_best_fallback` returns the least-violating candidate.
3. The decoding result returns `is_feasible = False` and `termination_reason = "infeasible_fallback"`.
4. `compute_dual_certificate` sets:
   - `is_feasible = False`
   - `suboptimality_gap = inf`
   - `certified_optimal = False`
   - `certified_globally_optimal = False`
   - `certified_optimal_within_scope = False`
5. Verified by `test_counterexample_6_empty_feasible_set_all_candidates_infeasible`.

---

## 8. Counterexample 7: Floating-Point Boundary Shift in Bound Pruning

This counterexample tests numerical boundary behavior in `filter_by_incumbent_bound`.

### Problem Statement
Given an incumbent with score $S_{\text{inc}} = -5.0$, a candidate prefix $u$ has completion upper bound $S(u) + U(u)$.  
Under finite-precision floating-point arithmetic:
- If a candidate cannot strictly beat the incumbent, exploring it cannot discover a new strictly optimal sequence.
- If $S(u) + U(u) \le S_{\text{inc}} + \epsilon$, pruning must discard candidates that cannot strictly improve upon the incumbent.
- If $S(u) + U(u) > S_{\text{inc}} + \epsilon$, pruning must preserve the candidate.

### Verification
In `test_counterexample_7_numerical_tolerance_boundary_shift`:
- Candidate $c_1$ with score $-5.1$ is pruned.
- Candidate $c_2$ with score $-5.0$ (exact tie) cannot strictly improve upon the existing feasible incumbent and is pruned.
- Candidate $c_3$ with score $-5.0 + 10^{-8} > -5.0 + 10^{-9}$ survives.
- Candidate $c_4$ with score $-4.9$ survives.

---

## 9. Counterexample 8: Certificate Scope Escalation Prevention

This counterexample proves that **a restricted or heuristic decoding run cannot silently masquerade as a global proof of optimality.**

### Attack Scenario
A caller executes decoding on a restricted sub-lattice (e.g. `successor_completeness = RESTRICTED_TOP_K`), but manually calls `compute_dual_certificate(..., scope=CertificateScope.RESTRICTED_SUBSET, is_global=True)`.

### Defense & Verification
In `dpd/certificates.py`:
```python
if scope != CertificateScope.GLOBAL_EXHAUSTIVE and is_global:
    is_global = False
```
The certificate sanitizer intercepts the conflicting parameters, strips `is_global`, and sets:
- `is_global = False`
- `certified_globally_optimal = False`
- `certified_optimal = False`
- `certificate_scope = CertificateScope.RESTRICTED_SUBSET`
- `certified_optimal_within_scope = True` (if gap within subset $\le \text{tol}$)
Verified by `test_counterexample_8_scope_mismatch_prevents_global_certificate`.

