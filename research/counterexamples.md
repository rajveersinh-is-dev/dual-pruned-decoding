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
1. For $\lambda \in [0, 1]$:
   - At $\lambda = 0$: $\max\{-5, -2, 0\} = 0 \implies d(0) = 0$.
   - As $\lambda$ increases from 0 to 1, sequence 3 achieves $0 - 5\lambda$ which dominates sequence 2 until $\lambda = 2/3$.
   - Specifically, for $\lambda = 0.4$:
     $S_1 - \lambda C_1 = -5.0$
     $S_2 - \lambda C_2 = -2 - 0.8 = -2.8$
     $S_3 - \lambda C_3 = 0 - 2.0 = -2.0$ (wins).
     $d(0.4) = 3(0.4) + (-2.0) = 1.2 - 2.0 = -0.8$.
2. At $\lambda = 1.0$:
   - $S_1 - \lambda C_1 = -5$
   - $S_2 - \lambda C_2 = -4$
   - $S_3 - \lambda C_3 = -5$
   - $d(1.0) = 3(1.0) + (-4) = -1.0$.
3. For $\lambda \ge 1.0$:
   - At $\lambda = 1.5$: $d(1.5) = 3(1.5) + \max\{-5, -5, -7.5\} = 4.5 - 5.0 = -0.5$.
   - As $\lambda \to \infty$, $d(\lambda) \to \infty$.

The minimum dual upper bound occurs at $\lambda^\star = 1.0$ with value:
$$d^\star = \min_{\lambda \ge 0} d(\lambda) = -1.0$$

### The Duality Gap
The true constrained optimum is $S(y^\star) = S(y_2) = -2.0$.  
The dual upper bound is $d^\star = -1.0$.  
The duality gap is:
$$\Delta = d^\star - S(y^\star) = -1.0 - (-2.0) = 1.0 > 0$$

### Conclusion
- The duality gap is strictly positive ($\Delta = 1.0$).
- Weak duality holds: $S(y^\star) = -2.0 \le -1.0 = d^\star$.
- But strong duality does NOT hold. This occurs because the discrete nature of token sequences creates a non-convex Pareto frontier where point $(C_2, S_2) = (2, -2.0)$ lies below the line connecting $(0, -5.0)$ and $(5, 0.0)$.
- **Scientific Implication:** DPD must report the certificate gap $\Delta$ honestly and never claim that the Lagrangian dual is tight in general.

---

## 3. Counterexample 3: Admissible vs Inadmissible Lookahead

Let true optimal future completion score be $S^*(\tau)$.
- **Admissible Bound:** An upper bound $U(u) \ge S^*(\tau)$ (e.g. $U(u) = 0$ since all log-probs $\le 0$). Pruning by $S(u) + U(u) \le S_{\text{inc}}$ is safe.
- **Inadmissible Heuristic:** If a draft model or heuristic predicts $H(u) = \hat{S}$, and $\hat{S} < S^*(\tau)$ (underestimation of future score), then a prefix with high future potential will be falsely discarded.
- **Empirical Tightness Dilemma:** The bound $U(u) = 0$ is globally admissible, but for long horizons $T$, $S(u) + 0 \le S_{\text{inc}}$ rarely triggers early in generation because $S_{\text{inc}}$ is negative and $S(u)$ is closer to 0. Tighter bounds that condition on remaining budget (e.g., maximum log-prob of tokens with cost $\le R(u)$) are required for effective early pruning.

---

## 4. Geometric Degeneracies in Hull Maintenance

When implementing the 2D convex hull algorithm, the following edge cases must be handled without crashing, numerical overflow, or non-deterministic ordering:
1. **Vertical Points (Identical Cost):** Multiple candidates with identical cost $C_i = C_j$ but different scores $S_i > S_j$. The upper-left hull must discard $j$ and retain only the maximum score candidate $i$.
2. **Horizontal Points (Identical Score):** Multiple candidates with identical score $S_i = S_j$ but different costs $C_i < C_j$. The upper-left hull must discard $j$ and retain only the minimum cost candidate $i$.
3. **Coincident Points:** Multiple candidates with identical $(C, S)$. Retain exactly one canonical representative.
4. **Collinear Vertices:** Three points lying on the exact same line with slope $\lambda$. Standard Graham/Andrew algorithms must choose whether to retain collinear boundary points or only extreme vertices. For DPD, extreme vertices suffice to represent the support.
