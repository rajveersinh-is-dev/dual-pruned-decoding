# Research Decision & Novelty Gate: Dual-Pruned Decoding (DPD)

**Project:** Dual-Pruned Decoding (DPD)  
**Date:** October 2026  
**Status:** Approved & Formally Scoped  

---

## 1. Feasibility & Originality Decision

### 1.1 Did the proposed contribution survive comparison with existing work?
**Yes, with precisely scoped mathematical boundaries.**

Our literature search confirms:
1. No existing method in NLP or machine learning uses the name "Dual-Pruned Decoding" or implements convex-hull pruning across an adaptive spectrum of Lagrangian prices for budget-constrained autoregressive generation.
2. Prior Lagrangian decoding algorithms in NLP (e.g., Rush & Collins 2012, Chang & Collins 2011) applied dual decomposition across static alignment or grammar constraints in statistical MT. They did not address dynamic autoregressive decoding under scalar resource budgets (tokens, cost, latency, risk).
3. Prior constrained beam search decoders (Hokamp & Liu 2017, Anderson et al. 2017, Lu et al. 2021/2022) focus on boolean lexical constraints (CNF satisfiability) or static length penalties. They do not maintain a Pareto hull or provide dual suboptimality certificates.

### 1.2 Mathematical Boundary & Crucial Downgrade
To maintain absolute scientific integrity, we explicitly acknowledge and formalize the **Convex Hull Pruning Trap**:
- **Fallacy:** "Because a point lies below the static prefix convex hull in $(C, S)$ space, it cannot be part of the optimal constrained complete sequence."
- **Reality:** We have constructed a formal counterexample showing that a candidate strictly dominated under all linear scalarizations $S - \lambda C$ at step $t$ can be the *unique* feasible and optimal path to EOS if future continuations have varying costs.
- **Defensible Scope:**
  1. Convex-hull pruning is **exact** for maximizing the *scalarized Lagrangian relaxation* at any price $\lambda \ge 0$.
  2. For the *discrete constrained problem*, pruning a prefix is **provably exact** if and only if:
     - Its admissible upper bound $U(u)$ satisfies $U(u) \le S_{\text{inc}}$ (Incumbent Bound Pruning), OR
     - Its minimum required future cost $c_{\min}(u)$ violates the remaining budget: $C(u) + c_{\min}(u) > B$ (Infeasibility Pruning), OR
     - It is strictly dominated in cost and score by another prefix sharing the identical forward state (State Dominance Pruning).
  3. When state representations differ, retaining the convex hull of candidate extensions acts as an **optimal multi-multiplier filter** that guarantees maximal coverage across all potential Lagrangian slopes, outperforming single-multiplier heuristic pruning.
  4. The algorithm computes a **certified dual upper bound** $d^*$ on the optimal constrained score over the search lattice, yielding a provable suboptimality certificate $\varepsilon = d^* - S_{\text{inc}}$.

---

## 2. Research Plan: 4 Core Modules

1. **Module 1: Convex Hull Engine (`dpd.hull`)**
   - Implements Andrew's Monotone Chain 2D convex hull algorithm ($O(M \log M)$).
   - Computes upper-left hull vertices corresponding to extreme points for non-negative multipliers $\lambda \in [0, \infty)$.
   - Handles all degeneracies: collinear points, identical costs, identical scores, single points.

2. **Module 2: Adaptive Price Controller (`dpd.multipliers`)**
   - Dynamically updates $\lambda_t$ using projected subgradient and proportional-integral (PI) tracking against the remaining resource budget.
   - Adjusts pressure smoothly without oscillation.

3. **Module 3: Admissible Bounds & Pruning (`dpd.bounds`, `dpd.dominance`)**
   - Admissible future completion bounds $U(u)$ and cost lower bounds $c_{\min}(u)$.
   - Pareto dominance filtering and incumbent-based branch-and-bound pruning.

4. **Module 4: Certified Dual Decoding & Verification (`dpd.decoding`, `dpd.certificates`)**
   - Complete decoding loop for discrete lattices and language models.
   - Exact certificate tracking: reports incumbent, dual upper bound $d^*$, and certificate gap $\Delta$.

---

## 3. Next Milestone
Proceed to **Phase One (Mathematical Formulation)** and **Phase Two (Theorems & Proofs)**.
