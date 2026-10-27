# Mathematical Formulation: Dual-Pruned Decoding (DPD)

**Project:** Dual-Pruned Decoding (DPD)  
**Date:** October 2026  
**Status:** Phase 1 Mathematical Specification  

---

## 1. The Autoregressive Generation Lattice

Let $\mathcal{V}$ be a finite vocabulary, and let $\mathcal{V}^* = \bigcup_{t=0}^\infty \mathcal{V}^t$ be the set of finite token sequences. Let $\langle \text{eos} \rangle \in \mathcal{V}$ denote the end-of-sequence token.

### 1.1 Node Definition
A search node at step $t$ represents a prefix sequence $u = (y_1, y_2, \dots, y_t) \in \mathcal{V}^t$ conditioned on a prompt $x$.
Formally, a node in the decoding lattice is a tuple:
$$n(u) = \big(u, s_\theta(u), S(u), C(u), R(u), \sigma(u)\big)$$
where:
1. $u \in \mathcal{V}^t$: The generated token sequence prefix.
2. $s_\theta(u) \in \mathcal{S}$: The internal autoregressive model state (e.g., Transformer KV cache) that determines $p_\theta(\cdot \mid x, u)$.
3. $S(u) = \sum_{i=1}^t \log p_\theta(y_i \mid x, y_{<i}) \le 0$: The accumulated model log-probability score.
4. $C(u) = \sum_{i=1}^t c(x, y_{<i}, y_i) \ge 0$: The accumulated resource cost under an additive step-cost function $c(\cdot) \ge 0$.
5. $R(u) = B - C(u)$: The remaining resource budget, where $B \ge 0$ is the total allocated budget.
6. $\sigma(u) \in \Sigma$: State of any optional auxiliary constraint automaton (e.g., regex / finite-state tracker).

### 1.2 Transitions and Terminals
- **Edge Transition:** An edge exists from $n(u)$ to $n(u \circ [v])$ for any token $v \in \mathcal{V}$ with transition weight $(\Delta S, \Delta C) = (\log p_\theta(v \mid x, u), c(x, u, v))$.
- **Terminal Sequence:** A node $n(u)$ is terminal if $u_t = \langle \text{eos} \rangle$ or if a maximum generation horizon $T_{\max}$ is reached.
- **Valid Completion Set:** Let $\mathcal{Y}(x) \subset \mathcal{V}^*$ denote the set of all valid terminal sequences originating from prompt $x$.

### 1.3 State Equivalence and Merging
Two distinct prefixes $u_1 \neq u_2$ are **state-equivalent** (denoted $u_1 \sim u_2$) if and only if:
$$s_\theta(u_1) = s_\theta(u_2) \quad \text{and} \quad \sigma(u_1) = \sigma(u_2)$$
When $u_1 \sim u_2$, for every continuation $\tau \in \mathcal{V}^*$:
$$p_\theta(\tau \mid x, u_1) = p_\theta(\tau \mid x, u_2) \quad \text{and} \quad C(u_1 \circ \tau) - C(u_1) = C(u_2 \circ \tau) - C(u_2)$$
In standard full-attention Transformers, KV caches for different prefixes are distinct, so exact state merging only applies under restricted state representations (e.g., $n$-gram Markov approximations, exact string merges, or fixed finite-state automata). DPD explicitely accounts for both state-equivalent and state-distinct prefixes.

---

## 2. Formal Constrained Optimization Objective

Given prompt $x$, cost function $c$, and budget $B \ge 0$:
The feasible set of terminal sequences is:
$$\mathcal{F}_B(x) = \{ y \in \mathcal{Y}(x) : C(y \mid x) \le B \}$$

### 2.1 Problem Statement
The idealized decoding objective is:
$$y^\star \in \arg\max_{y \in \mathcal{F}_B(x)} S(y \mid x)$$
subject to $\mathcal{F}_B(x) \neq \emptyset$.

### 2.2 Boundary and Infeasibility Conventions
1. **Empty Feasible Set ($\mathcal{F}_B(x) = \emptyset$):** If no sequence satisfies $C(y) \le B$ before termination, the problem is declared **infeasible**. DPD reports an infeasibility certificate and optionally falls back to the minimum-violation sequence:
   $$y_{\text{fallback}} = \arg\min_{y \in \mathcal{Y}(x)} \max(0, C(y) - B)$$
   flagged explicitly as an infeasible solution.
2. **Zero Budget ($B = 0$):** If $B = 0$, only paths with $C(y) = 0$ are feasible. If the first step costs $c > 0$, no continuation is feasible unless an immediate $\langle \text{eos} \rangle$ with zero cost is emitted.
3. **Budget Exhaustion Before EOS:** If $C(u) + c_{\min}(u) > B$ where $c_{\min}(u) = \min_{\tau: u \circ \tau \in \mathcal{Y}(x)} C(\tau)$ is the minimum cost to reach EOS, prefix $u$ cannot yield a feasible completion and must be pruned.

---

## 3. Lagrangian Relaxation and Dual Bounds

For any multiplier $\lambda \ge 0$, we define the Lagrangian score of sequence $y \in \mathcal{Y}(x)$:
$$\mathcal{L}(y, \lambda) = S(y \mid x) - \lambda (C(y \mid x) - B) = S(y \mid x) - \lambda C(y \mid x) + \lambda B$$

### 3.1 The Dual Function
The Lagrangian dual function $d: \mathbb{R}_+ \to \mathbb{R}$ is defined as:
$$d(\lambda) = \max_{y \in \mathcal{Y}(x)} \mathcal{L}(y, \lambda) = \lambda B + \max_{y \in \mathcal{Y}(x)} [ S(y \mid x) - \lambda C(y \mid x) ]$$

### 3.2 Weak Duality
**Proposition 1 (Weak Duality):** For any feasible sequence $y \in \mathcal{F}_B(x)$ and any $\lambda \ge 0$:
$$S(y \mid x) \le \mathcal{L}(y, \lambda) \le d(\lambda)$$
*Proof:* Since $y \in \mathcal{F}_B(x)$, $C(y \mid x) \le B \implies B - C(y \mid x) \ge 0$. Because $\lambda \ge 0$, $\lambda (B - C(y \mid x)) \ge 0$. Thus:
$$S(y \mid x) \le S(y \mid x) + \lambda (B - C(y \mid x)) = \mathcal{L}(y, \lambda) \le \max_{y' \in \mathcal{Y}(x)} \mathcal{L}(y', \lambda) = d(\lambda). \quad \blacksquare$$

### 3.3 The Optimal Dual Upper Bound
The tightest Lagrangian upper bound is given by:
$$d^\star = \min_{\lambda \ge 0} d(\lambda)$$
Therefore, the optimal constrained score satisfies:
$$S(y^\star) \le d^\star$$
If $y_{\text{inc}} \in \mathcal{F}_B(x)$ is any feasible sequence found during search, the duality gap:
$$\Delta = d^\star - S(y_{\text{inc}}) \ge 0$$
serves as a **certified suboptimality bound**: the true optimal score $S(y^\star)$ is at most $S(y_{\text{inc}}) + \Delta$.
When $\Delta = 0$, $y_{\text{inc}}$ is certified to be globally optimal.

---

## 4. Geometry of the Upper-Left Convex Hull

Let $\mathcal{C} = \{ (C_i, S_i) \}_{i=1}^M$ be a collection of candidates in the cost-score plane $\mathbb{R}^2$.
For a fixed resource price $\lambda \ge 0$, the scalarized objective is:
$$f_\lambda(C, S) = S - \lambda C$$

### 4.1 Geometric Interpretation
In the $(C, S)$ coordinate plane:
The level sets $S - \lambda C = k$ are lines with slope $\lambda \ge 0$ and vertical intercept $k$.
Maximizing $S - \lambda C$ corresponds to finding the line of slope $\lambda$ that supports the point set $\mathcal{C}$ from above, maximizing the vertical intercept.

### 4.2 Definition of the Upper-Left Convex Hull
A point $(C_j, S_j) \in \mathcal{C}$ is on the **upper-left convex hull** $\mathcal{H}(\mathcal{C})$ if and only if there exists some $\lambda \ge 0$ such that:
$$S_j - \lambda C_j = \max_{i=1,\dots,M} (S_i - \lambda C_i)$$
Geometrically, this corresponds to the sequence of extreme points sorted by increasing cost:
$$C_{(1)} < C_{(2)} < \dots < C_{(H)}$$
with strictly decreasing slopes:
$$\frac{S_{(2)} - S_{(1)}}{C_{(2)} - C_{(1)}} > \frac{S_{(3)} - S_{(2)}}{C_{(3)} - C_{(2)}} > \dots > \frac{S_{(H)} - S_{(H-1)}}{C_{(H)} - C_{(H-1)}} \ge 0$$

### 4.3 Candidate Invariance
**Theorem (Hull Invariance):**  
Let $\text{Ext}(\mathcal{C})$ be the vertices of the upper-left convex hull of $\mathcal{C}$. Then:
$$\forall \lambda \ge 0, \quad \max_{(C, S) \in \mathcal{C}} (S - \lambda C) = \max_{(C, S) \in \text{Ext}(\mathcal{C})} (S - \lambda C)$$
Thus, filtering candidates to the extreme vertices of the upper-left convex hull preserves the exact maximizer for **every possible linear resource price** $\lambda \in [0, \infty)$.

---

## 5. Safe Dominance Pruning with Admissible Lookahead

To safely prune prefixes during step-by-step search, DPD integrates three exact pruning rules:

### 5.1 Admissible Completion Bounds
For a prefix $u$:
- Let $U(u)$ be an admissible upper bound on the additional score achievable by any feasible completion $\tau$:
  $$S^*(u) \le S(u) + U(u)$$
  Since log-probabilities are nonpositive ($\log p \le 0$), the trivial bound $U_{\text{triv}}(u) = 0$ is universally admissible. Tighter horizon-bounded and draft-model bounds will be formalized in Phase 2.
- Let $c_{\min}(u)$ be an admissible lower bound on the cost of completing $u$ to EOS.

### 5.2 The Three Exact Pruning Rules
1. **Rule 1 (Budget Infeasibility):**
   $$\text{Prune } u \quad \text{if} \quad C(u) + c_{\min}(u) > B$$
2. **Rule 2 (Incumbent Bound Domination):**
   Let $S_{\text{inc}}$ be the score of a known feasible sequence.
   $$\text{Prune } u \quad \text{if} \quad S(u) + U(u) \le S_{\text{inc}}$$
3. **Rule 3 (State-Matched Pareto Domination):**
   If $u_1 \sim u_2$ (state-equivalent), then:
   $$\text{Prune } u_2 \quad \text{if} \quad C(u_1) \le C(u_2) \quad \text{and} \quad S(u_1) \ge S(u_2)$$
   (with at least one strict inequality or canonical tie-breaking).

These three rules are mathematically exact and guarantee zero loss of the constrained optimum.
