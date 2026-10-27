# Literature Review: Constrained Autoregressive Decoding & Dual Optimization

**Project:** Dual-Pruned Decoding (DPD): Lagrangian Hull Pruning for Budget-Constrained Language Model Generation  
**Date:** October 2026  
**Status:** Completed Phase 0 Audit  

---

## 1. Executive Summary

This literature review investigates whether Lagrangian duality, convex-hull geometry, adaptive resource pricing, and admissible completion bounds can provide an advantageous decoding algorithm for budget-constrained autoregressive sequence generation.

We conduct a cross-disciplinary survey spanning:
1. Combinatorial optimization and Resource-Constrained Shortest Path (RCSP) algorithms.
2. Lagrangian relaxation and dual decomposition in natural language processing.
3. Lexically and structurally constrained neural decoding (Grid Beam Search, NeuroLogic, FSM logit masking).
4. Heuristic and admissible search in sequence generation (A* search, Best-First Beam Search).
5. Multi-objective and Pareto decoding.

---

## 2. Theoretical Foundations: Resource-Constrained Shortest Path (RCSP)

The budget-constrained autoregressive decoding problem can be viewed as an instance of the Resource-Constrained Shortest Path Problem (RCSP) on a directed tree/DAG (decoding lattice):
$$\max_{y \in \mathcal{Y}(x)} S(y \mid x) \quad \text{s.t.} \quad C(y \mid x) \le B$$
where $S(y \mid x) = \sum_{t=1}^T \log p_\theta(y_t \mid x, y_{<t}) \le 0$ and $C(y \mid x) = \sum_{t=1}^T c(x, y_{<t}, y_t) \ge 0$.

### 2.1 Classical Dual Algorithms: LARAC and Handler-Zang
- **Handler & Zang (1980)** and **Jüttner et al. (2001) [LARAC]** established the Lagrangian relaxation framework for RCSP:
  $$L(\lambda) = \max_{y \in \mathcal{Y}} [S(y) - \lambda (C(y) - B)], \quad \lambda \ge 0$$
  By weak duality, for any $\lambda \ge 0$, $L(\lambda)$ provides an upper bound on the constrained maximum $S(y^\star)$.
- **Secant / Bisection Search:** LARAC finds the optimal multiplier $\lambda^\star = \arg\min_{\lambda \ge 0} L(\lambda)$ by alternating shortest-path computations with updated multipliers.
- **The Duality Gap:** Because RCSP is NP-hard (weakly NP-hard on DAGs), the Lagrangian dual optimal value $\min_{\lambda \ge 0} L(\lambda)$ does not in general equal $S(y^\star)$. The duality gap arises precisely from non-convexity of the discrete Pareto frontier (unsupported Pareto-optimal paths that lie inside the convex hull).
- **Implication for DPD:** Solving the Lagrangian relaxation alone can identify extreme points on the convex hull of the score-cost space, but cannot isolate unsupported points without branch-and-bound or state augmentation.

### 2.2 Pruning Mechanics: The Pulse Algorithm
- **Lozano & Medaglia (2013)** introduced the *Pulse Algorithm* for RCSP, integrating three pruning mechanisms during depth-first search:
  1. *Pruning by Infeasibility:* Discard path $u$ if $C(u) + c_{\min}(u) > B$, where $c_{\min}(u)$ is a lower bound on future cost to the target.
  2. *Pruning by Bound:* Discard path $u$ if $S(u) + U(u) \le S_{\text{inc}}$, where $U(u)$ is an upper bound on remaining score and $S_{\text{inc}}$ is a feasible incumbent.
  3. *Pruning by Dominance:* If two paths reach the same state $v$, path $u_1$ dominates $u_2$ if $S(u_1) \ge S(u_2)$ and $C(u_1) \le C(u_2)$.
- **Implication for DPD:** These three pruning rules are exact and sound. Any proposed "hull pruning" must be strictly aligned with these rules to prevent discarding optimal completions.

---

## 3. Lagrangian Relaxation & Dual Decomposition in NLP

In structured prediction, Lagrangian relaxation was popularized by **Rush & Collins (2012)** and **Chang & Collins (2011)**:
- **Phrase-Based Translation:** Chang & Collins (2011) applied Lagrangian relaxation to ensure each source word is translated exactly once. They decomposed the decoding problem into a dynamic program plus subgradient updates on multipliers enforcing word coverage.
- **Key Difference with DPD:** Prior NLP work used Lagrangian relaxation to decompose *global structural constraints* across independent DP subproblems on fixed lattices. In modern autoregressive LM decoding, the lattice is generated dynamically (exponential in horizon), and state representations (transformer KV caches) prevent exact state recombination unless state-merging or n-gram equivalence is defined.

---

## 4. Constrained Neural Decoding Algorithms

| Paradigm | Exemplars | Constraint Type | Mechanism | Guarantees |
|---|---|---|---|---|
| **Grid Search** | Hokamp & Liu (2017) | Lexical (phrases) | Expands beam in 2D grid: length $\times$ constraint steps | Heuristic; explodes beam size |
| **Tiered Search** | Anderson et al. (2017) | Lexical (phrases) | Partitions beam into finite-state constraint tiers | Heuristic; fixed beam per tier |
| **Lookahead Search** | Lu et al. (2021, 2022) [NeuroLogic, NeuroLogic A*] | CNF Predicate Logic | Prunes by unsatisfied clauses + $k$-step lookahead heuristic | Empirical heuristic; no certified suboptimality bound |
| **Grammar Masking** | Willard & Louf (2023) [Outlines], SynCode | Regular / CFG | Zeroes invalid logits via trie / Earley parsing | Exact satisfaction for formal languages; non-additive costs |
| **Penalty Search** | Length penalty (Wu et al. 2016) | Length / linear cost | Static additive/multiplicative penalty in score | Heuristic; does not guarantee hard bound satisfaction |

### 4.1 Detailed Analysis of NeuroLogic & NeuroLogic A*
- **Lu et al. (2021) [NeuroLogic]:** Formulates constraints as CNF formulas $\bigwedge_i \bigvee_j \ell_{i,j}$. Scores candidates by model likelihood plus a penalty proportional to unsatisfied clauses.
- **Lu et al. (2022) [NeuroLogic A*]:** Adds an A*-style heuristic $h(u)$ estimating future cost to satisfy remaining clauses via a $k$-step lookahead draft model or token matching.
- **Limitations:** NeuroLogic focuses on satisfiability of discrete boolean clauses (presence/absence of tokens). It does not handle general numeric resource budgets (e.g., continuous cost, monetary inference budget, risk budgets) and offers no dual certificates or bounding gaps.

---

## 5. Multi-Objective Decoding and Convex Hull Pruning

- **Martins (1984)** and **Hansen (1980):** Multi-objective shortest path algorithms maintain Pareto-optimal label sets $(c, s)$.
- **Scalarization & The Convex Hull:** A linear scalarization $S - \lambda C$ with $\lambda \ge 0$ can only discover points lying on the *upper convex hull* (in cost-score coordinates). Points on the Pareto frontier that lie in the interior of the convex hull are called *unsupported non-dominated points*.
- **The Core Hull Fallacy:** In step-by-step prefix search, candidate prefixes are NOT complete solutions; they are intermediate states with unknown future extensions.
  - If candidate $C$ is off the static prefix hull at step $t$, it may nevertheless have a low-cost, high-score completion that dominates all completions from hull vertices.
  - Therefore, naive static hull pruning on prefixes is **theoretically inadmissible** unless accompanied by admissible future bounds.

---

## 6. Synthesis: The Open Niche for DPD

The literature reveals a distinct, unoccupied research space:
1. **Existing constrained decoders** either target discrete grammars/lexical CNF (Outlines, NeuroLogic) or use static heuristics (length penalty).
2. **Existing dual relaxation decoders** (Chang & Collins, Rush & Collins) operated on fixed statistical translation lattices, not dynamic transformer token generation.
3. **No existing method** combines:
   - Dynamic adaptive Lagrangian pricing ($\lambda_t$) for scalar resource budgets.
   - Convex-hull geometric filtering of candidate expansions across the price spectrum $\lambda \in [0, \infty)$.
   - Admissible completion bounds providing certified suboptimality gaps relative to an incumbent.

This confirms the research premise is novel and worth investigating under a rigorous mathematical formulation.
