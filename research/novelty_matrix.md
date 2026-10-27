# Novelty Matrix: Comparison of DPD with Prior Decoding Methods

**Project:** Dual-Pruned Decoding (DPD)  
**Date:** October 2026  
**Status:** Completed Phase 0 Audit  

---

## 1. Dimensional Comparison Matrix

| Method | Venue & Year | Core Formulation | Pruning Mechanism | Guarantee Nature | Computational Complexity | Supported Constraints | Overlap with DPD | Distinct DPD Contribution |
|---|---|---|---|---|---|---|---|---|
| **Standard Beam Search** | Wiseman & Rush (2016) | $\max_y \sum \log p(y_t)$ | Top-$K$ by prefix log-prob | Heuristic (no optimality, no constraints) | $O(T \cdot K \cdot V)$ | None | Uses beam expansion | Budget-aware pruning & dual price feedback |
| **Length Penalty Beam Search** | Wu et al. (2016) | $\max_y \frac{S(y)}{(5+\|y\|)^\alpha / 6^\alpha}$ | Top-$K$ by normalized prefix score | Empirical heuristic | $O(T \cdot K \cdot V)$ | Soft length bias | Static cost penalty | Hard budget satisfaction, adaptive multiplier, certificates |
| **Grid Beam Search** | Hokamp & Liu (2017) | Lexical token insertion grid | Beam partitioned by number of satisfied constraints | Exhaustive along grid steps | $O(T \cdot K \cdot C \cdot V)$ where $C$ is constraints | Lexical phrases | Constraint tracking | Continuous/additive resource budgets, hull pruning |
| **Constrained Beam Search** | Anderson et al. (2017) | Finite-state automaton tracking | Fixed beam allocation per FSA state tier | Tiered heuristic | $O(T \cdot K \cdot \|Q\| \cdot V)$ | Regular expressions / lexical sets | Multi-state beam | Lagrangian price adaptation, certified bounds |
| **NeuroLogic Decoding** | Lu et al. (NAACL 2021) | CNF satisfiability penalty | Lexical constraint grouping & penalty ranking | Empirical heuristic | $O(T \cdot K \cdot V)$ | CNF boolean clauses | Penalty-based decoding | Quantitative budgets, convex hull geometry, dual certificate |
| **NeuroLogic A*esque** | Lu et al. (NAACL 2022) | Lookahead heuristic $f = g + h$ | Prunes states with poor clause fulfillment lookahead | Empirical heuristic (lookahead not certified) | $O(T \cdot K \cdot (\text{lookahead}) \cdot V)$ | CNF boolean clauses | Lookahead scoring | Formal upper bound admissibility, Lagrangian dual upper bounds |
| **Best-First Beam Search** | Meister et al. (TACL 2020) | Priority queue over search tree | Expands highest priority node in tree | Exact on explored tree subset | $O(T \cdot K \log K)$ | None (unconstrained) | Tree search priority | Resource constraints, dual relaxation, Pareto hull |
| **LARAC** | Jüttner et al. (INFOCOM 2001) | Lagrangian dual for RCSP | Dijkstra on combined weights $w = c + \lambda d$ | Weak duality upper bound; polynomial iterations | $O(I_{\text{LARAC}} \cdot (E + V \log V))$ | Single linear budget | Lagrangian relaxation formulation | Dynamic autoregressive tree generation, step-level beam pruning |
| **Pulse Algorithm** | Lozano & Medaglia (2013) | Depth-first search for RCSP | Infeasibility, bound, and dominance pruning | Exact global optimum on static graph | Exponential worst-case, fast empirical | Linear resource bounds | Infeasibility & bound pruning rules | Autoregressive generation, beam-restricted lattice, adaptive price |
| **Dual Decomposition in NLP** | Rush & Collins (2012) | Subgradient dual decomposition | Coordinate descent across subproblem decoders | Certificate of exactness when gap $= 0$ | $O(I \cdot \text{Cost}(\text{DP}))$ | Agreement constraints | Dual bounding | Autoregressive prefix pruning, convex-hull geometry |
| **Grammar Masking (Outlines)** | Willard & Louf (2023) | Deterministic FSA logit masking | Zero invalid tokens via index lookup | Hard satisfaction of formal grammar | $O(T \cdot K)$ + precomputation | Regular expressions, Context-Free Grammars | Hard constraint satisfaction | Additive resource budgets (monetary, latency, risk, tokens) |
| **Dual-Pruned Decoding (DPD)** | **This Work** | **Budget-constrained sequence generation via Lagrangian hull + admissible bounds** | **Upper-left convex hull filtering + admissible lookahead + state dominance** | **Certified suboptimality gap on finite search graph; exactness under matched state dominance** | **$O(T \cdot K \log K)$ hull overhead + inference** | **Additive resource budgets (tokens, cost, risk, latency)** | **—** | **Unified Lagrangian price adaptation, 2D hull candidate preservation, certified dual bound** |

---

## 2. Critical Analysis of Claimed Novelty

### 2.1 What is genuinely novel in DPD?
1. **Convex-Hull Geometric Pruning across Multiplier Spectra:**  
   Prior constrained beam decoders either use a fixed heuristic penalty $\lambda$ or partition the beam manually into arbitrary discrete buckets. DPD observes that for ANY nonnegative price $\lambda \in [0, \infty)$, the optimal candidate on the prefix score-cost frontier must be an extreme point of the upper-left convex hull. By retaining hull vertices, DPD maintains an optimal representation across all potential resource prices simultaneously.
2. **Dynamic Primal-Dual Multiplier Adaptation for LM Decoding:**  
   Applying a projected subgradient / integral controller to update the resource price $\lambda_t$ dynamically based on prefix consumption rate relative to the allocated budget horizon.
3. **Certified Dual Suboptimality Gap:**  
   Providing a formal certificate $U - S_{\text{inc}} \le \varepsilon$ for the output sequence relative to the explored search lattice, bridging combinatorial optimization certificates with neural text generation.

### 2.2 What is NOT novel (and must not be claimed as novel)?
1. Lagrangian relaxation itself (Handler & Zang 1980, Rush & Collins 2012).
2. Convex hulls in 2D (classical computational geometry; Graham 1972, Andrew 1979).
3. Lookahead heuristics in decoding (NeuroLogic A*, Lu et al. 2022).
4. Finite-state constraint tracking (Hokamp & Liu 2017, Anderson et al. 2017).
5. State dominance in shortest path (Martins 1984, Lozano & Medaglia 2013).

### 2.3 Explicit Novelty Gate Verdict
**PROCEED.** The integration of Lagrangian duality with convex-hull candidate filtering and certified dual bounds fills an unaddressed theoretical and empirical gap in budget-constrained generation. The novelty is distinct, mathematically substantive, and defensible provided claims are rigorously scoped.
