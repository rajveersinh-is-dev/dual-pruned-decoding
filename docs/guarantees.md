# Mathematical Guarantees and Proofs

This document formalizes the theoretical foundations of **Dual-Pruned Decoding (DPD)**, including duality bounds, convex hull geometry, safe dominance, and counterexamples.

---

## 1. Weak Duality and Bounded Suboptimality

### Theorem 1 (Certified Bounded Suboptimality)
Let $\mathcal{G}$ be any finite explored search lattice, and let $\mathcal{F}_B(\mathcal{G}) = \{ y \in \mathcal{Y}_\mathcal{G} : C(y) \le B \}$ be the non-empty feasible set.
For any non-negative multiplier $\lambda \ge 0$, define the Lagrangian dual function:
$$d(\lambda) = \lambda B + \max_{y \in \mathcal{Y}_\mathcal{G}} [ S(y) - \lambda C(y) ]$$

Let $\Lambda = \{\lambda^{(1)}, \dots, \lambda^{(P)}\} \subset \mathbb{R}_+$ be any finite set of evaluated multipliers, and let $d^\star_\Lambda = \min_{\lambda \in \Lambda} d(\lambda)$.
For any feasible incumbent sequence $y_{\text{inc}} \in \mathcal{F}_B(\mathcal{G})$:
1. $S(y^\star) \le d^\star_\Lambda$, where $y^\star \in \arg\max_{y \in \mathcal{F}_B(\mathcal{G})} S(y)$.
2. The additive suboptimality gap is bounded by:
   $$S(y^\star) - S(y_{\text{inc}}) \le \Delta_\Lambda \triangleq d^\star_\Lambda - S(y_{\text{inc}}) \ge 0$$
3. If $\Delta_\Lambda = 0$, $y_{\text{inc}}$ is certified to be an exact global maximizer on $\mathcal{G}$.

#### Proof
For any feasible sequence $y \in \mathcal{F}_B(\mathcal{G})$, by definition $C(y) \le B$, which implies $B - C(y) \ge 0$.
Since $\lambda \ge 0$, we have $\lambda(B - C(y)) \ge 0$.
Adding $S(y)$ yields:
$$S(y) \le S(y) + \lambda(B - C(y)) = S(y) - \lambda C(y) + \lambda B$$
Because this holds for all feasible $y$, it holds in particular for the optimal $y^\star$:
$$S(y^\star) \le S(y^\star) - \lambda C(y^\star) + \lambda B \le \max_{y' \in \mathcal{Y}_\mathcal{G}} [S(y') - \lambda C(y')] + \lambda B = d(\lambda)$$
Taking the minimum over $\lambda \in \Lambda$ establishes (1).
Subtracting $S(y_{\text{inc}})$ from both sides yields (2).
If $\Delta_\Lambda = 0$, then $S(y^\star) \le S(y_{\text{inc}})$. Since $y_{\text{inc}}$ is feasible, $S(y_{\text{inc}}) \le S(y^\star)$, which implies $S(y_{\text{inc}}) = S(y^\star)$, proving (3). $\blacksquare$

---

## 2. Convex Hull Invariance

### Theorem 2 (Convex Hull Invariance under Multiplier Price Spectrum)
Let $\mathcal{C} = \{(C_i, S_i)\}_{i=1}^M \subset \mathbb{R}^2$ be a finite set of candidates in the cost-score plane.
Let $\text{Ext}(\mathcal{H}_{\text{UL}})$ denote the extreme vertices of the upper-left convex hull of $\mathcal{C}$.
Then for every non-negative linear resource price $\lambda \in [0, \infty)$:
$$\max_{(C, S) \in \mathcal{C}} (S - \lambda C) = \max_{(C, S) \in \text{Ext}(\mathcal{H}_{\text{UL}})} (S - \lambda C)$$
Furthermore, any candidate point lying strictly in the interior below the hull achieves a strictly lower scalarized score than a hull vertex for all $\lambda \ge 0$.

#### Proof
The scalarized objective function $f_\lambda(C, S) = S - \lambda C = \langle (-\lambda, 1), (C, S) \rangle$ is a linear functional on $\mathbb{R}^2$.
By fundamental polyhedral theory, the maximum of any linear function over a compact set is attained at an extreme vertex of its convex hull.
Because $\lambda \ge 0$, the normal vector $(-\lambda, 1)$ has a non-positive first coordinate and a strictly positive second coordinate (pointing into the second quadrant, i.e., northwest).
Therefore, supporting hyperplanes with normal $(-\lambda, 1)$ can only make contact with the set along its upper-left boundary $\mathcal{H}_{\text{UL}}$.
Any point strictly below the upper-left hull is a strict convex combination of hull points or strictly dominated by them, and thus achieves a strictly lower value under all non-negative prices. $\blacksquare$

---

## 3. The Convex Hull Pruning Trap (Counterexample 1)

**Question:** Can candidates off the static upper-left convex hull be safely pruned during beam search?
**Answer:** **No.** While they are suboptimal for the *current* scalarized relaxation, they may lead to the unique feasible global optimum if future continuation costs vary across branches.

### Counterexample 1
- **Budget:** $B = 3.0$.
- **Step 1:** Three candidate prefixes are generated:
  - $A: (C=2.0, S=-1.0)$
  - $B: (C=0.0, S=-2.0)$
  - $C: (C=1.0, S=-1.8)$
- **Hull Analysis at Step 1:**
  - The line segment connecting $B(0, -2.0)$ and $A(2, -1.0)$ passes through $(1.0, -1.5)$.
  - Candidate $C$ has score $-1.8 < -1.5$. Thus, $C$ lies strictly below the chord and is **not on the upper-left convex hull**.
  - For every $\lambda \ge 0$:
    - If $\lambda < 0.5$: $S_A - \lambda C_A = -1.0 - 2\lambda > -1.8 - \lambda = S_C - \lambda C_C$.
    - If $\lambda \ge 0.5$: $S_B - \lambda C_B = -2.0 \ge -1.8 - \lambda = S_C - \lambda C_C$.
  - Under naive hull pruning, candidate $C$ is discarded.
- **Future Continuations from Step 1 to Terminal:**
  - Branch $A$: Requires cost $+2.0$, score $0.0$. Total cost = $4.0 > B=3.0 \implies$ **Infeasible!**
  - Branch $B$: Requires cost $+1.0$, score $-10.0$. Total cost = $1.0 \le 3.0$, Total score = $-12.0$.
  - Branch $C$: Requires cost $+2.0$, score $-0.1$. Total cost = $3.0 \le 3.0$, Total score = $\mathbf{-1.9}$.
- **Outcome:**
  - Candidate $C$ is the **unique constrained global optimum** ($S = -1.9, C = 3.0$).
  - Pruning candidate $C$ forces the search to return candidate $B$ ($S = -12.0$), suffering an optimality loss of $10.1$ log-prob units.

---

## 4. Provably Safe Dominance Pruning

### Theorem 3 (Safe Branch-and-Bound Pruning)
Let $u$ be an active prefix candidate.
Let $U(u)$ be an admissible upper bound on the future completion score ($S(y) \le S(u) + U(u)$ for all completions $y$ of $u$).
Let $c_{\min}(u)$ be an admissible lower bound on the future completion cost ($C(y) \ge C(u) + c_{\min}(u)$).
Let $S_{\text{inc}}$ be the score of any known feasible complete sequence ($C(y_{\text{inc}}) \le B$).

Candidate $u$ can be pruned without loss of the constrained global optimum if any of the following hold:
1. **Infeasibility:** $C(u) + c_{\min}(u) > B$.
2. **Incumbent Bound:** $S(u) + U(u) \le S_{\text{inc}}$.
3. **Markov State Dominance:** There exists an active candidate $u'$ sharing the identical forward state $\sigma(u') = \sigma(u)$ such that $C(u') \le C(u)$ and $S(u') \ge S(u)$.

#### Proof
1. If $C(u) + c_{\min}(u) > B$, then every completion $y$ of $u$ satisfies $C(y) \ge C(u) + c_{\min}(u) > B$. Thus, no feasible completion exists from $u$.
2. If $S(u) + U(u) \le S_{\text{inc}}$, then every completion $y$ of $u$ satisfies $S(y) \le S(u) + U(u) \le S_{\text{inc}}$. Thus, no completion of $u$ can strictly improve upon the known feasible incumbent $y_{\text{inc}}$.
3. If $u'$ shares the identical forward state $\sigma(u') = \sigma(u)$, then every suffix continuation $(y_{t+1}, \dots, y_T)$ valid for $u$ is also valid for $u'$ with identical transition probabilities and step costs. Since $C(u') \le C(u)$ and $S(u') \ge S(u)$, the completion of $u'$ will have total cost $\le$ the completion of $u$ and total score $\ge$ the completion of $u$. Thus, $u$ is weakly dominated by $u'$. $\blacksquare$
