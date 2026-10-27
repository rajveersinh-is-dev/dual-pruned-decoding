# Formal Proofs and Theoretical Guarantees: Dual-Pruned Decoding (DPD)

**Project:** Dual-Pruned Decoding (DPD)  
**Date:** October 2026  
**Status:** Phase 2 Mathematical Proofs  

---

## 1. Theorem 1: Certified Bounded Suboptimality via Lagrangian Dual

### 1.1 Setting and Definitions
Let $\mathcal{G} = (\mathcal{V}_\mathcal{G}, \mathcal{E}_\mathcal{G})$ be a finite directed acyclic graph representing a search lattice of token sequences originating from prompt $x$.
Let $\mathcal{Y}_\mathcal{G} \subseteq \mathcal{V}^*$ be the set of complete terminal sequences in $\mathcal{G}$.
For any sequence $y \in \mathcal{Y}_\mathcal{G}$, let $S(y) = \sum_{t=1}^{|y|} \log p_\theta(y_t \mid x, y_{<t})$ and $C(y) = \sum_{t=1}^{|y|} c(x, y_{<t}, y_t)$, where $c \ge 0$.
The feasible set on $\mathcal{G}$ is:
$$\mathcal{F}_B(\mathcal{G}) = \{ y \in \mathcal{Y}_\mathcal{G} : C(y) \le B \}$$
Assume $\mathcal{F}_B(\mathcal{G}) \neq \emptyset$. Let $y^\star \in \arg\max_{y \in \mathcal{F}_B(\mathcal{G})} S(y)$ be an optimal constrained solution on $\mathcal{G}$, with optimal value $S^\star = S(y^\star)$.

Define the Lagrangian relaxation for multiplier $\lambda \ge 0$:
$$\mathcal{L}(y, \lambda) = S(y) - \lambda (C(y) - B)$$
and the dual function:
$$d(\lambda) = \max_{y \in \mathcal{Y}_\mathcal{G}} \mathcal{L}(y, \lambda)$$

### 1.2 Theorem Statement
**Theorem 1 (Certified Bounded Suboptimality):**  
Let $\Lambda = \{\lambda^{(1)}, \lambda^{(2)}, \dots, \lambda^{(P)}\} \subset \mathbb{R}_+$ be any finite collection of evaluated nonnegative multipliers.  
Let $d^\star_\Lambda = \min_{\lambda \in \Lambda} d(\lambda)$.  
Let $y_{\text{inc}} \in \mathcal{F}_B(\mathcal{G})$ be any feasible incumbent sequence.  
Then:
1. **Valid Dual Bound:** $S^\star \le d^\star_\Lambda$.
2. **Suboptimality Certificate:** The additive score suboptimality of the incumbent is bounded by:
   $$S^\star - S(y_{\text{inc}}) \le \Delta_\Lambda \quad \text{where} \quad \Delta_\Lambda \triangleq d^\star_\Lambda - S(y_{\text{inc}}) \ge 0$$
3. **Exactness Criterion:** If $\Delta_\Lambda = 0$, then $y_{\text{inc}}$ is certified to be an exact global maximizer on $\mathcal{G}$:
   $$S(y_{\text{inc}}) = S(y^\star) = \max_{y \in \mathcal{F}_B(\mathcal{G})} S(y)$$

### 1.3 Proof
*Proof of 1:*  
Let $y^\star \in \mathcal{F}_B(\mathcal{G})$. By definition of feasibility, $C(y^\star) \le B$, which implies $B - C(y^\star) \ge 0$.  
For any $\lambda \ge 0$, multiplying by $\lambda$ preserves non-negativity:
$$\lambda (B - C(y^\star)) \ge 0$$
Adding $S(y^\star)$ to both sides:
$$S(y^\star) \le S(y^\star) + \lambda (B - C(y^\star)) = \mathcal{L}(y^\star, \lambda)$$
Since $\mathcal{Y}_\mathcal{G}$ contains $\mathcal{F}_B(\mathcal{G})$ (the feasible set is a subset of all terminal sequences in $\mathcal{G}$):
$$\mathcal{L}(y^\star, \lambda) \le \max_{y \in \mathcal{Y}_\mathcal{G}} \mathcal{L}(y, \lambda) = d(\lambda)$$
Since this holds for every $\lambda \ge 0$, it holds for every $\lambda \in \Lambda$. Taking the minimum over $\lambda \in \Lambda$:
$$S^\star \le \min_{\lambda \in \Lambda} d(\lambda) = d^\star_\Lambda$$
This establishes Part 1.

*Proof of 2:*  
Subtracting $S(y_{\text{inc}})$ from both sides of $S^\star \le d^\star_\Lambda$:
$$S^\star - S(y_{\text{inc}}) \le d^\star_\Lambda - S(y_{\text{inc}}) = \Delta_\Lambda$$
Furthermore, since $y_{\text{inc}} \in \mathcal{F}_B(\mathcal{G})$, by definition of $S^\star = \max_{y \in \mathcal{F}_B(\mathcal{G})} S(y)$, we have $S(y_{\text{inc}}) \le S^\star$. Therefore $\Delta_\Lambda \ge S^\star - S(y_{\text{inc}}) \ge 0$. This establishes Part 2.

*Proof of 3:*  
If $\Delta_\Lambda = 0$, then $0 \le S^\star - S(y_{\text{inc}}) \le \Delta_\Lambda = 0$, which implies $S(y_{\text{inc}}) = S^\star$. Since $y_{\text{inc}} \in \mathcal{F}_B(\mathcal{G})$, it is an exact constrained optimum on $\mathcal{G}$. $\blacksquare$

---

## 2. Theorem 2: Upper-Left Convex Hull Invariance for Linear Scalarizations

### 2.1 Setting
Let $\mathcal{C} = \{(C_i, S_i)\}_{i=1}^M \subset \mathbb{R}^2$ be a finite non-empty set of candidate points in the cost-score plane.
For any resource price $\lambda \ge 0$, define the scalarized objective function:
$$f_\lambda(C, S) = S - \lambda C$$
Let $\mathcal{H}_{\text{UL}}(\mathcal{C})$ denote the upper-left convex hull of $\mathcal{C}$, defined as the lower convex envelope of the point set when cost is on the x-axis and negative score $-S$ is on the y-axis, or equivalently the extreme points in $\mathcal{C}$ that can be supported by lines with slope $\lambda \in [0, \infty)$.
Let $\text{Ext}(\mathcal{H}_{\text{UL}})$ denote the extreme vertices of this hull.

### 2.2 Theorem Statement
**Theorem 2 (Hull Invariance under Lagrangian Scalarization):**  
For any price $\lambda \ge 0$:
$$\max_{(C, S) \in \mathcal{C}} (S - \lambda C) = \max_{(C, S) \in \text{Ext}(\mathcal{H}_{\text{UL}})} (S - \lambda C)$$
Furthermore, if a point $(C_k, S_k) \notin \mathcal{H}_{\text{UL}}(\mathcal{C})$ lies strictly in the interior below the hull, then:
$$\forall \lambda \ge 0, \quad S_k - \lambda C_k < \max_{(C, S) \in \text{Ext}(\mathcal{H}_{\text{UL}})} (S - \lambda C)$$

### 2.3 Proof
*Proof:*  
The function $f_\lambda(C, S) = S - \lambda C$ is a linear function on $\mathbb{R}^2$.  
Let $\text{Conv}(\mathcal{C})$ be the convex hull of $\mathcal{C}$.  
A fundamental result of linear programming and convex analysis (e.g., Rockafellar 1970, Corollary 32.3.1) states that the maximum of any linear function over a compact polyhedral set is achieved at an extreme point (vertex) of the set:
$$\max_{(C, S) \in \mathcal{C}} (S - \lambda C) = \max_{(C, S) \in \text{Conv}(\mathcal{C})} (S - \lambda C) = \max_{(C, S) \in \text{Ext}(\text{Conv}(\mathcal{C}))} (S - \lambda C)$$
Now consider the slope of the linear objective: the level curves are $S = \lambda C + \text{constant}$.  
Since $\lambda \ge 0$, the normal vector to the level lines pointing in the direction of increasing objective is $(-\lambda, 1)$.  
Since $-\lambda \le 0$ and $1 > 0$, this normal vector points strictly into the second quadrant (upper-left direction: non-positive cost, positive score).  
Therefore, the supporting hyperplanes with normal $(-\lambda, 1)$ for $\lambda \ge 0$ can only make contact with the set $\text{Conv}(\mathcal{C})$ along its **upper-left boundary**, which is precisely $\mathcal{H}_{\text{UL}}(\mathcal{C})$.  
Any vertex of $\text{Conv}(\mathcal{C})$ on the lower or right boundaries has an inward-pointing normal with $\lambda < 0$ or negative score direction, and hence can never maximize $S - \lambda C$ for any $\lambda \ge 0$.  
Therefore:
$$\max_{(C, S) \in \mathcal{C}} (S - \lambda C) = \max_{(C, S) \in \text{Ext}(\mathcal{H}_{\text{UL}})} (S - \lambda C)$$
If a point $(C_k, S_k)$ lies strictly below the upper-left boundary, by the Supporting Hyperplane Theorem there exists a line separating $(C_k, S_k)$ from the epigraph of the upper-left hull, which implies $(C_k, S_k)$ achieves a strictly lower objective value than the boundary point vertically above it or horizontally to its left for any non-negative $\lambda \ge 0$. $\blacksquare$

---

## 3. Theorem 3: Safe Dominance Pruning with Admissible Lookahead

### 3.1 Setting
Let $\mathcal{T}(x)$ be the search tree of all prefix sequences starting from prompt $x$.
For any prefix $u \in \mathcal{T}(x)$, let $\text{Comp}(u) = \{ \tau \in \mathcal{V}^* : u \circ \tau \in \mathcal{Y}(x) \}$ be the set of completions that reach a terminal node.
The constrained completion set of $u$ with remaining budget $R(u) = B - C(u)$ is:
$$\text{Comp}_B(u) = \{ \tau \in \text{Comp}(u) : C(\tau \mid u) \le R(u) \}$$
The optimal completion score of $u$ is:
$$S^\star(u) = \begin{cases} \max_{\tau \in \text{Comp}_B(u)} [S(u) + S(\tau \mid u)], & \text{if } \text{Comp}_B(u) \neq \emptyset \\ -\infty, & \text{if } \text{Comp}_B(u) = \emptyset \end{cases}$$
Let $U(u)$ be an **admissible upper bound** on the additional completion score, satisfying:
$$\forall \tau \in \text{Comp}_B(u), \quad S(\tau \mid u) \le U(u)$$
Let $c_{\min}(u)$ be an **admissible lower bound** on completion cost, satisfying:
$$\forall \tau \in \text{Comp}(u), \quad C(\tau \mid u) \ge c_{\min}(u)$$
Let $y_{\text{inc}} \in \mathcal{F}_B(x)$ be any known feasible complete sequence with score $S_{\text{inc}} = S(y_{\text{inc}})$.

### 3.2 Theorem Statement
**Theorem 3 (Preservation of Optimal Solutions under Pruning):**  
Let $\mathcal{A}$ be any subset of prefixes in $\mathcal{T}(x)$. Suppose candidate $u \in \mathcal{A}$ is pruned if ANY of the following conditions hold:
1. **(Infeasibility Condition):** $C(u) + c_{\min}(u) > B$.
2. **(Bound Condition):** $S(u) + U(u) \le S_{\text{inc}}$.
3. **(State-Dominance Condition):** There exists $u' \in \mathcal{A}$ such that $u' \sim u$ (state-equivalent), $C(u') \le C(u)$, and $S(u') \ge S(u)$, with $(C(u'), S(u')) \neq (C(u), S(u))$ (or tie broken in favor of $u'$).

Then the set of globally optimal constrained solutions is strictly preserved:
$$\max_{u \in \mathcal{A} \setminus \{u\}} S^\star(u) = \max_{u \in \mathcal{A}} S^\star(u)$$

### 3.3 Proof
*Proof:*  
We prove that under each condition, pruning $u$ cannot eliminate the unique strictly optimal sequence.

*Case 1 (Infeasibility):*  
Suppose $C(u) + c_{\min}(u) > B$. For any $\tau \in \text{Comp}(u)$, $C(u \circ \tau) = C(u) + C(\tau \mid u) \ge C(u) + c_{\min}(u) > B$.  
Thus $\text{Comp}_B(u) = \emptyset$, and $S^\star(u) = -\infty$. Pruning $u$ does not eliminate any feasible solution, let alone an optimal one.

*Case 2 (Bound Condition):*  
Suppose $S(u) + U(u) \le S_{\text{inc}}$.  
For any $\tau \in \text{Comp}_B(u)$:
$$S(u \circ \tau) = S(u) + S(\tau \mid u) \le S(u) + U(u) \le S_{\text{inc}}$$
Thus $S^\star(u) \le S_{\text{inc}}$. Since $y_{\text{inc}}$ is already known and retained in the incumbent pool, no completion of $u$ can strictly improve upon the incumbent. If $S^\star(u) < S_{\text{inc}}$, $u$ cannot be optimal. If $S^\star(u) = S_{\text{inc}}$, an optimal solution ($y_{\text{inc}}$) is already preserved. Hence pruning $u$ preserves the global maximum.

*Case 3 (State Dominance):*  
Suppose $u' \sim u$, $C(u') \le C(u)$, and $S(u') \ge S(u)$.  
Because $u' \sim u$, the future distributions and transitions are identical: for every $\tau \in \text{Comp}(u)$, $\tau \in \text{Comp}(u')$, with $S(\tau \mid u') = S(\tau \mid u)$ and $C(\tau \mid u') = C(\tau \mid u)$.  
Therefore:
1. Feasibility: If $\tau \in \text{Comp}_B(u)$, then $C(u') + C(\tau \mid u') \le C(u) + C(\tau \mid u) \le B$, so $\tau \in \text{Comp}_B(u')$.
2. Score: $S(u' \circ \tau) = S(u') + S(\tau \mid u') \ge S(u) + S(\tau \mid u) = S(u \circ \tau)$.
Thus for every feasible completion of $u$, there exists an equally or more feasible completion of $u'$ with an equal or strictly higher score.  
Therefore $S^\star(u') \ge S^\star(u)$. Pruning $u$ while retaining $u'$ preserves $\max_{w} S^\star(w)$. $\blacksquare$

---

## 4. Complexity and Convergence of Adaptive Lagrangian Price Updates

### 4.1 Update Rule
Let $\lambda_t \ge 0$ be the resource price at step $t$.  
At step $t$, let $\bar{C}_t$ be the average cost of current beam prefixes, and let $B_t = B \cdot \frac{t}{T_{\max}}$ be the nominal cumulative budget target at step $t$.  
Define the resource consumption signal:
$$g_t = \bar{C}_t - B_t$$
The projected subgradient update with step size $\eta_t > 0$ is:
$$\lambda_{t+1} = \max\left\{0, \lambda_t + \eta_t g_t\right\}$$

### 4.2 Properties
- When $\bar{C}_t > B_t$ (over-budget trajectory), $g_t > 0$, increasing $\lambda_{t+1}$ and penalizing expensive expansions.
- When $\bar{C}_t < B_t$ (under-budget trajectory), $g_t < 0$, decreasing $\lambda_{t+1}$ and allowing high-probability tokens even if cost $> 0$.
- Boundedness: If step costs are bounded by $c_{\max}$, the subgradient is bounded: $|g_t| \le c_{\max} \cdot T_{\max}$. For diminishing step sizes $\eta_t = \eta_0 / \sqrt{t}$, standard non-smooth convex optimization guarantees the convergence of dual bounds $d(\lambda)$ to within an $O(1/\sqrt{T})$ neighborhood of the optimal dual value $d^\star$.
