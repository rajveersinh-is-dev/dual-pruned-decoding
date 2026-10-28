# Independent Hostile Review & Correctness Audit

This document records the four-persona adversarial audit conducted on **Dual-Pruned Decoding (DPD)** to resolve critical mathematical, implementation, packaging, and empirical defects. Every identified defect is documented with a concrete counterexample, proof, or repair.

---

## 1. Reviewer A: Theoretical Computer Scientist

### Issue A.1 (CRITICAL): Inadmissible Completion Bound in `BudgetAwareBound`
- **Objection:** In `dpd/bounds.py`, `BudgetAwareBound` configured a default `max_log_prob_per_step = -0.05`. Any nonpositive constant is not automatically an upper bound on future log-probabilities. In autoregressive language models, next-token log-probabilities can approach $0.0$ (e.g. $p = 0.999 \implies \log p \approx -0.001 > -0.05$). Under $\alpha = -0.05$, the upper bound underestimates the true score of valid completions, causing `filter_by_incumbent_bound` to falsely prune the global constrained optimum!
- **Proof & Counterexample:** Formalized as **Counterexample 3** in `research/counterexamples.md` and verified in `tests/regression/test_adversarial_counterexamples.py::test_counterexample_3_inadmissible_bound_discards_optimum`.
  - Let prefix $u$ have cost 1.0, score -2.0, with a future token having $p = 0.99 \implies \log p = -0.01$.
  - Feasible incumbent exists with score -2.02.
  - With $\alpha = -0.05$, $S(u) + 1 \cdot (-0.05) = -2.05 \le -2.02 \implies$ prefix $u$ is discarded.
  - The true completion has score $-2.0 + (-0.01) = -2.01 > -2.02$, so the global optimum was discarded!
- **Resolution:**
  1. Default `max_log_prob_per_step` in `BudgetAwareBound` and `BudgetConditionedBound` is set to `0.0`, which is provably admissible for all autoregressive probability distributions ($\log p \le 0$).
  2. Any tighter negative bound requires an explicit opt-in flag (`allow_heuristic_tighter_bound=True`).
  3. Added `is_admissible` property on bounds to prevent heuristic bounds from claiming exact certified pruning.

### Issue A.2 (CRITICAL): Circular Reasoning in Lean 4 Zero-Gap Theorem
- **Objection:** In `formal/DPDTheory.lean`, theorem `certified_optimality_zero_gap` took hypothesis `h_is_opt : ∀ c, is_feasible c budget → c.score ≤ opt.score` and deduced `inc.score = opt.score`. Assuming the existence of an optimal sequence and that all feasible sequences are bounded by it begged the very question the certificate aimed to prove.
- **Resolution:**
  1. Reformulated `certified_optimality_zero_gap` without assuming `opt`:
     ```lean
     theorem certified_optimality_zero_gap (inc : SequenceCandidate) (budget lambda dual_val : Int)
         (_h_inc_feas : is_feasible inc budget)
         (h_lam : 0 ≤ lambda)
         (h_dual_bound : ∀ c : SequenceCandidate, lagrangian c budget lambda ≤ dual_val)
         (h_zero_gap : dual_val = inc.score) :
         ∀ c : SequenceCandidate, is_feasible c budget → c.score ≤ inc.score
     ```
  2. Directly proved from pointwise weak duality that every feasible candidate $c$ satisfies $c.\text{score} \le \text{inc.score}$. This proves that `inc` satisfies the definition of global optimality from first principles.
  3. Pinned toolchain `leanprover/lean4:v4.35.0` in `formal/lean-toolchain` and added CI verification step.

### Issue A.3 (CRITICAL): Unsound Dual Certificate Scope Claims
- **Objection:** `compute_dual_certificate` calculated the Lagrangian upper bound strictly over `all_terminal_candidates`. When beam width truncated unexpanded branches or static hull pruning dropped candidates, unexplored feasible sequences were omitted from `all_terminal_candidates`. Claiming `certified_optimal=True` globally was mathematically false.
- **Resolution:**
  1. Introduced explicit `CertificateScope` enum: `RESTRICTED_SUBSET`, `RESTRICTED_LATTICE`, `GLOBAL_EXHAUSTIVE`, `BRANCH_AND_BOUND`.
  2. Updated `DualCertificate` to distinguish `certified_globally_optimal` from `certified_optimal_within_scope`.
  3. In `dpd/decoding.py`, tracked `was_search_exhaustive`. If beam width or heuristic pruning drops any candidate, scope is set to `RESTRICTED_SUBSET` and `certified_optimal` is strictly `False`.
  4. Global optimality is certified only when all unexplored branches are proved bounded $\le S_{\text{inc}}$ via admissible bounds or proved infeasible.

---

## 2. Reviewer B: Language-Model Decoding Researcher

### Issue B.1 (HIGH): Omission of EOS Token in LM Generation Environment
- **Objection:** In `benchmarks/tasks/lm_budget_task.py`, candidate transitions were restricted to `torch.topk(log_probs, k=8)`. On open-ended prompts in DistilGPT-2, the EOS token (`50256`) is rarely in the top 8 branches. Consequently, sequences could never terminate naturally within budget, forcing 50% of runs into `infeasible_fallback`.
- **Resolution:**
  1. Updated `LMGenerationEnvironment.create_step_fn` to guarantee that `self.eos_token_id` is always available as a candidate branch with its exact model log-probability `log_probs[self.eos_token_id].item()`.
  2. This allows decoders to legally evaluate terminating within budget.
  3. Documented that pure likelihood decoding without length normalization naturally favors shorter sequences because log-probabilities are additive and nonpositive.

### Issue B.2 (HIGH): Exactness within Beam vs. Global Lattice
- **Objection:** The paper must not conflate exactness within a truncated beam with exactness over the full language model decoding lattice.
- **Resolution:**
  1. Updated `docs/algorithm.md`, `docs/guarantees.md`, and `paper/main.tex` to explicitly classify beam selection as a heuristic search operation.
  2. Explicitly documented that DPD changes prefix survival order compared to standard beam search, and no mathematical equivalence between the two truncated search trees is claimed.
  3. Formally distinguished the multiplier update (projected subgradient / PI) as an adaptive search heuristic rather than a guaranteed convex optimization step on discrete non-convex frontiers.

---

## 3. Reviewer C: Systems Engineer

### Issue C.1 (HIGH): Platform-Specific Compiled Binaries Committed to Git
- **Objection:** `dpd/dpd_kernel.dll` (a 115 KB Windows x86_64 PE dynamic library) was committed to the git repository. Distributing platform-specific compiled binaries in source repositories breaks reproducibility on Linux/macOS and poses security and licensing concerns.
- **Resolution:**
  1. Removed `dpd/dpd_kernel.dll` from git tracking (`git rm --cached dpd/dpd_kernel.dll`).
  2. Added `*.dll`, `*.so`, and `*.dylib` to `.gitignore`.
  3. Ensured that native kernels are built on the fly in CI and development via `python scripts/build_kernel.py` using `dtolnay/rust-toolchain@stable`.

### Issue C.2 (MEDIUM): Discrepancy between Lean Discrete Integers and Python Floating-Point
- **Objection:** The Lean 4 formalization operates over discrete exact integers (`Int`), whereas Python executes in IEEE 754 double-precision floats (`float`) with rounding errors and tolerance epsilons (`eps = 1e-9`). Claiming that the Python algorithm is "formally verified" is invalid.
- **Resolution:**
  1. Removed all assertions that the Python code is formally verified end-to-end.
  2. Documented in `formal/DPDTheory.lean`, `docs/guarantees.md`, and `README.md` that Lean 4 verifies the idealized discrete mathematical theory, while Python implements a numerical floating-point realization with finite tolerances.

---

## 4. Reviewer D: Skeptical Publication Reviewer

### Issue D.1 (CRITICAL): Contradiction between README Claims and Table Data
- **Objection:** `README.md` Figure 1 caption claimed: *"DPD establishes a superior non-dominated Pareto frontier, achieving 0% constraint violations at B=15 and B=20 while standard beam search suffers severe budget overrun."* Yet the table in the very same README reported a **50.0% feasible rate (50% violation rate)** for DPD at $B=15$ and $B=20$!
- **Resolution:**
  1. Investigated the source data in `benchmarks/results/benchmark_results.json`.
  2. Reconciled the text and table: at $B=15$, DPD achieved 50.0% feasibility; at $B=20$, DPD achieved 60.0% feasibility.
  3. Corrected Figure 1 caption in `README.md` and `paper/main.tex` to accurately reflect the data without false claims of 0% violations.
  4. Updated the README table to match `benchmark_results.json` and `paper/tables/table1_lm_results.tex` with 100% numerical parity.

### Issue D.2 (HIGH): Fake Publication Identifier (arXiv Badge)
- **Objection:** `README.md` displayed a badge: `[![arXiv](https://img.shields.io/badge/arXiv-2610.xxxxx-b31b1b.svg)]`. Fabricated arXiv identifiers undermine the credibility of research.
- **Resolution:**
  1. Removed the fake badge from `README.md`.
  2. Added clean CI, License, and Python badges.

### Issue D.3 (MEDIUM): Pilot vs. Production Evaluation Scope
- **Objection:** Evaluating on 10 prompts or 15 synthetic instances is a pilot study, not a massive production benchmark. Claims of broad empirical superiority must be bounded.
- **Resolution:**
  1. Explicitly designated Task 1 as a **pilot evaluation** on $N=10$ representative prompts.
  2. Explicitly documented hardware specifications (CPU execution, Windows x86_64, Python 3.14.0, PyTorch 2.x, DistilGPT-2 revision).
  3. Reported uncertainties, average pruned candidates, and certified rates without extrapolating beyond the empirical data.

---

## Prioritized Issue Summary & Action Matrix

| Priority | Issue ID | Area | Defect Description | Status |
|---|---|---|---|---|
| **P0** | A.1 | Bounds | Inadmissible $\alpha = -0.05$ completion bound pruned true optimum | **RESOLVED** (Default 0.0, Counterexample 3, Regression tests) |
| **P0** | A.2 | Theory | Circular Lean 4 theorem assumed `h_is_opt` | **RESOLVED** (Re-proven non-circularly in Lean 4) |
| **P0** | A.3 | Bounds | Certificate claimed global optimality over subset | **RESOLVED** (Scope tracking, Restricted subset vs Global) |
| **P0** | D.1 | Claims | README claimed 0% violations vs 50% in table | **RESOLVED** (Reconciled with raw benchmark JSON) |
| **P1** | B.1 | Tasks | EOS excluded from LM top-8 branches | **RESOLVED** (EOS candidate branch included with model log-prob) |
| **P1** | C.1 | Packaging | Committed binary `dpd_kernel.dll` in repo | **RESOLVED** (Removed from git, ignored in .gitignore) |
| **P1** | D.2 | Claims | Fake arXiv identifier badge | **RESOLVED** (Removed badge) |
| **P2** | C.2 | Theory | Conflating Lean `Int` proof with Python float code | **RESOLVED** (Clarified discrete math vs float realization) |
| **P2** | D.3 | Claims | Unqualified claims from pilot sample size | **RESOLVED** (Designated as pilot study with sample sizes) |
