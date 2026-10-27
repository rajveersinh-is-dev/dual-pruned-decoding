# Reproducibility Guide: Dual-Pruned Decoding (DPD)

This document provides complete instructions to independently reproduce all theoretical checks, unit/property tests, benchmarks, LaTeX tables, figures, and paper PDFs for **Dual-Pruned Decoding (DPD)**.

---

## 1. System Requirements & Environment

- **Operating System:** Windows, Linux, or macOS.
- **Python Version:** Python 3.10 to 3.14 (Tested on Python 3.14.6).
- **Core Dependencies:**
  - `torch >= 2.0.0`
  - `transformers >= 4.30.0`
  - `numpy >= 1.24.0`
  - `matplotlib >= 3.7.0`
  - `pytest >= 7.0.0`
  - `pytest-cov >= 4.0.0`
  - `ruff >= 0.1.0`
- **Document Compilers:**
  - `tectonic >= 0.15.0` (for compiling LaTeX paper directly to PDF)

---

## 2. Installation from Scratch

```bash
# Clone the repository
git clone https://github.com/rajveersinh-is-dev/dual-pruned-decoding.git
cd dual-pruned-decoding

# Create and activate a clean virtual environment
python -m venv .venv

# On Windows PowerShell:
.venv\Scripts\Activate.ps1
# On Linux/macOS:
# source .venv/bin/activate

# Upgrade packaging tools and install in editable mode with development & benchmark extras
python -m pip install --upgrade pip
python -m pip install -e ".[dev,benchmarks]"
```

---

## 3. Verification of Integrity (No Synthetic Preloaded Tables)

In compliance with our computational integrity standards:
- **Zero Preloaded Lookup Tables:** No precomputed recurrence constants or lookup tables exist in the codebase.
- **Dynamic Calculation:** Every score, cost, convex hull, Lagrangian multiplier, and certificate is dynamically computed from model distributions and user-specified cost models.
- **Statistical Timing:** All benchmark latencies are recorded in-process with `time.perf_counter_ns()`, preceded by 5 discarded warmup runs to eliminate cold-cache anomalies.

Run the test suite to verify 100% pass rate:
```bash
python -m pytest -v --cov=dpd tests/
```

Verify strict formatting and static code quality:
```bash
python -m ruff check dpd tests
```

---

## 4. Running the Complete Benchmark Suite

To execute the unadulterated benchmark suite (DistilGPT-2 NLP generation, structured code generation, and synthetic non-convex lattices):

```bash
python experiments/scripts/run_experiments.py
```

This script will:
1. Warm up the models (5 iterations per task).
2. Execute the evaluation runs deterministically across seeded prompts.
3. Save raw machine-readable results to `benchmarks/results/benchmark_results.json`.

---

## 5. Generating Publication Figures and LaTeX Tables

Once the benchmark results are produced, generate all publication figures (`.png` and `.pdf`) and LaTeX tables:

```bash
# Generate high-resolution figures
python experiments/scripts/generate_plots.py

# Generate LaTeX tables
python benchmarks/results/generate_tables.py
```

Generated figure artifacts in `paper/figures/`:
- `fig1_quality_vs_violation_pareto.pdf` / `.png`: Quality vs. Violation Pareto frontier.
- `fig2_latency_overhead.pdf` / `.png`: Wall-clock search latency comparison.
- `fig3_feasibility_vs_budget.pdf` / `.png`: Feasibility rate vs. budget envelope.
- `fig4_duality_gap.pdf` / `.png`: Distribution of dual suboptimality certificates.

Generated table artifacts in `paper/tables/`:
- `table1_lm_results.tex`: DistilGPT-2 real LM benchmark across budgets $B \in \{10, 15, 20\}$.
- `table2_synthetic_results.tex`: Non-convex synthetic lattice results with exact optimality certification.

---

## 6. Compiling the Research Paper

Compile the LaTeX manuscript with complete bibliography and embedded figures:

```bash
cd paper
tectonic main.tex
cd ..
```

The resulting paper will be written to `paper/main.pdf` (and `paper/paper.pdf`).
