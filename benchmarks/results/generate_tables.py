"""Generates LaTeX tables directly from verified benchmark_results.json."""

from __future__ import annotations

import json
import os


def generate_tables(results_path: str, tables_dir: str):
    """Parses benchmark_results.json and writes LaTeX tables."""
    os.makedirs(tables_dir, exist_ok=True)
    with open(results_path) as f:
        data = json.load(f)

    # Table 1: DistilGPT-2 Results across Budgets
    lm_data = data["task_1_lm"]
    budgets = ["10.0", "15.0", "20.0"]
    methods = ["DPD (Hybrid)", "Standard Beam", "Length Penalty", "Greedy", "DPD (No Hull)", "DPD (Fixed Lambda)"]

    t1_lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        (r"\caption{Empirical evaluation on Budget-Constrained Text Generation using DistilGPT-2 across $N=10$ evaluation prompts. "
        r"Score denotes model log-probability $S(y)$, Feasible \% denotes hard budget compliance, Latency is measured in seconds, "
        r"and Pruned denotes mean candidates discarded per sequence.}"),
        r"\label{tab:lm_results}",
        r"\begin{tabular}{llccccc}",
        r"\toprule",
        r"Budget & Decoding Method & Score ($\uparrow$) & Cost & Feasible (\%) ($\uparrow$) & Latency (s) ($\downarrow$) & Pruned / Step \\",
        r"\midrule",
    ]

    for b in budgets:
        b_label = f"$B={int(float(b))}$"
        for idx, m in enumerate(methods):
            m_res = lm_data[b][m]
            prefix = b_label if idx == 0 else ""
            score_str = f"{m_res['average_score']:.2f}"
            cost_str = f"{m_res['average_cost']:.1f}"
            feas_str = f"{m_res['feasible_rate'] * 100:.1f}\\%"
            lat_str = f"{m_res['average_latency_sec']:.2f}"
            pruned_str = f"{m_res['average_pruned_candidates']:.1f}"
            t1_lines.append(f"{prefix} & {m} & {score_str} & {cost_str} & {feas_str} & {lat_str} & {pruned_str} \\\\")
        t1_lines.append(r"\midrule")

    t1_lines[-1] = r"\bottomrule"
    t1_lines.extend([
        r"\end{tabular}",
        r"\end{table*}",
    ])

    t1_path = os.path.join(tables_dir, "table1_lm_results.tex")
    with open(t1_path, "w") as f:
        f.write("\n".join(t1_lines) + "\n")
    print(f"Saved: {t1_path}")

    # Table 2: Synthetic Lattice Ground Truth Comparison
    synth_data = data["task_3_synthetic"]
    synth_methods = ["DPD (Hybrid)", "Standard Beam", "Length Penalty", "DPD (No Hull)", "DPD (Fixed Lambda)"]

    t2_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        (r"\caption{Comparison on Synthetic Non-Convex Lattices ($N=15$ instances) with verified ground-truth optimality. "
        r"Cert.\% indicates fraction of instances with certified global optimality ($\Delta \le 10^{-4}$).}"),
        r"\label{tab:synthetic_results}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Method & Score ($\uparrow$) & Cost & Feas.\% ($\uparrow$) & Latency (ms) & Cert.\% \\",
        r"\midrule",
    ]

    for m in synth_methods:
        s = synth_data[m]
        score_str = f"{s['average_score']:.2f}"
        cost_str = f"{s['average_cost']:.2f}"
        feas_str = f"{s['feasible_rate'] * 100:.1f}\\%"
        lat_str = f"{s['average_latency_sec'] * 1000:.2f}"
        cert_str = f"{s['certified_optimal_rate'] * 100:.1f}\\%"
        t2_lines.append(f"{m} & {score_str} & {cost_str} & {feas_str} & {lat_str} & {cert_str} \\\\")

    t2_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    t2_path = os.path.join(tables_dir, "table2_synthetic_results.tex")
    with open(t2_path, "w") as f:
        f.write("\n".join(t2_lines) + "\n")
    print(f"Saved: {t2_path}")


if __name__ == "__main__":
    results_path = os.path.join("benchmarks", "results", "benchmark_results.json")
    tables_dir = os.path.join("paper", "tables")
    generate_tables(results_path, tables_dir)
