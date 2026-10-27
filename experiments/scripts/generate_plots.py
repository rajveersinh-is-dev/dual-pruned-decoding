"""Generates publication-quality figures and plots from verified experimental results."""

from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt
import numpy as np


def set_publication_style():
    """Sets publication style settings for Matplotlib."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "figure.dpi": 300,
        "lines.linewidth": 1.8,
        "lines.markersize": 6,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
    })


def generate_all_plots(results_path: str, figures_dir: str):
    """Generates all 4 scientific figures from verified benchmark results."""
    set_publication_style()
    os.makedirs(figures_dir, exist_ok=True)

    with open(results_path) as f:
        data = json.load(f)

    # -------------------------------------------------------------
    # Figure 1: Quality (Score) vs. Violation Rate (Pareto Frontier)
    # -------------------------------------------------------------
    _fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Subplot A: Language Model Task (DistilGPT-2)
    ax1 = axes[0]
    lm_data = data["task_1_lm"]
    budgets_lm = sorted(float(b) for b in lm_data)

    methods = [
        "DPD (Hybrid)",
        "Standard Beam",
        "Length Penalty",
        "Greedy",
        "DPD (No Hull)",
        "DPD (Fixed Lambda)",
    ]
    colors = {
        "DPD (Hybrid)": "#1f77b4",       # Deep blue
        "Standard Beam": "#d62728",      # Crimson red
        "Length Penalty": "#ff7f0e",     # Amber orange
        "Greedy": "#7f7f7f",             # Gray
        "DPD (No Hull)": "#2ca02c",      # Forest green
        "DPD (Fixed Lambda)": "#9467bd",  # Purple
    }
    markers = {
        "DPD (Hybrid)": "o",
        "Standard Beam": "s",
        "Length Penalty": "^",
        "Greedy": "x",
        "DPD (No Hull)": "D",
        "DPD (Fixed Lambda)": "v",
    }

    for method in methods:
        violations = []
        scores = []
        for b in budgets_lm:
            m_res = lm_data[str(b)][method]
            violations.append(m_res["violation_rate"] * 100.0)
            scores.append(m_res["average_score"])

        ax1.plot(
            violations,
            scores,
            label=method,
            color=colors.get(method, "#333"),
            marker=markers.get(method, "o"),
            alpha=0.9,
        )

    ax1.set_xlabel("Hard Constraint Violation Rate (%)")
    ax1.set_ylabel("Average Log-Probability Score")
    ax1.set_title("(a) DistilGPT-2 Text Generation")
    ax1.grid(True)
    ax1.legend(loc="lower right", framealpha=0.9)

    # Subplot B: Structured Code Task
    ax2 = axes[1]
    code_data = data["task_2_code"]
    budgets_code = sorted(float(b) for b in code_data)

    for method in methods:
        violations = []
        scores = []
        for b in budgets_code:
            m_res = code_data[str(b)][method]
            violations.append(m_res["violation_rate"] * 100.0)
            scores.append(m_res["average_score"])

        ax2.plot(
            violations,
            scores,
            label=method,
            color=colors.get(method, "#333"),
            marker=markers.get(method, "o"),
            alpha=0.9,
        )

    ax2.set_xlabel("Hard Constraint Violation Rate (%)")
    ax2.set_ylabel("Average Log-Probability Score")
    ax2.set_title("(b) Structured Code Generation")
    ax2.grid(True)
    ax2.legend(loc="lower right", framealpha=0.9)

    plt.tight_layout()
    fig1_path = os.path.join(figures_dir, "fig1_quality_vs_violation_pareto.png")
    fig1_pdf = os.path.join(figures_dir, "fig1_quality_vs_violation_pareto.pdf")
    plt.savefig(fig1_path, dpi=300, bbox_inches="tight")
    plt.savefig(fig1_pdf, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig1_path} and {fig1_pdf}")

    # -------------------------------------------------------------
    # Figure 2: Latency vs. Average Expansions (Efficiency Profile)
    # -------------------------------------------------------------
    _fig, ax = plt.subplots(figsize=(7, 5))
    synth_data = data["task_3_synthetic"]

    synth_methods = ["DPD (Hybrid)", "Standard Beam", "Length Penalty", "DPD (No Hull)", "DPD (Fixed Lambda)"]
    latencies = [synth_data[m]["average_latency_sec"] * 1000.0 for m in synth_methods]
    [synth_data[m]["average_pruned_candidates"] for m in synth_methods]
    bar_colors = [colors.get(m, "#555") for m in synth_methods]

    bars = ax.bar(synth_methods, latencies, color=bar_colors, alpha=0.85, width=0.55)
    ax.set_ylabel("Average Wall-Clock Latency (ms)")
    ax.set_title("Search Efficiency & Overhead on Finite Lattices")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # Label values on bars
    for bar in bars:
        yval = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + 0.02 * max(latencies),
            f"{yval:.2f} ms",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    fig2_path = os.path.join(figures_dir, "fig2_latency_overhead.png")
    fig2_pdf = os.path.join(figures_dir, "fig2_latency_overhead.pdf")
    plt.savefig(fig2_path, dpi=300, bbox_inches="tight")
    plt.savefig(fig2_pdf, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig2_path} and {fig2_pdf}")

    # -------------------------------------------------------------
    # Figure 3: Feasible Rate Comparison across Budgets
    # -------------------------------------------------------------
    _fig, ax = plt.subplots(figsize=(8, 5))
    x_indices = np.arange(len(budgets_lm))
    width = 0.15

    for idx, method in enumerate(methods):
        rates = [lm_data[str(b)][method]["feasible_rate"] * 100.0 for b in budgets_lm]
        ax.bar(
            x_indices + idx * width - 0.35,
            rates,
            width=width,
            label=method,
            color=colors.get(method, "#555"),
            alpha=0.9,
        )

    ax.set_xlabel("Token Budget (B)")
    ax.set_ylabel("Feasible Output Rate (%)")
    ax.set_title("Constraint Satisfaction Rate vs. Budget (DistilGPT-2)")
    ax.set_xticks(x_indices)
    ax.set_xticklabels([f"B={int(b)}" for b in budgets_lm])
    ax.set_ylim(0, 110)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", framealpha=0.9)

    plt.tight_layout()
    fig3_path = os.path.join(figures_dir, "fig3_feasibility_vs_budget.png")
    fig3_pdf = os.path.join(figures_dir, "fig3_feasibility_vs_budget.pdf")
    plt.savefig(fig3_path, dpi=300, bbox_inches="tight")
    plt.savefig(fig3_pdf, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig3_path} and {fig3_pdf}")

    # -------------------------------------------------------------
    # Figure 4: Dual Gap and Certification Rate
    # -------------------------------------------------------------
    _fig, ax = plt.subplots(figsize=(7, 4.5))
    synth_dpd = synth_data["DPD (Hybrid)"]
    synth_dpd["average_dual_gap"]
    dpd_cert_rate = synth_dpd["certified_optimal_rate"] * 100.0

    gaps = [synth_data[m]["average_dual_gap"] for m in synth_methods]
    ax.bar(synth_methods, gaps, color=bar_colors, alpha=0.85, width=0.55)
    ax.set_ylabel("Average Certified Duality Gap Δ")
    ax.set_title(f"Certified Suboptimality Bound on Lattices (DPD Cert Rate: {dpd_cert_rate:.1f}%)")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for bar, gap in zip(ax.patches, gaps):
        yval = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + 0.02 * max(gaps),
            f"Δ={gap:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    fig4_path = os.path.join(figures_dir, "fig4_duality_gap.png")
    fig4_pdf = os.path.join(figures_dir, "fig4_duality_gap.pdf")
    plt.savefig(fig4_path, dpi=300, bbox_inches="tight")
    plt.savefig(fig4_pdf, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig4_path} and {fig4_pdf}")


if __name__ == "__main__":
    results_path = os.path.join("benchmarks", "results", "benchmark_results.json")
    figures_dir = os.path.join("paper", "figures")
    generate_all_plots(results_path, figures_dir)
