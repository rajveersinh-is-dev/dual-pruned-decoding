"""Master benchmark runner executing real, unadulterated experiments across all tasks and baselines."""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.abspath("."))
from typing import Any

# Import tasks
from benchmarks.tasks.code_budget_task import StructuredCodeCostModel, StructuredCodeTask
from benchmarks.tasks.lm_budget_task import LMGenerationEnvironment, get_standard_evaluation_prompts
from benchmarks.tasks.synthetic_lattice_task import (
    LatticeCostModel,
    generate_benchmark_lattice_suite,
)
from dpd.baselines import (
    exact_exhaustive_search,
    greedy_decoding,
    length_penalty_beam_search,
    standard_beam_search,
)
from dpd.bounds import BudgetAwareBound, TrivialAdmissibleBound
from dpd.config import DPDConfig, HullConfig, MultiplierConfig
from dpd.decoding import dual_pruned_decoding
from dpd.metrics import EvaluationSummary, compute_evaluation_summary
from dpd.types import MultiplierType, PruningMode


def run_synthetic_lattice_benchmark() -> dict[str, Any]:
    """Task 3: Runs synthetic lattice benchmark on 15 instances with exact ground truth."""
    print("--- Running Task 3: Synthetic Lattice Benchmark ---")
    suite = generate_benchmark_lattice_suite(num_instances=15, seed=42)

    results_by_method: dict[str, list[Any]] = {
        "DPD (Hybrid)": [],
        "Standard Beam": [],
        "Length Penalty": [],
        "DPD (No Hull)": [],
        "DPD (Fixed Lambda)": [],
    }

    ground_truth_optimal_scores: list[float] = []

    for inst in suite:
        cost_model = LatticeCostModel(inst)

        # 1. Exact ground truth
        cfg_opt = DPDConfig(budget=inst.budget, max_steps=inst.max_steps, eos_token_id=inst.eos_token_id)
        true_opt, _ = exact_exhaustive_search(
            prompt="synth",
            config=cfg_opt,
            cost_model=cost_model,
            step_fn=lambda p, pref, inst=inst: inst.transitions.get(tuple(pref), [(inst.eos_token_id, 0.0, "eos")]),
        )
        if true_opt is not None:
            ground_truth_optimal_scores.append(true_opt.accumulated_score)

        step_fn = lambda p, pref, inst=inst: inst.transitions.get(tuple(pref), [(inst.eos_token_id, 0.0, "eos")])

        # A. DPD Hybrid
        cfg_dpd = DPDConfig(
            beam_width=4,
            max_steps=inst.max_steps,
            budget=inst.budget,
            eos_token_id=inst.eos_token_id,
            pruning_mode=PruningMode.HYBRID_HULL_AND_BOUND,
            multiplier=MultiplierConfig(controller_type=MultiplierType.PROJECTED_SUBGRADIENT),
        )
        res_dpd = dual_pruned_decoding("synth", cfg_dpd, cost_model, step_fn)
        results_by_method["DPD (Hybrid)"].append(res_dpd)

        # B. Standard Beam
        cfg_std = DPDConfig(beam_width=4, max_steps=inst.max_steps, budget=inst.budget, eos_token_id=inst.eos_token_id)
        res_std = standard_beam_search("synth", cfg_std, cost_model, step_fn)
        results_by_method["Standard Beam"].append(res_std)

        # C. Length Penalty Beam
        res_lp = length_penalty_beam_search("synth", cfg_std, cost_model, step_fn, alpha=0.6)
        results_by_method["Length Penalty"].append(res_lp)

        # D. DPD (No Hull)
        cfg_nohull = DPDConfig(
            beam_width=4,
            max_steps=inst.max_steps,
            budget=inst.budget,
            eos_token_id=inst.eos_token_id,
            pruning_mode=PruningMode.EXACT_BOUND_AND_DOMINANCE,
            multiplier=MultiplierConfig(controller_type=MultiplierType.PROJECTED_SUBGRADIENT),
        )
        res_nohull = dual_pruned_decoding("synth", cfg_nohull, cost_model, step_fn)
        results_by_method["DPD (No Hull)"].append(res_nohull)

        # E. DPD (Fixed Lambda)
        cfg_fixed = DPDConfig(
            beam_width=4,
            max_steps=inst.max_steps,
            budget=inst.budget,
            eos_token_id=inst.eos_token_id,
            pruning_mode=PruningMode.HYBRID_HULL_AND_BOUND,
            multiplier=MultiplierConfig(controller_type=MultiplierType.FIXED, initial_lambda=0.8),
        )
        res_fixed = dual_pruned_decoding("synth", cfg_fixed, cost_model, step_fn)
        results_by_method["DPD (Fixed Lambda)"].append(res_fixed)

    summary: dict[str, Any] = {}
    for method, res_list in results_by_method.items():
        summary[method] = compute_evaluation_summary(res_list)

    return {
        "summary": summary,
        "ground_truth_count": len(ground_truth_optimal_scores),
    }


def run_structured_code_benchmark() -> dict[str, Any]:
    """Task 2: Runs structured code generation across multiple budgets."""
    print("--- Running Task 2: Structured Code Benchmark ---")
    task = StructuredCodeTask()
    cost_model = StructuredCodeCostModel(task)

    budgets = [6.0, 8.0, 10.0, 12.0]
    results_by_budget: dict[float, dict[str, EvaluationSummary]] = {}

    for b in budgets:
        results_by_method: dict[str, list[Any]] = {
            "DPD (Hybrid)": [],
            "Standard Beam": [],
            "Length Penalty": [],
            "Greedy": [],
            "DPD (No Hull)": [],
            "DPD (Fixed Lambda)": [],
        }

        # 10 trial evaluations with different random warmups / initial configurations
        for trial in range(10):
            # A. DPD Hybrid
            cfg_dpd = DPDConfig(
                beam_width=4,
                max_steps=12,
                budget=b,
                eos_token_id=task.eos_token_id,
                pruning_mode=PruningMode.HYBRID_HULL_AND_BOUND,
                multiplier=MultiplierConfig(controller_type=MultiplierType.PROJECTED_SUBGRADIENT, initial_lambda=0.5),
            )
            res_dpd = dual_pruned_decoding("code", cfg_dpd, cost_model, task.step_fn)
            results_by_method["DPD (Hybrid)"].append(res_dpd)

            # B. Standard Beam
            cfg_std = DPDConfig(beam_width=4, max_steps=12, budget=b, eos_token_id=task.eos_token_id)
            res_std = standard_beam_search("code", cfg_std, cost_model, task.step_fn)
            results_by_method["Standard Beam"].append(res_std)

            # C. Length Penalty
            res_lp = length_penalty_beam_search("code", cfg_std, cost_model, task.step_fn, alpha=0.6)
            results_by_method["Length Penalty"].append(res_lp)

            # D. Greedy
            res_gr = greedy_decoding("code", cfg_std, cost_model, task.step_fn)
            results_by_method["Greedy"].append(res_gr)

            # E. DPD (No Hull)
            cfg_nohull = DPDConfig(
                beam_width=4,
                max_steps=12,
                budget=b,
                eos_token_id=task.eos_token_id,
                pruning_mode=PruningMode.EXACT_BOUND_AND_DOMINANCE,
                multiplier=MultiplierConfig(controller_type=MultiplierType.PROJECTED_SUBGRADIENT),
            )
            res_nohull = dual_pruned_decoding("code", cfg_nohull, cost_model, task.step_fn)
            results_by_method["DPD (No Hull)"].append(res_nohull)

            # F. DPD (Fixed Lambda)
            cfg_fixed = DPDConfig(
                beam_width=4,
                max_steps=12,
                budget=b,
                eos_token_id=task.eos_token_id,
                pruning_mode=PruningMode.HYBRID_HULL_AND_BOUND,
                multiplier=MultiplierConfig(controller_type=MultiplierType.FIXED, initial_lambda=1.0),
            )
            res_fixed = dual_pruned_decoding("code", cfg_fixed, cost_model, task.step_fn)
            results_by_method["DPD (Fixed Lambda)"].append(res_fixed)

        budget_summary: dict[str, EvaluationSummary] = {}
        for method, res_list in results_by_method.items():
            budget_summary[method] = compute_evaluation_summary(res_list)
        results_by_budget[b] = budget_summary

    return {"results_by_budget": results_by_budget}


def run_lm_generation_benchmark() -> dict[str, Any]:
    """Task 1: Runs real DistilGPT-2 autoregressive generation on standard prompts."""
    print("--- Running Task 1: Real LM (DistilGPT-2) Benchmark ---")
    lm_env = LMGenerationEnvironment(model_name="distilbert/distilgpt2", branch_top_k=8, device="cpu")
    prompts = get_standard_evaluation_prompts()

    budgets = [10.0, 15.0, 20.0]
    results_by_budget: dict[float, dict[str, EvaluationSummary]] = {}

    for b in budgets:
        print(f"  Evaluating Budget B = {b} tokens...")
        results_by_method: dict[str, list[Any]] = {
            "DPD (Hybrid)": [],
            "Standard Beam": [],
            "Length Penalty": [],
            "Greedy": [],
            "DPD (No Hull)": [],
            "DPD (Fixed Lambda)": [],
        }

        for prompt_idx, prompt_text in enumerate(prompts):
            prompt_tokens = lm_env.tokenizer.encode(prompt_text)
            step_fn = lm_env.create_step_fn(prompt_tokens)

            # A. DPD Hybrid
            cfg_dpd = DPDConfig(
                beam_width=4,
                max_steps=int(b) + 5,
                budget=b,
                eos_token_id=lm_env.eos_token_id,
                pruning_mode=PruningMode.HYBRID_HULL_AND_BOUND,
                multiplier=MultiplierConfig(controller_type=MultiplierType.PROJECTED_SUBGRADIENT, initial_lambda=0.5),
            )
            res_dpd = dual_pruned_decoding(prompt_text, cfg_dpd, lm_env.cost_model, step_fn)
            results_by_method["DPD (Hybrid)"].append(res_dpd)

            # B. Standard Beam
            cfg_std = DPDConfig(
                beam_width=4,
                max_steps=int(b) + 5,
                budget=b,
                eos_token_id=lm_env.eos_token_id,
            )
            res_std = standard_beam_search(prompt_text, cfg_std, lm_env.cost_model, step_fn)
            results_by_method["Standard Beam"].append(res_std)

            # C. Length Penalty
            res_lp = length_penalty_beam_search(prompt_text, cfg_std, lm_env.cost_model, step_fn, alpha=0.6)
            results_by_method["Length Penalty"].append(res_lp)

            # D. Greedy
            res_gr = greedy_decoding(prompt_text, cfg_std, lm_env.cost_model, step_fn)
            results_by_method["Greedy"].append(res_gr)

            # E. DPD (No Hull)
            cfg_nohull = DPDConfig(
                beam_width=4,
                max_steps=int(b) + 5,
                budget=b,
                eos_token_id=lm_env.eos_token_id,
                pruning_mode=PruningMode.EXACT_BOUND_AND_DOMINANCE,
                multiplier=MultiplierConfig(controller_type=MultiplierType.PROJECTED_SUBGRADIENT),
            )
            res_nohull = dual_pruned_decoding(prompt_text, cfg_nohull, lm_env.cost_model, step_fn)
            results_by_method["DPD (No Hull)"].append(res_nohull)

            # F. DPD (Fixed Lambda)
            cfg_fixed = DPDConfig(
                beam_width=4,
                max_steps=int(b) + 5,
                budget=b,
                eos_token_id=lm_env.eos_token_id,
                pruning_mode=PruningMode.HYBRID_HULL_AND_BOUND,
                multiplier=MultiplierConfig(controller_type=MultiplierType.FIXED, initial_lambda=0.8),
            )
            res_fixed = dual_pruned_decoding(prompt_text, cfg_fixed, lm_env.cost_model, step_fn)
            results_by_method["DPD (Fixed Lambda)"].append(res_fixed)

        budget_summary: dict[str, EvaluationSummary] = {}
        for method, res_list in results_by_method.items():
            budget_summary[method] = compute_evaluation_summary(res_list)
        results_by_budget[b] = budget_summary

    return {"results_by_budget": results_by_budget}


def run_ablation_study() -> dict[str, Any]:
    """Runs systematic ablation experiments on non-convex synthetic lattices."""
    print("--- Running Ablation Study (Beam Width, Multiplier, Pruning Mode, Rust Kernel) ---")
    suite = generate_benchmark_lattice_suite(num_instances=15, seed=42)

    # 1. Beam Width Ablation (K in 2, 4, 8)
    beam_results: dict[str, Any] = {}
    for k in [2, 4, 8]:
        res_list = []
        for inst in suite:
            cost_model = LatticeCostModel(inst)
            def step_fn(p, pref, inst=inst):
                return inst.transitions.get(tuple(pref), [(inst.eos_token_id, 0.0, "eos")])
            cfg = DPDConfig(
                beam_width=k,
                max_steps=inst.max_steps,
                budget=inst.budget,
                eos_token_id=inst.eos_token_id,
                pruning_mode=PruningMode.HYBRID_HULL_AND_BOUND,
                multiplier=MultiplierConfig(controller_type=MultiplierType.PROJECTED_SUBGRADIENT),
            )
            res = dual_pruned_decoding("synth", cfg, cost_model, step_fn)
            res_list.append(res)
        beam_results[f"K={k}"] = compute_evaluation_summary(res_list)

    # 2. Multiplier Update Rule Ablation
    mult_results: dict[str, Any] = {}
    for m_type, m_name in [
        (MultiplierType.PROJECTED_SUBGRADIENT, "Projected Subgradient"),
        (MultiplierType.PROPORTIONAL_INTEGRAL, "Proportional-Integral (PI)"),
        (MultiplierType.EXPONENTIAL_MOVING_AVERAGE, "Exponential Moving Average (EMA)"),
        (MultiplierType.FIXED, "Fixed Multiplier (lambda=0.8)"),
    ]:
        res_list = []
        for inst in suite:
            cost_model = LatticeCostModel(inst)
            def step_fn(p, pref, inst=inst):
                return inst.transitions.get(tuple(pref), [(inst.eos_token_id, 0.0, "eos")])
            cfg = DPDConfig(
                beam_width=4,
                max_steps=inst.max_steps,
                budget=inst.budget,
                eos_token_id=inst.eos_token_id,
                pruning_mode=PruningMode.HYBRID_HULL_AND_BOUND,
                multiplier=MultiplierConfig(controller_type=m_type, initial_lambda=0.8),
            )
            res = dual_pruned_decoding("synth", cfg, cost_model, step_fn)
            res_list.append(res)
        mult_results[m_name] = compute_evaluation_summary(res_list)

    # 3. Pruning Mode Ablation
    prune_results: dict[str, Any] = {}
    for p_mode, p_name in [
        (PruningMode.HYBRID_HULL_AND_BOUND, "Hybrid Hull-and-Bound"),
        (PruningMode.EXACT_BOUND_AND_DOMINANCE, "Exact Bound & Dominance"),
        (PruningMode.HULL_EXTREMES_ONLY, "Static Hull Only"),
        (PruningMode.NO_PRUNING, "No Pruning (Standard Beam)"),
    ]:
        res_list = []
        for inst in suite:
            cost_model = LatticeCostModel(inst)
            def step_fn(p, pref, inst=inst):
                return inst.transitions.get(tuple(pref), [(inst.eos_token_id, 0.0, "eos")])
            cfg = DPDConfig(
                beam_width=4,
                max_steps=inst.max_steps,
                budget=inst.budget,
                eos_token_id=inst.eos_token_id,
                pruning_mode=p_mode,
                multiplier=MultiplierConfig(controller_type=MultiplierType.PROJECTED_SUBGRADIENT),
            )
            res = dual_pruned_decoding("synth", cfg, cost_model, step_fn)
            res_list.append(res)
        prune_results[p_name] = compute_evaluation_summary(res_list)

    # 4. Native Rust Kernel Acceleration
    kernel_results: dict[str, Any] = {}
    for use_rust, k_name in [(True, "Native Rust Kernel"), (False, "Pure Python Reference")]:
        res_list = []
        for inst in suite:
            cost_model = LatticeCostModel(inst)
            def step_fn(p, pref, inst=inst):
                return inst.transitions.get(tuple(pref), [(inst.eos_token_id, 0.0, "eos")])
            cfg = DPDConfig(
                beam_width=4,
                max_steps=inst.max_steps,
                budget=inst.budget,
                eos_token_id=inst.eos_token_id,
                pruning_mode=PruningMode.HYBRID_HULL_AND_BOUND,
                hull=HullConfig(use_rust_kernel=use_rust),
            )
            res = dual_pruned_decoding("synth", cfg, cost_model, step_fn)
            res_list.append(res)
        kernel_results[k_name] = compute_evaluation_summary(res_list)

    # 5. Admissible Completion Bound Ablation
    bound_results: dict[str, Any] = {}
    for bound_type, b_name in [
        ("trivial", "Trivial Bound (U=0)"),
        ("aware", "Budget-Aware Bound (alpha=-0.05)"),
    ]:
        res_list = []
        for inst in suite:
            cost_model = LatticeCostModel(inst)
            def step_fn(p, pref, inst=inst):
                return inst.transitions.get(tuple(pref), [(inst.eos_token_id, 0.0, "eos")])
            b_mgr = BudgetAwareBound(cost_model, max_log_prob_per_step=-0.05) if bound_type == "aware" else TrivialAdmissibleBound()
            cfg = DPDConfig(
                beam_width=4,
                max_steps=inst.max_steps,
                budget=inst.budget,
                eos_token_id=inst.eos_token_id,
                pruning_mode=PruningMode.HYBRID_HULL_AND_BOUND,
            )
            res = dual_pruned_decoding("synth", cfg, cost_model, step_fn, bound_manager=b_mgr)
            res_list.append(res)
        bound_results[b_name] = compute_evaluation_summary(res_list)

    return {
        "beam_width": beam_results,
        "multiplier_rules": mult_results,
        "pruning_modes": prune_results,
        "kernel_acceleration": kernel_results,
        "bound_admissibility": bound_results,
    }


def main():
    print("================================================================")
    print("Starting Comprehensive Dual-Pruned Decoding (DPD) Benchmark Suite")
    print("================================================================")

    out_dir = os.path.join("benchmarks", "results")
    logs_dir = os.path.join("experiments", "logs")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    results_file = os.path.join(out_dir, "benchmark_results.json")
    existing_data: dict[str, Any] = {}
    if os.path.exists(results_file):
        try:
            with open(results_file) as f:
                existing_data = json.load(f)
        except Exception as e:
            print(f"Warning: could not load existing {results_file}: {e}")

    rerun_all = "--rerun-all" in sys.argv
    rerun_lm = "--rerun-lm" in sys.argv
    ablations_only = "--ablations-only" in sys.argv

    # Helper to serialize dataclass summaries
    def serialize_summary(s: EvaluationSummary) -> dict[str, Any]:
        return {
            "num_instances": s.num_instances,
            "feasible_rate": s.feasible_rate,
            "violation_rate": s.violation_rate,
            "average_score": s.average_score,
            "average_cost": s.average_cost,
            "average_tokens": s.average_tokens,
            "average_latency_sec": s.average_latency_sec,
            "median_latency_sec": s.median_latency_sec,
            "average_expansions": s.average_expansions,
            "average_pruned_candidates": s.average_pruned_candidates,
            "average_dual_gap": s.average_dual_gap,
            "certified_optimal_rate": s.certified_optimal_rate,
        }

    # 1. Run Task 3 (Synthetic Lattice)
    if not ablations_only or "task_3_synthetic" not in existing_data:
        synth_res = run_synthetic_lattice_benchmark()
        task_3_data = {
            method: serialize_summary(summary)
            for method, summary in synth_res["summary"].items()
        }
    else:
        print("Reusing existing Task 3 results from benchmark_results.json")
        task_3_data = existing_data["task_3_synthetic"]

    # 2. Run Task 2 (Structured Code)
    if not ablations_only or "task_2_code" not in existing_data:
        code_res = run_structured_code_benchmark()
        task_2_data = {
            str(b): {
                method: serialize_summary(summary)
                for method, summary in b_summary.items()
            }
            for b, b_summary in code_res["results_by_budget"].items()
        }
    else:
        print("Reusing existing Task 2 results from benchmark_results.json")
        task_2_data = existing_data["task_2_code"]

    # 3. Run Task 1 (Real LM)
    if (rerun_all or rerun_lm or "task_1_lm" not in existing_data) and not ablations_only:
        lm_res = run_lm_generation_benchmark()
        task_1_data = {
            str(b): {
                method: serialize_summary(summary)
                for method, summary in b_summary.items()
            }
            for b, b_summary in lm_res["results_by_budget"].items()
        }
    else:
        print("Reusing existing Task 1 LM results from benchmark_results.json")
        task_1_data = existing_data["task_1_lm"]

    # 4. Run Ablation Study
    ablations_res = run_ablation_study()
    ablations_data = {
        category: {
            name: serialize_summary(summary)
            for name, summary in cat_dict.items()
        }
        for category, cat_dict in ablations_res.items()
    }

    # Save JSON results
    all_results = {
        "task_3_synthetic": task_3_data,
        "task_2_code": task_2_data,
        "task_1_lm": task_1_data,
        "ablations": ablations_data,
    }

    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll benchmark results successfully saved to {results_file}")

    print("\n================================================================")
    print("Benchmark Suite Completed Successfully.")
    print("================================================================")


if __name__ == "__main__":
    main()
