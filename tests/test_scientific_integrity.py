"""Automated scientific integrity tests for Dual-Pruned Decoding (DPD).

Guards against:
1. Reintroduction of fake arXiv badges or placeholder publication identifiers.
2. Inadmissible default completion bounds (< 0.0).
3. Scope escalation (restricted search masquerading as global guarantee).
4. Discrepancy between raw benchmark JSON and generated tables.
5. Python version metadata regression.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dpd.bounds import BudgetAwareBound, BudgetConditionedBound
from dpd.certificates import compute_dual_certificate
from dpd.config import DPDConfig
from dpd.cost_models import TokenCountCostModel
from dpd.decoding import dual_pruned_decoding
from dpd.types import CertificateScope, PrefixCandidate, SuccessorCompleteness

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_no_fake_arxiv_badges_or_citations_in_readme():
    """Asserts that README.md and paper do not contain fabricated arXiv identifiers."""
    readme_path = REPO_ROOT / "README.md"
    assert readme_path.exists(), "README.md must exist"
    content = readme_path.read_text(encoding="utf-8")

    assert "arXiv-2610" not in content, "README must not contain fake arXiv badge"
    assert "2610.xxxxx" not in content, "README must not contain placeholder arXiv identifier"


def test_default_completion_bound_is_strictly_admissible():
    """Asserts that default completion bounds are admissible (max_log_prob_per_step == 0.0)."""
    cm = TokenCountCostModel(eos_token_id=0)

    bound_aware = BudgetAwareBound(cm)
    assert bound_aware.max_log_prob_per_step == 0.0, "BudgetAwareBound default must be 0.0"
    assert bound_aware.is_admissible is True

    bound_cond = BudgetConditionedBound(cm)
    assert bound_cond.max_log_prob_upper_bound == 0.0, "BudgetConditionedBound default must be 0.0"
    assert bound_cond.is_admissible is True

    # Attempting negative step bound without explicit opt-in must fail closed
    with pytest.raises(ValueError, match="is inadmissible"):
        BudgetAwareBound(cm, max_log_prob_per_step=-0.05)


def test_top_k_search_never_emits_global_certificate():
    """Asserts that default top-k search mode never claims global optimality."""
    cfg = DPDConfig(
        beam_width=5,
        max_steps=3,
        budget=5.0,
        eos_token_id=0,
    )
    # Default successor_completeness is RESTRICTED_TOP_K
    assert cfg.successor_completeness == SuccessorCompleteness.RESTRICTED_TOP_K

    cm = TokenCountCostModel(eos_token_id=0)

    def step_fn(prompt, prefix):
        if not prefix:
            return [(1, -0.1, "tok1"), (0, -0.5, "eos")]
        return [(0, 0.0, "eos")]

    res = dual_pruned_decoding("test", cfg, cm, step_fn)
    assert res.certificate is not None
    assert res.certificate.is_global is False
    assert res.certificate.certified_globally_optimal is False
    assert res.certificate.certificate_scope in (
        CertificateScope.RESTRICTED_SUBSET,
        CertificateScope.RESTRICTED_LATTICE,
    )


def test_certificate_sanitizer_prevents_scope_escalation():
    """Asserts that conflicting arguments (scope=RESTRICTED_SUBSET, is_global=True) are sanitized."""
    cand = PrefixCandidate(
        step=1, token_ids=[1, 0], accumulated_score=-1.0, accumulated_cost=1.0, is_terminal=True
    )
    cert = compute_dual_certificate(
        all_terminal_candidates=[cand],
        budget=5.0,
        incumbent=cand,
        scope=CertificateScope.RESTRICTED_SUBSET,
        is_global=True,  # Conflicting argument
    )
    assert cert.is_global is False
    assert cert.certified_globally_optimal is False
    assert cert.certificate_scope == CertificateScope.RESTRICTED_SUBSET


def test_synthetic_latency_parity_between_json_and_latex_tables():
    """Asserts that table2_synthetic_results.tex faithfully reflects raw JSON benchmark latency."""
    json_path = REPO_ROOT / "benchmarks" / "results" / "benchmark_results.json"
    table_path = REPO_ROOT / "paper" / "tables" / "table2_synthetic_results.tex"

    assert json_path.exists(), "benchmark_results.json must exist"
    assert table_path.exists(), "table2_synthetic_results.tex must exist"

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    synth_data = data["task_3_synthetic"]
    dpd_latency_ms = synth_data["DPD (Hybrid)"]["average_latency_sec"] * 1000.0
    beam_latency_ms = synth_data["Standard Beam"]["average_latency_sec"] * 1000.0

    table_text = table_path.read_text(encoding="utf-8")

    # DPD hybrid latency in ms formatted to 2 decimals
    expected_dpd_str = f"{dpd_latency_ms:.2f}"
    expected_beam_str = f"{beam_latency_ms:.2f}"

    assert expected_dpd_str in table_text, (
        f"table2_synthetic_results.tex must contain DPD latency {expected_dpd_str} ms"
    )
    assert expected_beam_str in table_text, (
        f"table2_synthetic_results.tex must contain Standard Beam latency {expected_beam_str} ms"
    )


def test_python_compatibility_metadata():
    """Asserts that pyproject.toml requires Python >=3.10 and targets py310."""
    pyproject_path = REPO_ROOT / "pyproject.toml"
    assert pyproject_path.exists()
    content = pyproject_path.read_text(encoding="utf-8")

    assert 'requires-python = ">=3.10"' in content
    assert 'target-version = "py310"' in content
    assert "Programming Language :: Python :: 3.10" in content
    assert "Programming Language :: Python :: 3.11" in content
    assert "Programming Language :: Python :: 3.12" in content
