"""Automated verification of Lean 4 formal mathematical proofs.

Verifies:
1. Zero `sorry` declarations in formal proofs.
2. Zero unproven `axiom` declarations in formal proofs.
3. Clean compilation via Lean 4 compiler if installed.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


def test_lean_formal_proof_integrity():
    lean_file = Path("formal/DPDTheory.lean")
    assert lean_file.exists(), f"Lean proof file {lean_file} must exist."

    content = lean_file.read_text(encoding="utf-8")
    assert "sorry" not in content, "Formal proof must contain zero `sorry` declarations."
    assert "axiom" not in content, "Formal proof must contain zero `axiom` declarations."


def test_lean_compiler_verification():
    lean_exe = shutil.which("lean")
    if not lean_exe:
        pytest.skip("Lean 4 executable not found in PATH.")

    result = subprocess.run(
        [lean_exe, "formal/DPDTheory.lean"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"Lean compilation failed:\n{result.stderr}\n{result.stdout}"
