"""Rust FFI acceleration loader for Dual-Pruned Decoding (DPD).

Safely loads the compiled native Rust pruning kernel `dpd_kernel` via ctypes,
with seamless fallback to the pure-Python reference implementation.
"""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path

from dpd.types import CostScorePoint

_LIB: ctypes.CDLL | None = None
_LOAD_ATTEMPTED: bool = False


def get_rust_kernel() -> ctypes.CDLL | None:
    """Attempts to load the compiled Rust pruning kernel.
    
    Returns:
        ctypes.CDLL instance if available and valid, None otherwise.
    """
    global _LIB, _LOAD_ATTEMPTED
    if _LOAD_ATTEMPTED:
        return _LIB

    _LOAD_ATTEMPTED = True
    pkg_dir = Path(__file__).parent
    repo_root = pkg_dir.parent
    ext = ".dll" if sys.platform == "win32" else (".dylib" if sys.platform == "darwin" else ".so")

    candidate_paths = [
        pkg_dir / f"dpd_kernel{ext}",
        pkg_dir / f"libdpd_kernel{ext}",
        repo_root / "dpd_kernel" / "target" / "release" / f"libdpd_kernel{ext}",
        repo_root / "dpd_kernel" / "target" / "release" / f"dpd_kernel{ext}",
    ]

    for dll_path in candidate_paths:
        if dll_path.exists():
            try:
                lib = ctypes.CDLL(str(dll_path))
                lib.rust_compute_upper_left_hull.argtypes = [
                    ctypes.POINTER(ctypes.c_double),
                    ctypes.POINTER(ctypes.c_double),
                    ctypes.c_size_t,
                    ctypes.POINTER(ctypes.c_size_t),
                    ctypes.c_size_t,
                    ctypes.POINTER(ctypes.c_size_t),
                ]
                lib.rust_compute_upper_left_hull.restype = ctypes.c_int32
                _LIB = lib
                break
            except (OSError, AttributeError):
                continue

    return _LIB


def is_rust_kernel_available() -> bool:
    """Checks whether the compiled native Rust pruning kernel is active."""
    return get_rust_kernel() is not None


def compute_upper_left_hull_rust(points: list[CostScorePoint]) -> list[CostScorePoint] | None:
    """Computes upper-left hull using the native Rust kernel.
    
    Returns:
        List of CostScorePoint vertices if native kernel succeeded, None if unavailable.
    """
    lib = get_rust_kernel()
    if lib is None or not points:
        return None

    count = len(points)
    c_arr = (ctypes.c_double * count)(*(p.cost for p in points))
    s_arr = (ctypes.c_double * count)(*(p.score for p in points))
    out_indices = (ctypes.c_size_t * count)()
    out_count = ctypes.c_size_t(0)

    res = lib.rust_compute_upper_left_hull(
        c_arr,
        s_arr,
        count,
        out_indices,
        count,
        ctypes.byref(out_count),
    )

    if res != 0:
        return None

    n_out = out_count.value
    # Map returned indices back to points
    return [points[out_indices[i]] for i in range(n_out)]
