"""Build and install the native Rust pruning kernel into the dpd package."""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


def build_and_install_kernel() -> None:
    """Builds dpd_kernel in release mode and copies the shared library to dpd/."""
    root = Path(__file__).resolve().parent.parent
    manifest = root / "dpd_kernel" / "Cargo.toml"
    print(f"Building Rust kernel via {manifest}...")
    subprocess.check_call(["cargo", "build", "--release", "--manifest-path", str(manifest)])

    target_dir = root / "dpd_kernel" / "target" / "release"
    ext = ".dll" if sys.platform == "win32" else (".dylib" if sys.platform == "darwin" else ".so")
    dpd_dir = root / "dpd"

    copied = 0
    for file_path in target_dir.glob(f"*{ext}"):
        if "dpd_kernel" in file_path.name:
            dest = dpd_dir / f"dpd_kernel{ext}"
            shutil.copy2(file_path, dest)
            print(f"Successfully copied {file_path.name} to {dest}")
            copied += 1

    if copied == 0:
        print(f"Warning: No matching shared library found in {target_dir} with extension {ext}")


if __name__ == "__main__":
    build_and_install_kernel()
