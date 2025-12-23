"""
Work around CUDA runtime (libcudart) loader conflicts in some Python envs.

In this workspace, torch (cu13) can load a stub `libcudart.so.12` from
`site-packages/nvidia/cuda_runtime/` that does not export newer symbols like
`cudaGetDriverEntryPointByVersion`. FlashAttention 3's extension may then fail
to import with:

  undefined symbol: cudaGetDriverEntryPointByVersion, version libcudart.so.12

Preloading a "real" libcudart (with RTLD_GLOBAL) before importing FA3 fixes it.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Iterable, Optional


def _default_candidates() -> list[str]:
    candidates: list[str] = []

    # User override.
    env_path = os.getenv("TK_CUDART_PATH")
    if env_path:
        candidates.append(env_path)

    # Common locations in this environment.
    candidates.extend(
        [
            "/root/miniconda3/lib/libcudart.so.12",
            "/usr/local/cuda-12.8/lib64/libcudart.so.12",
            "/usr/local/cuda/lib64/libcudart.so.12",
        ]
    )

    # Also try to find a libcudart shipped alongside torch / nvidia wheels.
    try:
        import torch  # noqa: F401

        torch_lib = Path(torch.__file__).resolve().parent / "lib"
        candidates.append(str(torch_lib / "libcudart.so.12"))
        candidates.append(str(torch_lib / "libcudart.so.13"))
    except Exception:
        pass

    # Deduplicate while preserving order.
    seen: set[str] = set()
    out: list[str] = []
    for p in candidates:
        if p and p not in seen:
            out.append(p)
            seen.add(p)
    return out


_PRELOADED = False


def preload(candidates: Optional[Iterable[str]] = None) -> Optional[str]:
    """
    Preload a CUDA runtime library into the global namespace.

    Returns the path that was successfully loaded, or None if nothing worked.
    Safe to call multiple times.
    """
    global _PRELOADED
    if _PRELOADED:
        return None

    cand_list = list(candidates) if candidates is not None else _default_candidates()
    for p in cand_list:
        try:
            if not Path(p).exists():
                continue
            ctypes.CDLL(p, mode=ctypes.RTLD_GLOBAL)
            _PRELOADED = True
            return p
        except OSError:
            continue

    return None


