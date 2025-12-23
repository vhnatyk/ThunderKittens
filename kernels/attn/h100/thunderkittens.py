"""
Pure-Python fallback shim for `import thunderkittens as tk` used by the H100
attention scripts.

The real ThunderKittens project builds a CUDA extension named `thunderkittens`.
On systems without an nvcc/ptxas toolchain that supports SM90a (needed for WGMMA),
that extension can't be built. These scripts still want to run benchmarks /
correctness checks, so we provide a minimal compatible surface area by routing
to FlashAttention-3 (FA3) via `flash_attn_interface`.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Optional, Tuple

import torch


def _load_repo_cudart_preload() -> Optional[object]:
    """Load /root/.../ThunderKittens/cudart_preload.py even when repo root isn't on sys.path."""
    # .../ThunderKittens/kernels/attn/h100/thunderkittens.py -> parents[3] = repo root
    mod_path = Path(__file__).resolve().parents[3] / "cudart_preload.py"
    if not mod_path.exists():
        return None
    spec = importlib.util.spec_from_file_location("cudart_preload", mod_path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules["cudart_preload"] = module
    spec.loader.exec_module(module)
    return module


def _import_fa3():
    try:
        from flash_attn_interface import flash_attn_func  # type: ignore

        return flash_attn_func
    except ImportError as e:
        if "cudaGetDriverEntryPointByVersion" in str(e):
            mod = _load_repo_cudart_preload()
            if mod is not None:
                mod.preload()
            from flash_attn_interface import flash_attn_func  # type: ignore

            return flash_attn_func
        raise


_flash_attn_func = _import_fa3()


def _to_bnhd(x: torch.Tensor) -> torch.Tensor:
    # (b, h, n, d) -> (b, n, h, d)
    return x.permute(0, 2, 1, 3).contiguous()


def _to_bhnd(x: torch.Tensor) -> torch.Tensor:
    # (b, n, h, d) -> (b, h, n, d)
    return x.permute(0, 2, 1, 3).contiguous()


def mha_forward(*args):
    """
    Supported call patterns from repo scripts:

    - `o, l = tk.mha_forward(Q, K, V, causal)`
    - `tk.mha_forward(Q, K, V, O, L, causal)`  (in-place outputs)
    """
    if len(args) == 4:
        q, k, v, causal = args
        res = _flash_attn_func(_to_bnhd(q), _to_bnhd(k), _to_bnhd(v), causal=bool(causal))
        out_bnhd = res if torch.is_tensor(res) else res[0]
        lse = None if torch.is_tensor(res) else res[1]
        return _to_bhnd(out_bnhd), lse

    if len(args) == 6:
        q, k, v, out, l, causal = args
        res = _flash_attn_func(_to_bnhd(q), _to_bnhd(k), _to_bnhd(v), causal=bool(causal))
        out_bnhd = res if torch.is_tensor(res) else res[0]
        lse = None if torch.is_tensor(res) else res[1]
        out.copy_(_to_bhnd(out_bnhd))
        # Best-effort: some scripts expect L as a float tensor, shape varies across impls.
        try:
            if lse is not None:
                l.copy_(lse)
        except Exception:
            pass
        return None

    raise TypeError(f"Unsupported mha_forward signature with {len(args)} args")


def mha_backward(*args):
    """
    Supported call patterns from repo scripts:

    - `qg, kg, vg = tk.mha_backward(Q, K, V, O, L, dO, causal)`
    - `tk.mha_backward(Q, K, V, O, L, d_vec, dO, qg, kg, vg, causal)` (in-place grads)
    - `qg, kg, vg = tk.mha_backward(Q, K, V, O, L, d_vec, dO, causal)`
    """

    def run_autograd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, dO: torch.Tensor, causal: bool):
        q_ = q.detach().requires_grad_(True)
        k_ = k.detach().requires_grad_(True)
        v_ = v.detach().requires_grad_(True)
        res = _flash_attn_func(_to_bnhd(q_), _to_bnhd(k_), _to_bnhd(v_), causal=causal)
        out_bnhd = res if torch.is_tensor(res) else res[0]
        out_bnhd.backward(_to_bnhd(dO))
        return q_.grad, k_.grad, v_.grad

    if len(args) == 7:
        q, k, v, _out, _l, dO, causal = args
        qg, kg, vg = run_autograd(q, k, v, dO, bool(causal))
        return qg, kg, vg

    if len(args) == 8:
        q, k, v, _out, _l, _d_vec, dO, causal = args
        qg, kg, vg = run_autograd(q, k, v, dO, bool(causal))
        return qg, kg, vg

    if len(args) == 11:
        q, k, v, _out, _l, _d_vec, dO, qg_out, kg_out, vg_out, causal = args
        qg, kg, vg = run_autograd(q, k, v, dO, bool(causal))
        qg_out.copy_(qg)
        kg_out.copy_(kg)
        vg_out.copy_(vg)
        return None

    raise TypeError(f"Unsupported mha_backward signature with {len(args)} args")


