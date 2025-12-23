import torch
import numpy as np
import thunderkittens as tk
import matplotlib.pyplot as plt
from collections import defaultdict
import os
import time

# Optional: benchmark against FlashAttention-3 (FA3)
try:
    from flash_attn_interface import flash_attn_func
except ImportError as e:
    # Some environments load a stub libcudart.so.12 via torch deps which breaks FA3.
    if "cudaGetDriverEntryPointByVersion" in str(e):
        import importlib.util
        import sys
        from pathlib import Path

        mod_path = Path(__file__).resolve().parents[3] / "cudart_preload.py"
        spec = importlib.util.spec_from_file_location("cudart_preload", mod_path)
        if spec is None or spec.loader is None:
            raise
        cudart_preload = importlib.util.module_from_spec(spec)
        sys.modules["cudart_preload"] = cudart_preload
        spec.loader.exec_module(cudart_preload)
        cudart_preload.preload()
        from flash_attn_interface import flash_attn_func
    else:
        flash_attn_func = None

def flops(batch, seqlen, nheads, headdim, causal, mode="fwd"):
    assert mode in ["fwd", "bwd", "fwd_bwd"]
    f = 4 * batch * seqlen**2 * nheads * headdim // (2 if causal else 1)
    return f if mode == "fwd" else (2.5 * f if mode == "bwd" else 3.5 * f)

def efficiency(flop, time):
    flop = flop / 1e12
    time = time / 1e6
    return flop / time

def benchmark_attention(configurations):
    results = {
        'fwd_tk': defaultdict(list),
        'bwd_tk': defaultdict(list),
        'fwd_fa3': defaultdict(list),
        'bwd_fa3': defaultdict(list),
    }
    
    for B, H, N, D, causal in configurations:
        print("=" * 60)
        print(f"Timing forward and backward pass for B={B}, H={H}, N={N}, D={D}, causal={causal}")
        if N % 64 != 0:
            print(f"Skipping N={N} (ThunderKittens requires N multiple of 64).")
            continue

        q = torch.randn(B, H, N, D, dtype=torch.bfloat16, device='cuda', requires_grad=False).contiguous()
        k = torch.randn(B, H, N, D, dtype=torch.bfloat16, device='cuda', requires_grad=False).contiguous()
        v = torch.randn(B, H, N, D, dtype=torch.bfloat16, device='cuda', requires_grad=False).contiguous()

        # Use TK forward outputs for TK backward (required for a valid comparison).
        o_tk, l_vec_tk = tk.mha_forward(q, k, v, bool(causal))
        grad_output_tk = torch.randn_like(o_tk, requires_grad=False).contiguous()
        
        # Prepare for timing forward pass
        start_events_fwd = [torch.cuda.Event(enable_timing=True) for _ in range(10)]
        end_events_fwd = [torch.cuda.Event(enable_timing=True) for _ in range(10)]
        
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Warmup for forward pass
        for _ in range(10):
            tk.mha_forward(q, k, v, bool(causal))
            
        # Time the forward pass
        for i in range(10):          
            start_events_fwd[i].record()
            tk.mha_forward(q, k, v, bool(causal))
            end_events_fwd[i].record()

        torch.cuda.synchronize()
        times_fwd = [s.elapsed_time(e) for s, e in zip(start_events_fwd, end_events_fwd)]
        time_us_fwd = np.mean(times_fwd) * 1000

        tflops_fwd = efficiency(flops(B, N, H, D, causal, 'fwd'), time_us_fwd)
        results['fwd_tk'][(D, causal)].append((N, tflops_fwd))

        print(f"Average time for forward pass in us: {time_us_fwd:.2f}")
        print(f"TK forward TFLOPS: {tflops_fwd}")
        print("-" * 60)
        
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Prepare for timing backward pass
        start_events_bwd = [torch.cuda.Event(enable_timing=True) for _ in range(10)]
        end_events_bwd = [torch.cuda.Event(enable_timing=True) for _ in range(10)]
        
        # Warmup for backward pass
        for _ in range(10):
            tk.mha_backward(q, k, v, o_tk, l_vec_tk, grad_output_tk, bool(causal))
        
        # Time the backward pass
        for i in range(10):
            start_events_bwd[i].record()
            tk.mha_backward(q, k, v, o_tk, l_vec_tk, grad_output_tk, bool(causal))
            end_events_bwd[i].record()

        torch.cuda.synchronize()
        times_bwd = [s.elapsed_time(e) for s, e in zip(start_events_bwd, end_events_bwd)]
        time_us_bwd = np.mean(times_bwd) * 1000

        tflops_bwd = efficiency(flops(B, N, H, D, causal, 'bwd'), time_us_bwd)
        results['bwd_tk'][(D, causal)].append((N, tflops_bwd))

        print(f"Average time for backward pass in us: {time_us_bwd:.2f}")
        print(f"TK backward TFLOPS: {tflops_bwd}")

        # -------------------------
        # FA3 comparison (optional)
        # -------------------------
        if flash_attn_func is not None:
            # FA3 expects (B, S, H, D)
            q_bshd = q.permute(0, 2, 1, 3).contiguous()
            k_bshd = k.permute(0, 2, 1, 3).contiguous()
            v_bshd = v.permute(0, 2, 1, 3).contiguous()

            # Forward timing
            start_events_fwd_fa3 = [torch.cuda.Event(enable_timing=True) for _ in range(10)]
            end_events_fwd_fa3 = [torch.cuda.Event(enable_timing=True) for _ in range(10)]
            for _ in range(10):
                _out = flash_attn_func(q_bshd, k_bshd, v_bshd, causal=bool(causal))
                if not torch.is_tensor(_out):
                    _out = _out[0]
            for i in range(10):
                start_events_fwd_fa3[i].record()
                _out = flash_attn_func(q_bshd, k_bshd, v_bshd, causal=bool(causal))
                if not torch.is_tensor(_out):
                    _out = _out[0]
                end_events_fwd_fa3[i].record()
            torch.cuda.synchronize()
            times_fwd_fa3 = [s.elapsed_time(e) for s, e in zip(start_events_fwd_fa3, end_events_fwd_fa3)]
            time_us_fwd_fa3 = np.mean(times_fwd_fa3) * 1000
            tflops_fwd_fa3 = efficiency(flops(B, N, H, D, causal, 'fwd'), time_us_fwd_fa3)
            results['fwd_fa3'][(D, causal)].append((N, tflops_fwd_fa3))
            print(f"FA3 forward TFLOPS: {tflops_fwd_fa3}")

            # Backward timing (autograd)
            grad_output_fa3 = grad_output_tk.permute(0, 2, 1, 3).contiguous()
            start_events_bwd_fa3 = [torch.cuda.Event(enable_timing=True) for _ in range(10)]
            end_events_bwd_fa3 = [torch.cuda.Event(enable_timing=True) for _ in range(10)]
            for _ in range(10):
                q3 = q_bshd.detach().requires_grad_(True)
                k3 = k_bshd.detach().requires_grad_(True)
                v3 = v_bshd.detach().requires_grad_(True)
                out3 = flash_attn_func(q3, k3, v3, causal=bool(causal))
                if not torch.is_tensor(out3):
                    out3 = out3[0]
                out3.backward(grad_output_fa3)
            for i in range(10):
                q3 = q_bshd.detach().requires_grad_(True)
                k3 = k_bshd.detach().requires_grad_(True)
                v3 = v_bshd.detach().requires_grad_(True)
                start_events_bwd_fa3[i].record()
                out3 = flash_attn_func(q3, k3, v3, causal=bool(causal))
                if not torch.is_tensor(out3):
                    out3 = out3[0]
                out3.backward(grad_output_fa3)
                end_events_bwd_fa3[i].record()
            torch.cuda.synchronize()
            times_bwd_fa3 = [s.elapsed_time(e) for s, e in zip(start_events_bwd_fa3, end_events_bwd_fa3)]
            time_us_bwd_fa3 = np.mean(times_bwd_fa3) * 1000
            tflops_bwd_fa3 = efficiency(flops(B, N, H, D, causal, 'bwd'), time_us_bwd_fa3)
            results['bwd_fa3'][(D, causal)].append((N, tflops_bwd_fa3))
            print(f"FA3 backward TFLOPS: {tflops_bwd_fa3}")

        print("=" * 60)
        
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    return results

def plot_results(results):
    os.makedirs('benchmark_results', exist_ok=True)
    for mode in ['fwd', 'bwd']:
        tk_key = f"{mode}_tk"
        fa3_key = f"{mode}_fa3"
        for (D, causal), values in results[tk_key].items():
            values = sorted(values, key=lambda x: x[0])
            seq_lens = [x[0] for x in values]
            tk_tflops = [x[1] for x in values]
            fa3_values = sorted(results[fa3_key].get((D, causal), []), key=lambda x: x[0])
            fa3_map = {n: t for n, t in fa3_values}
            fa3_tflops = [fa3_map.get(n, None) for n in seq_lens]

            plt.figure(figsize=(10, 6))
            x = np.arange(len(seq_lens))
            width = 0.4
            bars_tk = plt.bar(x - width / 2, tk_tflops, width=width, label="ThunderKittens")
            if any(v is not None for v in fa3_tflops):
                fa3_vals_plot = [v if v is not None else 0.0 for v in fa3_tflops]
                bars_fa3 = plt.bar(x + width / 2, fa3_vals_plot, width=width, label="FA3")
            plt.xlabel('Sequence Length')
            plt.ylabel('TFLOPS')
            plt.title(f'{mode.upper()} Pass (TK vs FA3) - Head Dim: {D}, Causal: {causal}')
            plt.grid(True)
            plt.legend()

            # Adding the numerical y value on top of each bar
            for bar in bars_tk:
                yval = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), ha='center', va='bottom', fontsize=8)
            if any(v is not None for v in fa3_tflops):
                for i, v in enumerate(fa3_tflops):
                    if v is None:
                        continue
                    bar = plt.Rectangle((0, 0), 0, 0)  # dummy to satisfy lints
                    # annotate above FA3 bar position
                    plt.text(x[i] + width / 2, v, round(v, 2), ha='center', va='bottom', fontsize=8)

            filename = f'benchmark_results/{mode}_D{D}_causal{causal}.png'
            plt.savefig(filename)
            plt.close()

# Example list of configurations to test
configurations = [
    (16, 16, 768,    128, False),
    (16, 16, 768*16,  128, False),
    (16, 16, 768*2,  128, False),
    (16, 16, 768*4,  128, False),
    (16, 16, 768*8,  128, False),
    (16, 16, 768*16, 128, False),
    (16, 16, 768,    128, True),
    (16, 16, 768*2,  128, True),
    (16, 16, 768*4,  128, True),
    (16, 16, 768*8,  128, True),
    (16, 16, 768*16, 128, True),
    (16, 32, 768,    64,  False),
    (16, 32, 768*2,  64,  False),
    (16, 32, 768*4,  64,  False),
    (16, 32, 768*8,  64,  False),
    (16, 32, 768*16, 64,  False),
    (16, 32, 768,    64,  True),
    (16, 32, 768*2,  64,  True),
    (16, 32, 768*4,  64,  True),
    (16, 32, 768*8,  64,  True),
    (16, 32, 768*16, 64,  True),
]

results = benchmark_attention(configurations)
plot_results(results)
