import torch
import thunderkittens as tk
import numpy as np
from tqdm import tqdm

try:
    from flash_attn_interface import flash_attn_func
except ImportError as e:
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
        raise

def generate_tensor(shape, mean, std, dtype, device):
    
    tensor = torch.randn(shape, dtype=dtype, device=device)
    magnitude = torch.norm(tensor, dim=-1, keepdim=True)
    scaled_tensor = tensor * (torch.randn(magnitude.shape, dtype=dtype, device=device) * std + mean) / magnitude
    
    return scaled_tensor.contiguous()


def tk_forward_test(Q, K, V, causal):
    O, L = tk.mha_forward(Q, K, V, bool(causal))
    return O, L

def tk_backward_test(Q, K, V, O, L, dO, causal):
    return tk.mha_backward(Q, K, V, O, L, dO, bool(causal))

def check_consistency(b, h, n, d, causal, mean, std, num_iterations=100000):
    
    # Generate fixed inputs
    torch.manual_seed(0)
    Q  = generate_tensor((b, h, n, d), mean, std, torch.bfloat16, 'cuda')
    K  = generate_tensor((b, h, n, d), mean, std, torch.bfloat16, 'cuda')
    V  = generate_tensor((b, h, n, d), mean, std, torch.bfloat16, 'cuda')
    O  = generate_tensor((b, h, n, d), mean, std, torch.bfloat16, 'cuda')
    dO = generate_tensor((b, h, n, d), mean, std, torch.bfloat16, 'cuda')
    
    _, L = tk_forward_test(Q, K, V, causal)

    # Initial run to get reference outputs
    ref_qg, ref_kg, ref_vg = tk_backward_test(Q, K, V, O, L, dO, causal)

    max_diff_qg, max_diff_kg, max_diff_vg = 0, 0, 0

    for _ in tqdm(range(num_iterations), desc="Checking consistency"):
        torch.cuda.synchronize()
        qg, kg, vg = tk_backward_test(Q, K, V, O, L, dO, causal)
        torch.cuda.synchronize()
        
        max_diff_qg = torch.abs(qg - ref_qg).max()
        max_diff_kg = torch.abs(kg - ref_kg).max()
        max_diff_vg = torch.abs(vg - ref_vg).max()
        
        if max_diff_qg > 1e-7 or max_diff_kg > 1e-7 or max_diff_vg > 1e-7:
            breakpoint()

    return max_diff_qg, max_diff_kg, max_diff_vg

# Example usage
b, h, n, d = 1, 16, 768*8, 128
causal = True
mean = 1e-1
std = 1

max_diff_qg, max_diff_kg, max_diff_vg = check_consistency(b, h, n, d, causal, mean, std)