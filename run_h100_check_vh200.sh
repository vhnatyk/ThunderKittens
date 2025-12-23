#!/usr/bin/env bash
set -euo pipefail

cd /root/vh/ThunderKittens

source /root/vh/ThunderKittens/vh200/bin/activate

# Work around environments where torch loads a stub libcudart.so.12 (missing symbols)
# that breaks FlashAttention-3. This is safe even if not needed.
export TK_CUDART_PATH="${TK_CUDART_PATH:-/root/miniconda3/lib/libcudart.so.12}"

python kernels/attn/h100/h100_check.py


