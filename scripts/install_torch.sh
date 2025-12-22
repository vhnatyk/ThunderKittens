#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: install_torch.sh [--python PY_BIN] [--cuda FLAVOR] [--extra PKGS]

Installs torch, torchvision, and torchaudio into the active Python environment.
By default it targets the CUDA 13.0 wheels. Override via --cuda or set TORCH_CUDA.

Supported CUDA FLAVOR values:
  cpu      Use CPU-only wheels
  cu121    Use CUDA 12.1 wheels
  cu124    Use CUDA 12.4 wheels
  cu130    Use CUDA 13.0 wheels (default)
  rocm6.1  Use ROCm 6.1 wheels
EOF
}

PY_BIN="${PY_BIN:-python3}"
CUDA_FLAVOR="${TORCH_CUDA:-cu130}"
EXTRA_PKGS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python)
      [[ $# -ge 2 ]] || { echo "Missing value for --python" >&2; exit 1; }
      PY_BIN="$2"
      shift 2
      ;;
    --cuda)
      [[ $# -ge 2 ]] || { echo "Missing value for --cuda" >&2; exit 1; }
      CUDA_FLAVOR="$2"
      shift 2
      ;;
    --extra)
      [[ $# -ge 2 ]] || { echo "Missing value for --extra" >&2; exit 1; }
      EXTRA_PKGS="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

case "$CUDA_FLAVOR" in
  cpu)
    INDEX_URL="https://download.pytorch.org/whl/cpu"
    ;;
  cu121)
    INDEX_URL="https://download.pytorch.org/whl/cu121"
    ;;
  cu124)
    INDEX_URL="https://download.pytorch.org/whl/cu124"
    ;;
  cu130)
    INDEX_URL="https://download.pytorch.org/whl/nightly/cu130"
    ;;
  rocm6.1)
    INDEX_URL="https://download.pytorch.org/whl/rocm6.1"
    ;;
  *)
    echo "Unsupported CUDA flavor: $CUDA_FLAVOR" >&2
    usage
    exit 1
    ;;
esac

echo "Using python interpreter: $PY_BIN"
echo "Installing torch packages from: $INDEX_URL"

"$PY_BIN" -m pip install --upgrade pip
"$PY_BIN" -m pip install --index-url "$INDEX_URL" torch torchvision torchaudio ${EXTRA_PKGS:+$EXTRA_PKGS}


