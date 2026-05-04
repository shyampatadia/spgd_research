#!/bin/bash
# ----------------------------------------------------------------------------
# One-time bootstrap on the Turing login node.
# Run AFTER the first scp push has placed the project at ~/spgd-study.
#
#   ssh <user>@<host>
#   cd ~/spgd-study
#   bash turing/setup_turing.sh
#
# What this does:
#   1. Installs `uv` (no admin needed; lands in ~/.local/bin).
#   2. Resolves the project's Python deps from pyproject.toml + uv.lock.
#   3. Pre-downloads CIFAR-10 into .cifar10_cache/ so the 15 array jobs
#      do NOT race-download the dataset in parallel.
#   4. Smoke-tests CUDA visibility from the *login node* (best-effort;
#      most clusters expose GPUs only on compute nodes -- a "no GPU on
#      login" message here is normal and not an error).
# ----------------------------------------------------------------------------

set -euo pipefail

cd "$(dirname "$0")/.."        # project root, regardless of where invoked
PROJECT_ROOT="$(pwd)"
echo "[setup] project root: $PROJECT_ROOT"

# 1. Install uv if missing -----------------------------------------------------
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1 ; then
    echo "[setup] installing uv to ~/.local/bin ..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
echo "[setup] uv: $(uv --version)"

# 2. Sync deps -----------------------------------------------------------------
# uv.lock pins everything; cu121 wheels work on A30 (Ampere).
echo "[setup] uv sync (this will take a few minutes the first time) ..."
uv sync

# 3. Prefetch CIFAR-10 ---------------------------------------------------------
mkdir -p .cifar10_cache results logs
echo "[setup] prefetching CIFAR-10 into .cifar10_cache/ ..."
uv run python -c "
from spgd_study.data import load_cifar10
tr, te = load_cifar10(data_dir='.cifar10_cache', batch_size=128,
                      num_workers=0, test_batch_size=256)
print('  train batches:', len(tr), ' test batches:', len(te))
print('  CIFAR-10 cache ready')
"

# 4. CUDA / torch sanity (login node; OK if no GPU here) -----------------------
echo "[setup] torch / CUDA visibility:"
uv run python -c "
import torch
print('  torch         :', torch.__version__)
print('  cuda built    :', torch.version.cuda)
print('  cuda available:', torch.cuda.is_available())
print('  device count  :', torch.cuda.device_count())
"

echo ""
echo "[setup] DONE.  You can now submit the array job:"
echo "        sbatch turing/slurm_exp4.sbatch"
