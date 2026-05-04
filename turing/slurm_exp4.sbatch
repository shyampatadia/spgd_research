#!/bin/bash
# ----------------------------------------------------------------------------
# Experiment 4 — CIFAR-10 / ResNet-18 anchor experiment
# 5 optimizers x 3 seeds = 15 runs, submitted as a Slurm array job.
# Array index i maps to (opt, seed) via:
#     OPT  = OPTS[i / 3]    where OPTS  = (sgd adam pgd rpgd spgd)
#     SEED = SEEDS[i % 3]   where SEEDS = (0 1 2)
#
# Submission (run from project root on Turing login node):
#     mkdir -p logs results
#     sbatch turing/slurm_exp4.sbatch
#
# Single-task re-run (e.g., resubmit just the SPGD seed-1 run = index 13):
#     sbatch --array=13 turing/slurm_exp4.sbatch
# ----------------------------------------------------------------------------

#SBATCH --job-name=spgd-exp4
#SBATCH --partition=academic
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --array=0-14
#SBATCH --output=logs/exp4_%A_%a.out
#SBATCH --error=logs/exp4_%A_%a.err

set -euo pipefail

echo "========================================================================"
echo "[$(date)] job=$SLURM_JOB_ID  array_id=$SLURM_ARRAY_TASK_ID"
echo "          host=$(hostname)  cwd=$SLURM_SUBMIT_DIR"
echo "========================================================================"

cd "$SLURM_SUBMIT_DIR"

# ----------------------------------------------------------------------------
# Module environment.  WPI Turing uses Lmod; module names below are common
# defaults.  If `module avail` on Turing shows different names, edit these
# two lines (or comment them out -- uv ships its own Python and pip-installs
# CUDA-bundled torch wheels, so a system Python/CUDA is not strictly needed).
# ----------------------------------------------------------------------------
module purge || true
module load cuda/12.1 2>/dev/null || module load cuda 2>/dev/null || true

# ----------------------------------------------------------------------------
# Make uv visible.  Bootstrap installs uv to ~/.local/bin (see setup_turing.sh).
# ----------------------------------------------------------------------------
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || { echo "ERROR: uv not on PATH; run turing/setup_turing.sh first"; exit 1; }

# ----------------------------------------------------------------------------
# Decode array index -> (opt, seed)
# ----------------------------------------------------------------------------
OPTS=(sgd adam pgd rpgd spgd)
SEEDS=(0 1 2)
i=$SLURM_ARRAY_TASK_ID
OPT=${OPTS[$((i / 3))]}
SEED=${SEEDS[$((i % 3))]}

echo "config: opt=$OPT  seed=$SEED  device=cuda"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
echo "------------------------------------------------------------------------"

# ----------------------------------------------------------------------------
# Run.  --data-dir is pinned to a project-local cache so all array tasks share
# the same prefetched CIFAR-10 (download is done once in setup_turing.sh).
# ----------------------------------------------------------------------------
uv run python experiments/exp4_cifar10.py \
    --opt   "$OPT" \
    --seed  "$SEED" \
    --device cuda \
    --data-dir "$SLURM_SUBMIT_DIR/.cifar10_cache" \
    --out-dir  "$SLURM_SUBMIT_DIR/results"

echo "========================================================================"
echo "[$(date)] task $SLURM_ARRAY_TASK_ID (opt=$OPT seed=$SEED) finished OK"
echo "========================================================================"
