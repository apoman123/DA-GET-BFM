#!/usr/bin/env bash
# Pretrain the large physics encoder from its existing JSON config.
#SBATCH --job-name=bfm-pretrain
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:?Submit this job with sbatch from the repository root}"
cd "$PROJECT_ROOT"
test -f pyproject.toml || { echo 'Submit from the repository root.' >&2; exit 1; }
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"
test -x "$PYTHON_BIN" || { echo "Python not found: $PYTHON_BIN" >&2; exit 1; }
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/data}"
PRETRAIN_CONFIG="${PRETRAIN_CONFIG:-$PROJECT_ROOT/configs/pretraining/selected_large.json}"
export HF_HOME="${HF_HOME:-$DATA_ROOT/.hf-cache}"
export HF_HUB_DISABLE_TELEMETRY=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

test -f "$PRETRAIN_CONFIG" || {
    echo "Missing pretraining config: $PRETRAIN_CONFIG" >&2
    exit 1
}
mkdir -p "$HF_HOME"
"$PYTHON_BIN" scripts/train_physics_da_get.py "$PRETRAIN_CONFIG"
