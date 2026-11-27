#!/usr/bin/env bash
# Build the physics pretraining cache before submitting pretrain.sbatch.
#SBATCH --job-name=bfm-preprocess
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
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
test -f "$PRETRAIN_CONFIG" || { echo "Pretraining config missing: $PRETRAIN_CONFIG" >&2; exit 1; }
export HF_HOME="${HF_HOME:-$DATA_ROOT/.hf-cache}"
export HF_HUB_DISABLE_TELEMETRY=1

mkdir -p "$HF_HOME"
"$PYTHON_BIN" scripts/preprocess_physics_data.py "$PRETRAIN_CONFIG"
