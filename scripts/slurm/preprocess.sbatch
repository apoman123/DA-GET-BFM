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
DATA_CONFIG="${DATA_CONFIG:-$PROJECT_ROOT/configs/data/pretraining_selected.json}"
CACHE_DIR="${CACHE_DIR:-$PROJECT_ROOT/.cache/physics_data}"
PREPROCESS_WORKERS="${PREPROCESS_WORKERS:-8}"
test -f "$DATA_CONFIG" || { echo "Data config missing: $DATA_CONFIG" >&2; exit 1; }
export HF_HOME="${HF_HOME:-$DATA_ROOT/.hf-cache}"
export HF_HUB_DISABLE_TELEMETRY=1

mkdir -p "$CACHE_DIR" "$HF_HOME"
"$PYTHON_BIN" -c '
import sys

from model.training import build_physics_datasets

datasets = build_physics_datasets(
    sys.argv[1], cache_dir=sys.argv[2], workers=int(sys.argv[3]), progress=True
)
print(
    f"Preprocessing ready: {len(datasets.records)} embodiments, "
    f"{len(datasets.train)} training samples"
)
' "$DATA_CONFIG" "$CACHE_DIR" "$PREPROCESS_WORKERS"
