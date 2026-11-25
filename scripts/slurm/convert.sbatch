#!/usr/bin/env bash
# Convert datasets previously downloaded to DATA_ROOT.
#SBATCH --job-name=bfm-convert
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:?Submit this job with sbatch from the repository root}"
cd "$PROJECT_ROOT"
test -f pyproject.toml || { echo 'Submit from the repository root.' >&2; exit 1; }
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"
test -x "$PYTHON_BIN" || { echo "Python not found: $PYTHON_BIN" >&2; exit 1; }
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/data}"
DATASETS="${DATASETS:-grandtour,mipo,g1_wbt,g1_wbt_q36,g1_wbt_smpl,xwbc_g1_amass,locomujoco_g1,g1_dex3,lafan1_g1,lafan1_h1}"
export HF_HOME="${HF_HOME:-$DATA_ROOT/.hf-cache}"
export HF_HUB_DISABLE_TELEMETRY=1

mkdir -p "$DATA_ROOT" "$HF_HOME"
"$PYTHON_BIN" scripts/convert_datasets.py \
    --dataset "$DATASETS" --root "$DATA_ROOT" "$@"
