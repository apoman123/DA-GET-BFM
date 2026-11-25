#!/usr/bin/env bash
# Download the datasets used by the selected encoder and BFM training configs.
# Submit from the repository root with: sbatch -A PROJECT_ID scripts/slurm/download.sbatch
#SBATCH --job-name=bfm-download
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

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
"$PYTHON_BIN" scripts/download_datasets.py \
    --dataset "$DATASETS" --root "$DATA_ROOT" --no-convert \
    --workers "${DOWNLOAD_WORKERS:-8}" "$@"
