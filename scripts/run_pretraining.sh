#!/usr/bin/env bash
# Launch large physics pretraining using configs from generate_training_configs.py.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

DATA_CONFIG="configs/data/pretraining_selected.json"
TRAIN_CONFIG="configs/pretraining/selected_large.json"
if [[ ! -f "$DATA_CONFIG" || ! -f "$TRAIN_CONFIG" ]]; then
    echo "Missing training configs; run: uv run python scripts/generate_training_configs.py" >&2
    exit 1
fi

exec uv run python scripts/train_physics_da_get.py "$TRAIN_CONFIG"
