#!/usr/bin/env bash
# Launch FB-CPR using local LAFAN1 expert motions and a pretrained encoder.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

DATA_CONFIG="configs/rl/selected_lafan_local.json"
ENCODER_CHECKPOINT="${ENCODER_CHECKPOINT:-$PROJECT_ROOT/runs/pretraining_large_selected}"
SIMULATOR="${SIMULATOR:-mujoco}"
if [[ ! -f "$DATA_CONFIG" ]]; then
    echo "Missing RL data config; run: uv run python scripts/generate_training_configs.py" >&2
    exit 1
fi
if [[ ! -f "$ENCODER_CHECKPOINT/config.json" ]]; then
    echo "Missing pretrained encoder at $ENCODER_CHECKPOINT; run run_pretraining_large.sh first." >&2
    exit 1
fi

case "$SIMULATOR" in
    mujoco) PYTHON_COMMAND=(uv run python) ;;
    isaaclab) PYTHON_COMMAND=("${ISAACLAB_PYTHON:-python}") ;;
    *) echo "SIMULATOR must be mujoco or isaaclab" >&2; exit 1 ;;
esac

exec "${PYTHON_COMMAND[@]}" scripts/train_bfm.py \
    --data_config "$DATA_CONFIG" \
    --encoder_checkpoint "$ENCODER_CHECKPOINT" \
    --output_dir runs/rl_lafan_large \
    --simulator "$SIMULATOR" \
    --device cuda \
    --num_env_steps "${NUM_ENV_STEPS:-2000000}" \
    --online_parallel_envs "${PARALLEL_ENVS:-8}" \
    --buffer_size "${BUFFER_SIZE:-100000}" \
    --num_seed_steps "${SEED_STEPS:-64}" \
    --update_agent_every "${UPDATE_EVERY:-8}" \
    --num_agent_updates "${UPDATES_PER_ROUND:-1}" \
    --batch_size "${BATCH_SIZE:-16}" \
    --physics_batch_size "${PHYSICS_BATCH_SIZE:-8}" \
    --physics_replay_envs 2 \
    --log_every 1000 \
    --eval_every 50000 \
    --eval_motions 2 \
    --checkpoint_every 50000 \
    "$@"
