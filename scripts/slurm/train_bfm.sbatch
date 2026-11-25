#!/usr/bin/env bash
# Continue from the encoder produced by pretrain.sbatch.
#SBATCH --job-name=bfm-rl
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:?Submit this job with sbatch from the repository root}"
cd "$PROJECT_ROOT"
test -f pyproject.toml || { echo 'Submit from the repository root.' >&2; exit 1; }
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"
test -x "$PYTHON_BIN" || { echo "Python not found: $PYTHON_BIN" >&2; exit 1; }
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/data}"
BFM_DATA_CONFIG="${BFM_DATA_CONFIG:-$PROJECT_ROOT/configs/rl/selected_lafan_local.json}"
ENCODER_CHECKPOINT="${ENCODER_CHECKPOINT:-$PROJECT_ROOT/runs/pretraining_large_selected}"
BFM_OUTPUT_DIR="${BFM_OUTPUT_DIR:-$PROJECT_ROOT/runs/rl_lafan_large}"
SIMULATOR="${SIMULATOR:-mujoco}"
export HF_HOME="${HF_HOME:-$DATA_ROOT/.hf-cache}"
export HF_HUB_DISABLE_TELEMETRY=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

test -f "$ENCODER_CHECKPOINT/config.json" || {
    echo "Encoder checkpoint missing: $ENCODER_CHECKPOINT/config.json" >&2
    exit 1
}
mkdir -p "$BFM_OUTPUT_DIR" "$HF_HOME"
test -f "$BFM_DATA_CONFIG" || {
    echo "Missing RL data config: $BFM_DATA_CONFIG" >&2
    exit 1
}

resume_args=()
if [[ -n "${BFM_RESUME_FROM:-}" ]]; then
    resume_args=(--resume_from "$BFM_RESUME_FROM")
fi

"$PYTHON_BIN" scripts/train_bfm.py \
    --data_config "$BFM_DATA_CONFIG" \
    --encoder_checkpoint "$ENCODER_CHECKPOINT" \
    --output_dir "$BFM_OUTPUT_DIR" \
    --simulator "$SIMULATOR" --device cuda \
    --num_env_steps "${NUM_ENV_STEPS:-2000000}" \
    --online_parallel_envs "${PARALLEL_ENVS:-8}" \
    --buffer_size "${BUFFER_SIZE:-100000}" \
    --num_seed_steps "${SEED_STEPS:-64}" \
    --update_agent_every "${UPDATE_EVERY:-8}" \
    --num_agent_updates "${UPDATES_PER_ROUND:-1}" \
    --batch_size "${BATCH_SIZE:-16}" \
    --physics_batch_size "${PHYSICS_BATCH_SIZE:-8}" \
    --physics_replay_envs 2 \
    --log_every 1000 --eval_every 50000 --eval_motions 2 \
    --checkpoint_every 50000 "${resume_args[@]}" "$@"
