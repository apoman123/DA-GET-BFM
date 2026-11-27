# Run the selected training pipeline on TWCC Nano5

These five independent jobs follow the
[Nano5 manual](https://man.twcc.ai/@AI-Pilot/manual): download, convert,
preprocess, pretrain the physics encoder, and train the BFM. Submit each script
separately from the repository root on a Nano5 login node. The work runs on
compute nodes.
An iService project account is required for each submission.

The default partition is `normal` (H100, up to 48 hours per job). Use
`sbatch -p normal2` for H200.
The `dev` partition has a two-hour limit and is useful for short checks. The
download, conversion, and preprocessing jobs request CPUs and memory; each
training job requests one GPU. Resource needs and time limits are estimates,
so adjust them to your quota and actual data size.

## Prepare the Python environment

On an allocated Nano5 compute node, install the project into a Nano5-created
environment with Python 3.10 or newer and CUDA-enabled PyTorch. For example,
if `uv` is available:

```bash
cd /path/to/DA-GET-BFM
uv sync --locked --all-extras
.venv/bin/python -c 'import torch; print(torch.__version__)'
```

The scripts use `.venv/bin/python` by default. Set `PYTHON_BIN` to the absolute
path of another prepared Python interpreter when needed. A virtual environment
copied from another computer is generally unsuitable. For Isaac Lab, use an
interpreter with both this project and Isaac Lab installed. Do not install
packages or perform heavy preprocessing on a login node.

## Submit the jobs

The default data selection includes GrandTour, MIPO, G1 WBT, G1 WBT q36,
G1 WBT SMPL, xwbc G1 AMASS, LocoMuJoCo G1, G1 Dex3, LAFAN1 G1, and LAFAN1 H1.
It is a large download and conversion; check [dataset licenses](data.md) and
available storage first. Raw downloads remain after conversion. `--no-hf_dataset`
is unsuitable for the selected configs, which expect local Hugging Face copies.

Each script uses this default data location:

```bash
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/data}"
```

Set `DATA_ROOT` once in your shell before submitting each job, or replace the
default in each script with your absolute storage path. For example:

```bash
cd /path/to/DA-GET-BFM
export DATA_ROOT=/work/YOUR_PROJECT/robotics_datasets
mkdir -p logs
sbatch -A YOUR_ISERVICE_PROJECT_ID scripts/slurm/download.sbatch
# After the download completes:
sbatch -A YOUR_ISERVICE_PROJECT_ID scripts/slurm/convert.sbatch
# After conversion completes, prepare configs with Nano5 paths (see below), then:
sbatch -A YOUR_ISERVICE_PROJECT_ID scripts/slurm/preprocess.sbatch
# After preprocessing completes:
sbatch -A YOUR_ISERVICE_PROJECT_ID scripts/slurm/pretrain.sbatch
# After pretraining produces an encoder checkpoint:
sbatch -A YOUR_ISERVICE_PROJECT_ID scripts/slurm/train_bfm.sbatch
```

`PROJECT_ROOT` is the directory where you run `sbatch`; submit from the
repository root. The scripts have no Slurm dependencies, so submit only the
stage you want. You can run the download and conversion stages again; they
skip completed files.

The preprocessing job runs `scripts/preprocess_physics_data.py` with the same
`PRETRAIN_CONFIG` JSON used by pretraining. It calls the same
`build_physics_datasets()` function and writes a reusable Arrow cache. A change
to the data config or URDF contents creates a new cache key. An interrupted run
reuses completed sources. The script can also run locally from the repository
root with `uv run python scripts/preprocess_physics_data.py`.

Use `squeue -u "$USER"` or `sacct -j JOB_ID` to check status. Slurm logs are
`logs/<job-name>-<job-id>.out` and `.err` in the repository root. Create
`logs/` before submitting. Avoid
rapid polling with `watch` or a loop, as the Nano5 manual requests.

To estimate the download before submitting, run the catalog's `--list` and
`--dry_run` options from a short allocated job. The `download.sbatch` script
accepts extra catalog arguments after its filename, such as `--limit 20`.
The matching conversion job must use the same selection. `DATASETS` can select
another comma-separated catalog subset, but the selected pretraining config
generator requires all ten defaults; for a subset, use an appropriate custom
data config and training command instead.

The training scripts read their existing JSON configurations as-is. The
checked-in selected configs contain paths from the machine where they were
generated; update those paths for Nano5 before submitting. If you use the
default dataset selection, the project's existing
`.venv/bin/python scripts/generate_training_configs.py --data-root "$DATA_ROOT"`
can regenerate the selected configs once after conversion, before preprocessing.
This is a separate setup command,
not part of any Slurm job. Alternatively, supply your own config paths through
`PRETRAIN_CONFIG` and `BFM_DATA_CONFIG`. Changing `DATA_ROOT` alone does not
rewrite paths inside JSON files.

`pretrain.sbatch` passes its JSON directly to the training entry point. Set
the batch size and other training values in that JSON before submitting. The
shipped batch size is 64 and may need to be reduced for one GPU. To resume a
pretraining run, set `resume_from_checkpoint` in the JSON to a saved Trainer
checkpoint directory. Set `BFM_RESUME_FROM` to a saved `checkpoint.pt` to resume
BFM training. The BFM
job uses MuJoCo by default. To use Isaac Lab, submit it with `SIMULATOR=isaaclab`
and an appropriate `PYTHON_BIN` in the exported environment. Both training jobs
save checkpoints under `runs/`.

The manual describes the [partition limits](https://man.twcc.ai/@AI-Pilot/BJYB5G_zke),
[batch scripts](https://man.twcc.ai/@AI-Pilot/r1os5G_Mkl), and
[environment setup](https://man.twcc.ai/@AI-Pilot/SyXURRpP1x).
