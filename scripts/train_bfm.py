"""Train the cross-embodiment behavior foundation model with online off-policy FB-CPR.

Each robot in the data config has expert motions and its own simulated worlds; defaults
follow BFM-Zero. On a machine with an NVIDIA GPU and Isaac Lab 3::

    python scripts/train_bfm.py --data_config configs/rl/go1_h1_ground.json \\
        --encoder_checkpoint runs/encoder --output_dir runs/bfm --headless

On a CPU, for a small check of the pipeline::

    uv run python scripts/train_bfm.py --simulator mujoco --device cpu ... \\
        --num_env_steps 2000 --online_parallel_envs 8 --num_seed_steps 64 --batch_size 64

Watch the run with ``tensorboard --logdir runs``.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.configuration_bfm import BfmConfig  # noqa: E402
from model.rl.configuration_fb_cpr import FbCprConfig  # noqa: E402
from model.rl.trainer import FbCprTrainer, TrainingArguments  # noqa: E402

MODEL_FIELDS = (
    "z_dim", "hidden_size", "num_attention_heads", "intermediate_size", "num_latent_tokens", "actor_layers",
    "forward_layers", "num_parallel", "actor_std", "seq_length", "state_hidden_size", "state_num_layers", "state_num_heads",
    "state_intermediate_size", "backward_hidden_size", "backward_hidden_layers", "discriminator_hidden_size", "discriminator_hidden_layers",
)
OPTIONAL_INT = {"eval_motions"}


def to_bool(text: str) -> bool:
    """Parse a boolean flag value."""
    if text.lower() in ("1", "true", "yes"):
        return True
    if text.lower() in ("0", "false", "no"):
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean, got {text!r}")


def add_fields(parser: argparse._ArgumentGroup, values: dict) -> None:
    """Add one optional flag per default value, using the value's type."""
    for name, default in values.items():
        if name in OPTIONAL_INT:
            kind = int
        else:
            kind = to_bool if isinstance(default, bool) else type(default) if default is not None else str
        parser.add_argument(f"--{name}", type=kind, default=default, help=f"default: {default}")


def build_parser() -> argparse.ArgumentParser:
    """Return the parser of run, model and agent settings."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data_config", required=True)
    parser.add_argument("--output_dir", required=True)
    add_fields(
        parser.add_argument_group("run"),
        {item.name: item.default for item in dataclasses.fields(TrainingArguments) if item.name not in ("data_config", "output_dir")},
    )
    defaults = {item.name: item.default for item in dataclasses.fields(BfmConfig)}
    add_fields(parser.add_argument_group("model"), {name: defaults[name] for name in MODEL_FIELDS})
    add_fields(
        parser.add_argument_group("agent"),
        {item.name: item.default for item in dataclasses.fields(FbCprConfig) if item.name != "aux_reward_scales"},
    )
    return parser


def split_settings(values: dict) -> tuple[TrainingArguments, dict, FbCprConfig]:
    """Split parsed values into run, model and agent settings."""
    run = TrainingArguments(**{item.name: values[item.name] for item in dataclasses.fields(TrainingArguments)})
    model = {name: values[name] for name in MODEL_FIELDS}
    agent = FbCprConfig(**{item.name: values[item.name] for item in dataclasses.fields(FbCprConfig) if item.name != "aux_reward_scales"})
    return run, model, agent


def train(run: TrainingArguments, model: dict, agent: FbCprConfig, simulation=None) -> None:
    """Train and print where the results are."""
    trainer = FbCprTrainer(run, model, agent, simulation)
    names = ", ".join(f"{e.record.name} ({e.record.graph.num_joints} DoF, {len(e.motions)} motion chunks)" for e in trainer.embodiments)
    print(f"robots: {names}", flush=True)
    print(f"networks: {sum(p.numel() for p in trainer.agent.model.parameters()) / 1e6:.2f}M parameters on {run.device}", flush=True)
    print(f"TensorBoard: tensorboard --logdir {run.output_dir}", flush=True)
    trainer.train()
    print(f"checkpoint: {Path(run.output_dir) / 'checkpoint.pt'}", flush=True)


def main(argv: list[str] | None = None) -> None:
    """Parse the arguments, launch the simulator if needed, and train."""
    parser = build_parser()
    known, _ = parser.parse_known_args(argv)
    if known.simulator == "isaaclab":
        try:
            from isaaclab.app import add_launcher_args, launch_simulation
        except ImportError as error:
            parser.error(f"Isaac Lab is not installed ({error}). Run inside its Python environment, or pass --simulator mujoco for a small CPU check.")
        add_launcher_args(parser)
        arguments = parser.parse_args(argv)
        run, model, agent = split_settings(vars(arguments))
        import isaaclab.sim as sim_utils

        physics_dt = 0.005
        sim_cfg = sim_utils.SimulationCfg(dt=physics_dt, device=run.device)
        with launch_simulation(sim_cfg, arguments):
            train(run, model, agent, sim_utils.SimulationContext(sim_cfg))
        return
    run, model, agent = split_settings(vars(parser.parse_args(argv)))
    train(run, model, agent)


if __name__ == "__main__":
    main()
