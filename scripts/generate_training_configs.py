"""Generate the selected pretraining and RL configs from local datasets.

Run this once before either training shell script. Generated config files stay
inside the project; ``--data-root`` is only read. ``--cache-dir`` selects where
processed physics episodes are saved and reused by both training stages.
Existing selected-large training hyperparameters are preserved when configs
are regenerated.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRETRAIN_DATASETS = (
    "grandtour",
    "mipo",
    "g1_wbt",
    "g1_wbt_q36",
    "g1_wbt_smpl",
    "xwbc_g1_amass",
    "locomujoco_g1",
    "g1_dex3",
)
RL_DATASETS = (("lafan1_g1", "g1"), ("lafan1_h1", "h1"))


def load_sources(data_root: Path, dataset: str) -> list[dict[str, Any]]:
    """Read converted source entries for one dataset.

    Args:
        data_root (Path): Directory containing the converted dataset folders.
        dataset (str): Folder name beneath ``data_root``.

    Returns:
        list[dict[str, Any]]: Nonempty source entries in stored order.

    Raises:
        FileNotFoundError: The dataset has no ``sources.json``.
        ValueError: ``sources.json`` is empty or is not a list.
        json.JSONDecodeError: The source file is not valid JSON.
    """
    path = data_root / dataset / "sources.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing converted sources: {path}")
    sources = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(sources, list) or not sources:
        raise ValueError(f"Expected a nonempty source list in {path}")
    return sources


def write_config(path: Path, config: dict[str, Any]) -> None:
    """Write a generated UTF-8 JSON configuration inside the project.

    Args:
        path (Path): Output file in an existing project config directory.
        config (dict[str, Any]): JSON-compatible configuration content.

    Returns:
        None: Writes formatted JSON to ``path``.

    Raises:
        OSError: The file cannot be written.
        TypeError: A configuration value cannot be serialized.
    """
    path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    """Generate data configs with a shared cache and large-model arguments.

    Returns:
        None: Writes pretraining data, RL data, and model argument JSON files.

    Raises:
        FileNotFoundError: A template or selected converted dataset is missing.
        ValueError: A selected dataset has an unexpected source count.
        json.JSONDecodeError: A template or source file is not valid JSON.
        OSError: A generated configuration cannot be written.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/mnt/e/robotics_datasets"),
        help="Directory with converted dataset folders and sources.json files.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=PROJECT_ROOT / ".cache/physics_data",
        help="Directory for reusable processed physics episodes.",
    )
    options = parser.parse_args()
    data_root = options.data_root.expanduser().resolve()
    cache_dir = options.cache_dir.expanduser().resolve()

    pretrain = json.loads(
        (PROJECT_ROOT / "configs/data/zero_shot.json").read_text(encoding="utf-8")
    )
    for dataset in PRETRAIN_DATASETS:
        pretrain["sources"].extend(load_sources(data_root, dataset))
    pretrain["sampling_alpha"] = None
    pretrain_path = PROJECT_ROOT / "configs/data/pretraining_selected.json"
    write_config(pretrain_path, pretrain)

    selected_path = PROJECT_ROOT / "configs/pretraining/selected_large.json"
    if selected_path.is_file():
        arguments = json.loads(selected_path.read_text(encoding="utf-8"))
    else:
        arguments = json.loads(
            (PROJECT_ROOT / "configs/pretraining/zero_shot.json").read_text(
                encoding="utf-8"
            )
        )
        arguments.update(
            {
                "output_dir": str(PROJECT_ROOT / "runs/pretraining_large_selected"),
                "per_device_train_batch_size": 8,
                "gradient_accumulation_steps": 8,
                "bf16": True,
                "preprocess_workers": 2,
                "dataloader_num_workers": 2,
                "max_steps": -1,
                "num_train_epochs": 1,
            }
        )
    arguments.update(
        {
            "data_config": str(pretrain_path),
            "cache_dir": str(cache_dir),
            "preset": "large",
        }
    )
    arguments.pop("root_token", None)
    write_config(selected_path, arguments)

    ground = json.loads(
        (PROJECT_ROOT / "configs/data/ground_contact.json").read_text(encoding="utf-8")
    )
    environments = {source["name"]: source for source in ground["sources"]}
    rl = {
        key: pretrain[key]
        for key in (
            "history_size",
            "joint_features",
            "seed",
            "control_dt",
            "target_normalization",
            "dimensionless_inputs",
            "group_variants",
        )
    }
    rl["cache_dir"] = str(cache_dir)
    rl["sources"] = []
    for dataset, robot in RL_DATASETS:
        sources = load_sources(data_root, dataset)
        if len(sources) != 1:
            raise ValueError(f"Expected one source for {dataset}; got {len(sources)}")
        environment = environments[f"{robot}_ground"]
        source = dict(sources[0])
        source.update({
            "robot": robot,
            "expert": True,
            "armature": 0.01,
            "nominal_pose": environment["nominal_pose"],
            "contact_geometry": environment["contact_geometry"],
        })
        rl["sources"].append(source)
    rl_path = PROJECT_ROOT / "configs/rl/selected_lafan_local.json"
    write_config(rl_path, rl)
    print(f"Pretraining: {len(pretrain['sources'])} sources -> {pretrain_path}")
    print(f"RL: {len(rl['sources'])} expert sources -> {rl_path}")


if __name__ == "__main__":
    main()
