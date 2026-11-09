"""Write a data config for a robot (and optionally a dataset) automatically.

From the URDF it infers whether the base is floating, which links are feet,
whether feet are flat, which joints are end effectors, and, for multirotors,
which joints are rotors and plausible propeller coefficients. From a dataset
(a Hub repo id or a converted folder) it matches columns to URDF joints and
finds base pose, twist, gravity, IMU, and contact columns. It prints what it
decided and why; check the report and edit the JSON where needed.

Examples::

    # A Hub dataset of a floating-base humanoid
    uv run python scripts/make_data_config.py --urdf assets/urdf/g1_29dof.urdf \\
        --dataset unitreerobotics/G1_WBT_Inspire_Pick_Up_Fruit --synthetic --output configs/data/g1.json

    # Synthetic data only, for any URDF (fixed, legged, or multirotor)
    uv run python scripts/make_data_config.py --urdf my_drone.urdf --synthetic --output configs/data/drone.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model import EmbodimentGraph  # noqa: E402
from model.conversion.mapping import inspect_dataset  # noqa: E402
from model.robot_analysis import analyze_robot, describe, suggested_position_gains  # noqa: E402


def build_config(arguments: argparse.Namespace) -> dict:
    """Assemble the data config from the robot profile and the dataset mapping."""
    urdf = str(Path(arguments.urdf).resolve())
    profile = analyze_robot(
        urdf,
        thrust_to_weight=arguments.thrust_to_weight,
        max_rotor_speed=arguments.max_rotor_speed,
        drag_ratio=arguments.drag_ratio,
    )
    if arguments.kind == "fixed":
        profile.floating_base, profile.contact_links, profile.kind = False, [], "fixed"
    elif arguments.kind == "floating":
        profile.floating_base = True
    print("robot:")
    print("  " + describe(profile).replace("\n", "\n  "))
    name = arguments.name or Path(urdf).stem
    sources = []
    if arguments.dataset:
        graph = EmbodimentGraph.from_urdf(urdf)
        result = inspect_dataset(arguments.dataset, list(graph.joint_names), arguments.match_threshold)
        source = {"type": "huggingface", "name": f"{name}_data"}
        dataset_path = Path(arguments.dataset)
        if dataset_path.is_dir():
            source["path"] = str(dataset_path.resolve())
        else:
            source["repo_id"] = arguments.dataset
            source["data_files"] = "data/*/*.parquet"
        converted = dataset_path / "source.json"
        if dataset_path.is_dir() and converted.exists():
            # Converters know their dataset best; keep their settings.
            source.update({k: v for k, v in json.loads(converted.read_text()).items() if k not in ("type", "name", "path")})
        source.update({key: value for key, value in result["source"].items() if key not in source or key in ("joint_names", "state_key", "state_indices")})
        source["urdf"] = urdf
        source.update({k: v for k, v in profile.source_settings().items() if k not in source or k == "floating_base"})
        if source.get("action_type") == "position" and "actuator_defaults" not in source:
            gains = suggested_position_gains(graph)
            overrides = source.setdefault("actuator_overrides", {})
            for joint, values in gains.items():
                overrides.setdefault(joint, {}).update(values)
        if arguments.max_episodes:
            source["max_episodes"] = arguments.max_episodes
        print(f"dataset {arguments.dataset}: matched {len(source['joint_names'])} joints")
        print("\n".join(result["report"]))
        if result["unmatched"]:
            print(f"  not recorded (frozen at 0): {result['unmatched']}")
        sources.append(source)
    if arguments.synthetic:
        modes = {"multirotor": ["flight"], "fixed": ["fixed"]}.get(profile.kind, ["fixed", "floating"])
        synthetic = {
            "type": "synthetic",
            "name": f"{name}_synthetic",
            "urdf": urdf,
            "base_modes": modes,
            "randomized_embodiments": arguments.randomized_embodiments,
            "episodes_per_embodiment": 8,
            "steps": 200,
            **profile.source_settings(),
        }
        if profile.kind == "multirotor":
            synthetic["randomization"] = {"damping": [0.0, 0.0], "friction": [0.0, 0.0]}
        sources.append(synthetic)
        print(f"synthetic source: base modes {modes}")
        if profile.contact_links and arguments.ground:
            sources.append({
                "type": "mujoco",
                "name": f"{name}_ground",
                "urdf": urdf,
                "randomized_embodiments": arguments.randomized_embodiments,
                "episodes_per_embodiment": 8,
                "steps": 300,
                **profile.source_settings(),
            })
            print("mujoco ground-contact source added (needs `uv sync --extra sim`)")
    if not sources:
        raise SystemExit("nothing to write: pass --dataset and/or --synthetic")
    return {"history_size": 3, "joint_features": "da_get_zero", "eval_fraction": 0.05, "seed": 0, "sources": sources}


def main() -> None:
    """Parse arguments and write the config."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--urdf", required=True)
    parser.add_argument("--dataset", help="Hub dataset id or local (converted) folder.")
    parser.add_argument("--name")
    parser.add_argument("--output", help="JSON file to write; printed when omitted.")
    parser.add_argument("--synthetic", action="store_true", help="Add a synthetic source for the robot.")
    parser.add_argument("--ground", action="store_true", help="Also add a MuJoCo ground-contact source (legged robots).")
    parser.add_argument("--kind", choices=["auto", "fixed", "floating"], default="auto")
    parser.add_argument("--thrust_to_weight", type=float, default=2.5)
    parser.add_argument("--max_rotor_speed", type=float)
    parser.add_argument("--drag_ratio", type=float, default=0.016)
    parser.add_argument("--match_threshold", type=float, default=0.4)
    parser.add_argument("--max_episodes", type=int)
    parser.add_argument("--randomized_embodiments", type=int, default=8)
    arguments = parser.parse_args()
    config = build_config(arguments)
    text = json.dumps(config, indent=2)
    if arguments.output:
        Path(arguments.output).parent.mkdir(parents=True, exist_ok=True)
        Path(arguments.output).write_text(text + "\n")
        print(f"wrote {arguments.output}")
    else:
        print(text)


if __name__ == "__main__":
    main()
