"""Convert a public robot dataset to the training table format.

Writes ``<output>/data/chunk-000/file-000.parquet``, ``<output>/meta/info.json``,
and ``<output>/source.json`` (a data-config source you can paste into a data
config, or pass to ``scripts/make_data_config.py --dataset <output>``).

Examples::

    uv sync --extra convert --extra validation
    uv run python scripts/convert_dataset.py mipo --output data/mipo --pattern "indoor/*trot*" --max_files 10
    uv run python scripts/convert_dataset.py grandtour --output data/grandtour --max_missions 2
    uv run python scripts/convert_dataset.py neurobem --output data/neurobem --max_flights 20
    uv run python scripts/convert_dataset.py locomujoco --output data/loco_g1 --robot UnitreeG1
    uv run python scripts/convert_dataset.py amass_g1 --output data/amass_g1 --pattern "*Walk*" --max_files 100
    uv run python scripts/convert_dataset.py lafan1 --output data/lafan1_g1 --robot g1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.conversion.datasets import CONVERTERS  # noqa: E402


def main() -> None:
    """Parse arguments and run one converter."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="dataset", required=True)

    def add(name: str, help_text: str):
        command = sub.add_parser(name, help=help_text)
        command.add_argument("--output", required=True)
        command.add_argument("--pattern", action="append", help="Glob over file names; repeatable.")
        return command

    command = add("mipo", "MIPO Unitree Go1 (Open-IO/mipo-go1-odometry): torques, IMU, foot forces")
    command.add_argument("--max_files", type=int)
    command.add_argument("--rate", type=float, default=200.0)
    command.add_argument("--contact_threshold", type=float, default=200.0)
    command.add_argument("--urdf", default="assets/urdf/go1.urdf")
    command = add("grandtour", "GrandTour ANYmal D (leggedrobotics/grand_tour_dataset): torques, contacts, base state")
    command.add_argument("--max_missions", type=int, default=1)
    command.add_argument("--rate", type=float, default=200.0)
    command.add_argument("--episode_seconds", type=float, default=20.0)
    command.add_argument("--urdf", default="assets/urdf/anymal_d.urdf")
    command = add("neurobem", "NeuroBEM quadrotor (UZH): motor speeds, Vicon state; identifies the propeller model")
    command.add_argument("--max_flights", type=int)
    command.add_argument("--rate", type=float, default=200.0)
    command.add_argument("--source", help="Local processed_data.zip or folder (default: read the remote zip).")
    command.add_argument("--arm_length", type=float, default=0.13)
    command = add("locomujoco", "LocoMuJoCo humanoid motions (robfiras/loco-mujoco-datasets)")
    command.add_argument("--robot", default="UnitreeG1")
    command.add_argument("--collection", default="DefaultDatasets", choices=["DefaultDatasets", "Lafan1"])
    command.add_argument("--max_files", type=int)
    command.add_argument("--urdf", default="assets/urdf/g1_29dof.urdf")
    command.add_argument("--rename", help="JSON file mapping LocoMuJoCo joint names to URDF names.")
    command = add("amass_g1", "AMASS retargeted to Unitree G1 (ember-lab-berkeley/AMASS_Retargeted_for_G1)")
    command.add_argument("--max_files", type=int, default=50)
    command.add_argument("--urdf", default="assets/urdf/g1_29dof.urdf")
    command = add("lafan1", "LAFAN1 retargeted to Unitree G1/H1/H1_2 (lvhaidong/LAFAN1_Retargeting_Dataset)")
    command.add_argument("--robot", default="g1", choices=["g1", "h1", "h1_2"])
    command.add_argument("--max_files", type=int)
    command.add_argument("--urdf")
    arguments = vars(parser.parse_args())
    name = arguments.pop("dataset")
    arguments["patterns"] = arguments.pop("pattern")
    if name == "neurobem":
        arguments["max_flights"] = arguments.pop("max_flights")
    if name == "locomujoco" and arguments.get("rename"):
        arguments["rename"] = json.loads(Path(arguments["rename"]).read_text())
    source = CONVERTERS[name](**arguments)
    print(json.dumps(source, indent=2)[:2000])
    print(f"\nwrote {arguments['output']} (source.json has the data-config source)")


if __name__ == "__main__":
    main()
