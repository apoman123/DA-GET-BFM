"""Convert a public robot dataset to the training table format.

Writes ``<output>/data/chunk-000/file-000.parquet``, ``<output>/meta/info.json``,
and ``<output>/source.json`` (a data-config source you can paste into a data
config, or pass to ``scripts/make_data_config.py --dataset <output>``). Unless
``--no-hf_dataset`` is given, the table is also saved as a Hugging Face dataset in
``<output>/hf`` (``datasets.Dataset.save_to_disk``, opened memory-mapped by the
``huggingface`` source), and ``<output>/sources.json`` lists the source that reads it.

This script converts one dataset with every option of its converter and downloads
through the Hugging Face cache. To download datasets into folders of your choice and
convert them, use ``scripts/download_datasets.py``; to convert what it downloaded,
``scripts/convert_datasets.py``.

The datasets belong to their creators and keep their licenses (MIPO AGPL-3.0;
GrandTour CC BY-SA 4.0 data on its website, MIT on the card; NeuroBEM none stated;
LocoMuJoCo CC BY-NC-ND 4.0; AMASS non-commercial research terms; LAFAN1
CC BY-NC-ND 4.0; xwbc MIT on the card, with the AMASS or LAFAN1 terms). Creators, papers and licenses are listed in
``model/conversion/datasets.py`` and ``docs/data.md#references``.

Examples::

    uv sync --extra convert --extra validation
    uv run python scripts/convert_dataset.py mipo --output data/mipo --pattern "indoor/*trot*" --max_files 10
    uv run python scripts/convert_dataset.py grandtour --output data/grandtour --max_missions 2
    uv run python scripts/convert_dataset.py neurobem --output data/neurobem --max_flights 20
    uv run python scripts/convert_dataset.py locomujoco --output data/loco_g1 --robot UnitreeG1
    uv run python scripts/convert_dataset.py amass_g1 --output data/amass_g1 --pattern "*Walk*" --max_files 100
    uv run python scripts/convert_dataset.py xwbc --output data/xwbc_g1 --archive data/unitree_g1_29dof.tar.gz
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

from model.conversion.common import HF_FOLDER, save_hf_dataset  # noqa: E402
from model.conversion.datasets import CONVERTERS  # noqa: E402


def main() -> None:
    """Parse arguments and run one converter."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="dataset", required=True)

    def add(name: str, help_text: str):
        command = sub.add_parser(name, help=help_text)
        command.add_argument("--output", required=True)
        command.add_argument("--pattern", action="append", help="Glob over file names; repeatable.")
        command.add_argument("--hf_dataset", action=argparse.BooleanOptionalAction, default=True,
                             help="Also save the table as a Hugging Face dataset in <output>/hf (default: on).")
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
    command.add_argument("--urdf", default="assets/urdf/g1_29dof_rev_1_0.urdf")
    command.add_argument("--rename", help="JSON file mapping LocoMuJoCo joint names to URDF names.")
    command = add("amass_g1", "AMASS retargeted to Unitree G1 (fleaven/Retargeted_AMASS_for_robotics)")
    command.add_argument("--max_files", type=int, default=50)
    command.add_argument("--rate", type=float, default=50.0, help="Output rate in Hz (clips are 59-250 Hz).")
    command.add_argument("--ground", default="body", choices=["body", "feet"],
                         help="Points put on the ground: the whole collision model (needs the meshes from "
                              "scripts/download_urdfs.py) or the sole spheres.")
    command.add_argument("--ground_window", type=float, default=1.0,
                         help="Seconds of the ground envelope that removes root-height drift; 0 for one constant per clip.")
    command.add_argument("--root_height", type=float,
                         help="Add this to the relative root z (0.793 as in the upstream visualize.py) instead.")
    command.add_argument("--urdf", default="assets/urdf/g1_29dof_rev_1_0.urdf")
    command = add("xwbc", "AMASS or LAFAN1 retargeted to one humanoid with every joint moving (TonyZhang1999/xwbc-retarget-data)")
    command.add_argument("--robot", default="unitree_g1_29dof")
    command.add_argument("--collection", default="amass", choices=["amass", "lafan"])
    command.add_argument("--max_files", type=int)
    command.add_argument("--archive", help="Local .tar.gz (otherwise it is downloaded, 5-8 GB for AMASS).")
    command.add_argument("--urdf", help="Robot URDF (default for unitree_g1_29dof).")
    command.add_argument("--contact_link", dest="contact_links", action="append", help="Foot link; repeatable.")
    command.add_argument("--ground", default="body", choices=["body", "feet", "none"])
    command.add_argument("--ground_window", type=float, default=1.0)
    command = add("lafan1", "LAFAN1 retargeted to Unitree G1/H1/H1_2 (lvhaidong/LAFAN1_Retargeting_Dataset)")
    command.add_argument("--robot", default="g1", choices=["g1", "h1", "h1_2"])
    command.add_argument("--max_files", type=int)
    command.add_argument("--urdf")
    arguments = vars(parser.parse_args())
    name = arguments.pop("dataset")
    arguments["patterns"] = arguments.pop("pattern")
    hf_dataset = arguments.pop("hf_dataset")
    if name == "neurobem":
        arguments["max_flights"] = arguments.pop("max_flights")
    if name == "locomujoco" and arguments.get("rename"):
        arguments["rename"] = json.loads(Path(arguments["rename"]).read_text())
    source = CONVERTERS[name](**arguments)
    if hf_dataset:
        source = save_hf_dataset(source, Path(arguments["output"]) / HF_FOLDER)
        Path(arguments["output"], "sources.json").write_text(json.dumps([source], indent=2))
    print(json.dumps(source, indent=2)[:2000])
    note = "sources.json reads the Hugging Face copy in hf/; source.json reads the table" if hf_dataset else "source.json has the data-config source"
    print(f"\nwrote {arguments['output']} ({note})")


if __name__ == "__main__":
    main()
