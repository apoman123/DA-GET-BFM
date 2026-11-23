"""Download the robot URDFs used by the example data configs into assets/urdf.

The robot models are other people's work and are downloaded, not redistributed:

* Go1, H1, G1 and Go2: Unitree Robotics (HangZhou YuShu Technology Co., Ltd.),
  ``unitreerobotics/unitree_ros``, BSD-3-Clause. The G1 is ``g1_29dof_rev_1_0.urdf``, the
  model of the robots in Unitree's G1 datasets and of the retargeted motions
  (``docs/data.md#g1-model``).
* ANYmal D: ANYbotics AG, ``ANYbotics/anymal_d_simple_description``, BSD-3-Clause.
* Talos: PAL Robotics, S.L. The file states "Copyright (c) 2016, PAL Robotics, S.L."
  and the Creative Commons Attribution-NonCommercial-NoDerivs 3.0 license; it is
  distributed in ``Gepetto/example-robot-data`` (LAAS-CNRS, University of
  Edinburgh, INRIA; the repository is BSD-3-Clause).

See ``THIRD_PARTY_NOTICES.md``. The files are used unmodified. The pipeline needs only
kinematics and inertials, except that ``convert_amass_g1`` and ``convert_xwbc`` put each
motion on the ground with the whole-body collision geometry, so the G1 collision meshes (the
``<collision>`` meshes of ``g1_29dof_rev_1_0.urdf``, about 11 MB) are downloaded too
(``--no_meshes`` skips them).

Usage::

    uv run python scripts/download_urdfs.py [--output assets/urdf] [--no_meshes]
"""

from __future__ import annotations

import argparse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path, PurePosixPath

BASE = "https://raw.githubusercontent.com/unitreerobotics/unitree_ros/master/robots"
URDFS = {
    "go1.urdf": f"{BASE}/go1_description/urdf/go1.urdf",
    "h1.urdf": f"{BASE}/h1_description/urdf/h1.urdf",
    "g1_29dof_rev_1_0.urdf": f"{BASE}/g1_description/g1_29dof_rev_1_0.urdf",
    # Zero-shot evaluation robot (L4, configs/splits/zero_shot.json); never used in training.
    "go2.urdf": f"{BASE}/go2_description/urdf/go2_description.urdf",
    # Zero-shot evaluation robot (L4): PAL Robotics' Talos (CC BY-NC-ND 3.0) from Gepetto/example-robot-data.
    "talos_reduced.urdf": "https://raw.githubusercontent.com/Gepetto/example-robot-data/master/robots/talos_data/robots/talos_reduced.urdf",
    "anymal_d.urdf": "https://raw.githubusercontent.com/ANYbotics/anymal_d_simple_description/master/urdf/anymal.urdf",
}
# URDFs whose collision meshes are downloaded, with the folder their relative mesh paths start from.
MESH_FOLDERS = {"g1_29dof_rev_1_0.urdf": f"{BASE}/g1_description"}


def collision_meshes(urdf: Path) -> list[str]:
    """Return the relative mesh paths of a URDF's ``<collision>`` elements."""
    names = set()
    for collision in ET.parse(urdf).getroot().iter("collision"):
        for mesh in collision.iter("mesh"):
            name = mesh.get("filename", "")
            path = PurePosixPath(name)
            if name and "://" not in name and not path.is_absolute() and ".." not in path.parts:
                names.add(name)
    return sorted(names)


def fetch(url: str, target: Path, force: bool) -> None:
    if target.exists() and not force:
        print(f"exists  {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=60) as response:
        target.write_bytes(response.read())
    print(f"fetched {target}")


def main() -> None:
    """Download every URDF (and the G1 collision meshes) that is not present yet."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", default=str(Path(__file__).resolve().parents[1] / "assets" / "urdf"))
    parser.add_argument("--force", action="store_true", help="Download again even if present.")
    parser.add_argument("--no_meshes", action="store_true", help="Skip the G1 collision meshes.")
    arguments = parser.parse_args()
    output = Path(arguments.output)
    output.mkdir(parents=True, exist_ok=True)
    for name, url in URDFS.items():
        fetch(url, output / name, arguments.force)
    if arguments.no_meshes:
        return
    for name, folder in MESH_FOLDERS.items():
        for mesh in collision_meshes(output / name):
            fetch(f"{folder}/{mesh}", output / mesh, arguments.force)


if __name__ == "__main__":
    main()
