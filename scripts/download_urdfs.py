"""Download the robot URDFs used by the example data configs into assets/urdf.

The robot models are other people's work and are downloaded, not redistributed:

* Go1, H1, G1 and Go2: Unitree Robotics (HangZhou YuShu Technology Co., Ltd.),
  ``unitreerobotics/unitree_ros``, BSD-3-Clause.
* ANYmal D: ANYbotics AG, ``ANYbotics/anymal_d_simple_description``, BSD-3-Clause.
* Talos: PAL Robotics, S.L. The file states "Copyright (c) 2016, PAL Robotics, S.L."
  and the Creative Commons Attribution-NonCommercial-NoDerivs 3.0 license; it is
  distributed in ``Gepetto/example-robot-data`` (LAAS-CNRS, University of
  Edinburgh, INRIA; the repository is BSD-3-Clause).

See ``THIRD_PARTY_NOTICES.md``. The files are used unmodified; the pipeline needs
only kinematics and inertials, so meshes are not downloaded.

Usage::

    uv run python scripts/download_urdfs.py [--output assets/urdf]
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

BASE = "https://raw.githubusercontent.com/unitreerobotics/unitree_ros/master/robots"
URDFS = {
    "go1.urdf": f"{BASE}/go1_description/urdf/go1.urdf",
    "h1.urdf": f"{BASE}/h1_description/urdf/h1.urdf",
    "g1_29dof.urdf": f"{BASE}/g1_description/g1_29dof.urdf",
    # Zero-shot evaluation robot (L4, configs/splits/zero_shot.json); never used in training.
    "go2.urdf": f"{BASE}/go2_description/urdf/go2_description.urdf",
    # Zero-shot evaluation robot (L4): PAL Robotics' Talos (CC BY-NC-ND 3.0) from Gepetto/example-robot-data.
    "talos_reduced.urdf": "https://raw.githubusercontent.com/Gepetto/example-robot-data/master/robots/talos_data/robots/talos_reduced.urdf",
    "anymal_d.urdf": "https://raw.githubusercontent.com/ANYbotics/anymal_d_simple_description/master/urdf/anymal.urdf",
}


def main() -> None:
    """Download every URDF that is not present yet."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", default=str(Path(__file__).resolve().parents[1] / "assets" / "urdf"))
    parser.add_argument("--force", action="store_true", help="Download again even if present.")
    arguments = parser.parse_args()
    output = Path(arguments.output)
    output.mkdir(parents=True, exist_ok=True)
    for name, url in URDFS.items():
        target = output / name
        if target.exists() and not arguments.force:
            print(f"exists  {target}")
            continue
        with urllib.request.urlopen(url, timeout=60) as response:
            target.write_bytes(response.read())
        print(f"fetched {target}")


if __name__ == "__main__":
    main()
