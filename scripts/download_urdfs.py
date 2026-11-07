"""Download the robot URDFs used by the example data configs into assets/urdf.

The files come from Unitree's ``unitree_ros`` repository (BSD-3-Clause). The
examples need only kinematics and inertials, so meshes are not downloaded.

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
