"""Write a suite of procedurally generated legged robots (URDF) and a manifest.

Robot ``i`` of a suite is ``model.morphology.generate_robot(family_i, seed_i)`` with
``seed_i = suite_seeds(seed, count)[i]``, so a ``procedural`` training source with the
same family, count, seed and ranges simulates exactly these robots. The manifest
(``manifest.json``) lists each file with its family, seed, sampled parameters, feet,
nominal pose, mass, size, torque ratio and actuator limits.

The topology, geometry and variation schedule are those of GenBot-1K (Ai, Dai,
Bohlinger, Li, Mu, Wu, Fay, Christensen, Peters and Su, CoRL 2025, PMLR 305:3483-3515,
arXiv:2505.05753) and the mass variation that of GenLoco (Feng et al., CoRL 2022,
PMLR 205:1893-1903, arXiv:2209.05309); ``model.morphology`` cites every value.

Examples::

    uv run python scripts/generate_morphologies.py --family quadruped --count 100 --seed 0 \\
        --output .cache/morphologies/quadrupeds
    uv run python scripts/generate_morphologies.py --family quadruped hexapod humanoid --count 30 \\
        --ranges my_ranges.json --output .cache/morphologies/mixed
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.morphology import FAMILIES, write_robot_suite  # noqa: E402


def main(argv: list[str] | None = None) -> list[Path]:
    """Parse arguments and write the suite.

    Args:
        argv (list[str] | None): Command-line arguments; ``sys.argv`` when None.

    Returns:
        list[Path]: The written URDF files.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--family", nargs="+", choices=FAMILIES, required=True, help="families, cycled over robots")
    parser.add_argument("--count", type=int, required=True, help="number of robots")
    parser.add_argument("--seed", type=int, default=0, help="suite seed")
    parser.add_argument("--output", type=Path, required=True, help="output directory")
    parser.add_argument("--ranges", type=Path, help="JSON file of range overrides (see DEFAULT_RANGES)")
    arguments = parser.parse_args(argv)
    ranges = json.loads(arguments.ranges.read_text()) if arguments.ranges else None
    family = arguments.family[0] if len(arguments.family) == 1 else arguments.family
    paths = write_robot_suite(arguments.output, family, arguments.count, arguments.seed, ranges)
    manifest = json.loads((arguments.output / "manifest.json").read_text())
    for entry in manifest["robots"]:
        print(
            f"{entry['file']}: {entry['family']}, {entry['params']['knee_joints']} knee(s), "
            f"{entry['total_mass']:.1f} kg, L {entry['characteristic_length']:.2f} m, "
            f"torque ratio {entry['torque_ratio']:.1f}"
        )
    print(f"wrote {len(paths)} robots and manifest.json to {arguments.output}")
    return paths


if __name__ == "__main__":
    main()
