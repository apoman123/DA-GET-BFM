"""Retarget a recorded motion from one robot to another (proposal G3).

The motion is a dataset in the repository's table format (``model/conversion/common.py``:
``data/chunk-000/*.parquet`` and ``meta/info.json``, as ``scripts/convert_dataset.py``
writes it), read with the ``huggingface`` data source. Every episode is retargeted with
``model.rl.retarget.retarget`` (keypoints and inverse kinematics after Peng, Coumans,
Zhang, Lee, Tan and Levine, RSS 2020, DOI 10.15607/RSS.2020.XVI.064; damped least
squares after Wampler 1986 and Sugihara 2011, cited in that module) and written in the same format for the target robot, with a ``source.json``
that a data config can use as it is, for example as expert motions of
``scripts/train_bfm.py`` or as prompts for ``DaGetBfmModel.tracking_inference`` and
``goal_inference``.

Example (Go1 motions to ANYmal D)::

    uv run python scripts/retarget_motion.py \\
        --source-urdf assets/urdf/go1.urdf --source-floating-base \\
        --target-urdf assets/urdf/anymal_d.urdf --target-floating-base \\
        --motion data/go1_motions --keypoints configs/go1_to_anymal.json \\
        --target-initial-pose '{"LF_HFE": 0.6, "RF_HFE": 0.6, "LH_HFE": -0.6, "RH_HFE": -0.6,
                                "LF_KFE": -1.0, "RF_KFE": -1.0, "LH_KFE": 1.0, "RH_KFE": 1.0}' \\
        --output data/anymal_from_go1

The keypoint file is described in ``KeypointMap``. ``--scale`` is
``characteristic_length`` (default), ``keypoints``, or a number.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scipy.spatial.transform import Rotation  # noqa: E402

from model.conversion.common import Episode, source_config, write_dataset  # noqa: E402
from model.embodiment import EmbodimentGraph, JointFeatureConfig  # noqa: E402
from model.rl.retarget import KeypointMap, retarget  # noqa: E402
from model.training.sources import SOURCE_TYPES  # noqa: E402
from model.urdf import parse_urdf  # noqa: E402

MOVABLE = ("revolute", "continuous", "prismatic")


def table_source(motion: str | Path, urdf: str | Path, floating_base: bool, max_episodes: int | None = None) -> dict[str, Any]:
    """Return a ``huggingface`` source that reads a dataset in the table format.

    Args:
        motion (str | Path): Dataset folder with ``meta/info.json``.
        urdf (str | Path): URDF of the recorded robot.
        floating_base (bool): The recorded robot's base is free.
        max_episodes (int | None): Read at most this many episodes.

    Returns:
        dict[str, Any]: Data-config source.

    Raises:
        ValueError: The folder has no ``meta/info.json`` or no joint names.
    """
    folder = Path(motion).expanduser().resolve()
    info_path = folder / "meta" / "info.json"
    if not info_path.exists():
        raise ValueError(f"{folder} is not a dataset in the table format (no meta/info.json)")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    features = info["features"]
    names = features.get("observation.state", {}).get("names")
    if not names:
        raise ValueError("meta/info.json names no joints for observation.state")
    source: dict[str, Any] = {
        "type": "huggingface",
        "name": folder.name,
        "path": str(folder),
        "urdf": str(Path(urdf).expanduser().resolve()),
        "joint_names": list(names),
        "fps": float(info["fps"]),
        "floating_base": floating_base,
    }
    if "observation.base_position" in features:
        source["base_position_key"] = "observation.base_position"
    if "observation.base_orientation" in features:
        source["base_orientation_key"] = "observation.base_orientation"
        source["base_orientation_format"] = "xyzw"
    if max_episodes:
        source["max_episodes"] = int(max_episodes)
    return source


def target_graph(urdf: str | Path, floating_base: bool, joints: list[str] | None = None) -> EmbodimentGraph:
    """Build the target robot's graph, keeping only ``joints`` if given (others frozen at 0).

    Args:
        urdf (str | Path): Target URDF.
        floating_base (bool): The target's base is free.
        joints (list[str] | None): Joints to keep, in this order.

    Returns:
        EmbodimentGraph: Target graph.
    """
    robot = parse_urdf(Path(urdf).expanduser())
    frozen = None
    if joints:
        frozen = {name: 0.0 for name, joint in robot.joints.items() if joint.joint_type in MOVABLE and name not in joints}
    return EmbodimentGraph.from_urdf(robot, joint_order=joints, frozen_joints=frozen, floating_base=floating_base)


def _scale(value: str) -> str | float:
    """Parse ``--scale``: a method name or a number."""
    try:
        return float(value)
    except ValueError:
        return value


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv (list[str] | None): Arguments; ``sys.argv`` by default.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-urdf", required=True, help="URDF of the recorded robot.")
    parser.add_argument("--target-urdf", required=True, help="URDF of the robot to retarget to.")
    parser.add_argument("--motion", required=True, help="Dataset folder in the table format (meta/info.json).")
    parser.add_argument("--keypoints", required=True, help="Keypoint map JSON (see model.rl.retarget.KeypointMap).")
    parser.add_argument("--output", required=True, help="Folder for the retargeted dataset.")
    parser.add_argument("--source-floating-base", action="store_true", help="The recorded robot's base is free.")
    parser.add_argument("--target-floating-base", action="store_true", help="The target robot's base is free.")
    parser.add_argument("--target-joints", nargs="+", help="Target joints to keep; others are frozen at 0.")
    parser.add_argument("--target-initial-pose", help="JSON object or .json file: target joint name to start position.")
    parser.add_argument("--target-contact-links", nargs="+", help="Target feet, written to source.json.")
    parser.add_argument("--scale", type=_scale, default="characteristic_length", help="characteristic_length, keypoints, or a number.")
    parser.add_argument("--rest-weight", type=float, default=0.0, help="Weight of Peng et al.'s default-pose term (0: off).")
    parser.add_argument("--max-episodes", type=int, help="Retarget at most this many episodes.")
    parser.add_argument("--name", help="Name of the written source (the target URDF stem by default).")
    return parser.parse_args(argv)


def _json_argument(value: str | None) -> Any:
    """Parse a JSON string or the content of a JSON file."""
    if value is None:
        return None
    path = Path(value)
    return json.loads(path.read_text(encoding="utf-8") if path.suffix == ".json" and path.exists() else value)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    """Retarget every episode and write the target dataset.

    Args:
        argv (list[str] | None): Command-line arguments; ``sys.argv`` by default.

    Returns:
        dict[str, Any]: The written ``source.json`` content plus a ``retargeting``
            summary (scale and the largest keypoint error per episode).
    """
    arguments = parse_arguments(argv)
    keypoints = KeypointMap.from_json(arguments.keypoints)
    source = table_source(arguments.motion, arguments.source_urdf, arguments.source_floating_base, arguments.max_episodes)
    records, trajectories = SOURCE_TYPES["huggingface"](source, None, JointFeatureConfig())
    source_graph = records[0].graph
    graph = target_graph(arguments.target_urdf, arguments.target_floating_base, arguments.target_joints)
    initial_pose = _json_argument(arguments.target_initial_pose)
    episodes, errors, scale = [], [], None
    for index, trajectory in enumerate(trajectories):
        result = retarget(
            source_graph, trajectory, graph, keypoints,
            scale=arguments.scale, initial_pose=initial_pose, rest_weight=arguments.rest_weight,
        )
        scale = result.scale
        signals = {"state": result.positions.numpy()}
        if result.base_position is not None:
            signals["base_position"] = result.base_position.numpy()
            signals["base_orientation"] = Rotation.from_matrix(result.base_orientation.numpy()).as_quat()
        episodes.append(Episode(signals, {"source_episode": index}))
        errors.append(float(result.keypoint_error.max()))
        print(f"episode {index}: {result.positions.shape[0]} frames, largest keypoint error {errors[-1]:.4f} m")
    summary = {
        "source_urdf": str(Path(arguments.source_urdf).resolve()),
        "source_motion": str(Path(arguments.motion).resolve()),
        "keypoints": keypoints.to_dict(),
        "scale": scale,
        "max_keypoint_error": errors,
    }
    info = write_dataset(
        arguments.output, episodes, float(source["fps"]), list(graph.joint_names), info={"retargeting": summary}
    )
    name = arguments.name or Path(arguments.target_urdf).stem
    written = source_config(
        arguments.output, info, arguments.target_urdf, name, arguments.target_floating_base, arguments.target_contact_links
    )
    print(f"scale {scale:.4f}; wrote {len(episodes)} episodes to {arguments.output}")
    return {**written, "retargeting": summary}


if __name__ == "__main__":
    main()
