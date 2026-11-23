"""Map a dataset's columns to URDF joints and data-config keys.

``inspect_dataset`` reads ``meta/info.json`` (LeRobot layout, from the Hub or a
local folder) and proposes a ``huggingface`` data-config source:

* joint-state columns, matched to URDF joints by name. Names are split into
  words (``kLeftShoulderPitch`` → left shoulder pitch, ``FL_hip_joint`` → fl
  hip), sides are normalized (l/left, fl/front left, ...), and each URDF
  joint gets the column entry with the highest word overlap. Unitree's
  ``FL0/FL1/FL2`` style is understood as hip, thigh, calf. Columns split by
  body part (``observation.state.left_arm`` with names ``shoulder_pitch``...)
  borrow words from the column name;
* actions with the same names;
* base pose, twist, projected gravity, IMU, and contact columns, from their
  feature names (``x y z qx qy qz qw``, ``gx gy gz wx wy wz``, ...).

The score is the size of the intersection of the two word sets over the size of their
union (the Jaccard index), and joints are assigned greedily by score; the word rules
are this repository's. ``meta/info.json`` is the LeRobot dataset metadata file
(https://github.com/huggingface/lerobot); the naming conventions handled are those of
the datasets in ``docs/data.md`` (for example Unitree's ``kLeftShoulderPitch`` in the
``unitreerobotics/G1_WBT_*`` sets).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

SIDE_WORDS = {
    "l": "left", "r": "right", "lf": "left front", "rf": "right front", "lh": "left hind",
    "rh": "right hind", "fl": "front left", "fr": "front right", "rl": "rear left", "rr": "rear right",
    "hl": "hind left", "hr": "hind right", "bl": "back left", "br": "back right",
}
DROP_WORDS = {"joint", "k", "motor", "dof", "j", "observation", "state", "action", "q"}
UNITREE_LEG_PARTS = {"0": "hip", "1": "thigh", "2": "calf"}
# Feature names that do not match the stored values, by ``robot_type`` prefix. Unitree's G1
# LeRobot sets (every ``G1_WBT_*`` and most ``G1_Dex1_*``; 184 of the 262 ``unitreerobotics``
# repos on 2026-11-23) name the arm entries ``..., elbow, wrist_yaw, wrist_roll, wrist_pitch``,
# but store them in the motor order ``wrist_roll, wrist_pitch, wrist_yaw`` (``G1JointIndex`` of
# unitree_sdk2_python). Checked with forward kinematics against the recorded end-effector poses
# on four repos (docs/data.md#g1-joint-order-check).
KNOWN_NAME_FIXES = {
    "unitree_g1": {
        ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_yaw", "wrist_roll", "wrist_pitch"):
        ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw"],
    },
}


def words(name: str) -> list[str]:
    """Split a joint or column name into normalized words."""
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", name)
    tokens = [token for token in re.split(r"[^A-Za-z0-9]+", text.lower()) if token]
    output: list[str] = []
    for token in tokens:
        match = re.fullmatch(r"(fl|fr|rl|rr|lf|rf|lh|rh)([0-2])", token)
        if match:
            output += SIDE_WORDS[match.group(1)].split() + [UNITREE_LEG_PARTS[match.group(2)]]
            continue
        if token in SIDE_WORDS:
            output += SIDE_WORDS[token].split()
        elif token not in DROP_WORDS:
            output.append(token)
    return output


def _synonyms(tokens: set[str]) -> set[str]:
    aliases = {"rear": "hind", "back": "hind", "hind": "hind", "torso": "waist", "trunk": "waist", "shank": "calf", "knee": "knee"}
    return {aliases.get(token, token) for token in tokens}


def match_score(joint: str, column: str, context: list[str] | None = None) -> float:
    """Word overlap between a URDF joint and a column entry (0 to 1)."""
    a = _synonyms(set(words(joint)))
    b = _synonyms(set(words(column)) | set(context or []))
    if not a or not b:
        return 0.0
    for side in ("left", "right"):
        if (side in a) != (side in b) and ({"left", "right"} & b) and ({"left", "right"} & a):
            return 0.0
    return len(a & b) / len(a | b)


def match_joints(
    urdf_joints: list[str], candidates: list[tuple[str, int, str, list[str]]], threshold: float = 0.4
) -> tuple[dict[str, tuple[str, int]], list[str]]:
    """Assign each URDF joint to a column entry by best word overlap.

    Args:
        urdf_joints (list[str]): Joint names.
        candidates (list): ``(column, index, entry name, context words)``.
        threshold (float): Minimum score.

    Returns:
        tuple: ``{joint: (column, index)}`` and the unmatched joints.
    """
    scores = []
    for joint in urdf_joints:
        for column, index, name, context in candidates:
            score = match_score(joint, name, context)
            if score >= threshold:
                scores.append((score, joint, column, index))
    scores.sort(key=lambda item: -item[0])
    assigned: dict[str, tuple[str, int]] = {}
    used: set[tuple[str, int]] = set()
    for _, joint, column, index in scores:
        if joint in assigned or (column, index) in used:
            continue
        assigned[joint] = (column, index)
        used.add((column, index))
    return assigned, [joint for joint in urdf_joints if joint not in assigned]


def load_info(dataset: str | Path) -> dict[str, Any]:
    """Read ``meta/info.json`` of a local folder or a Hub dataset."""
    path = Path(dataset)
    if path.is_dir():
        return json.loads((path / "meta" / "info.json").read_text())
    from huggingface_hub import hf_hub_download

    return json.loads(Path(hf_hub_download(str(dataset), "meta/info.json", repo_type="dataset")).read_text())


def _names(feature: dict) -> list[str]:
    names = feature.get("names")
    if isinstance(names, dict):
        names = next(iter(names.values()), [])
    if names and isinstance(names[0], list):
        names = names[0]
    return [str(name) for name in names or []]


def corrected_names(info: dict[str, Any], column: str) -> tuple[list[str], bool]:
    """Return the entry names of a column, corrected by ``KNOWN_NAME_FIXES``, and whether they were."""
    names = _names(info["features"][column])
    robot = str(info.get("robot_type") or "")
    for prefix, fixes in KNOWN_NAME_FIXES.items():
        if robot.startswith(prefix) and tuple(names) in fixes:
            return list(fixes[tuple(names)]), True
    return names, False


def _find_vector(features: dict, required: list[str], prefer: tuple[str, ...] = ()) -> tuple[str, list[int]] | None:
    """Find a column whose names contain ``required`` (in any order)."""
    found = []
    for column, feature in features.items():
        names = [name.lower() for name in _names(feature)]
        if all(name in names for name in required):
            bonus = sum(word in column.lower() for word in prefer)
            found.append((bonus, column, [names.index(name) for name in required]))
    if not found:
        return None
    found.sort(key=lambda item: -item[0])
    return found[0][1], found[0][2]


def inspect_dataset(dataset: str | Path, urdf_joints: list[str], threshold: float = 0.4) -> dict[str, Any]:
    """Propose data-config keys for a dataset and a robot.

    Args:
        dataset (str | Path): Local converted folder or Hub repo id.
        urdf_joints (list[str]): Moving joints of the URDF.
        threshold (float): Minimum name-match score.

    Returns:
        dict: ``source`` keys, plus ``report`` lines and ``unmatched`` joints.
    """
    info = load_info(dataset)
    features = info["features"]
    report: list[str] = []

    def state_like(column: str) -> bool:
        lowered = column.lower()
        if not lowered.startswith("observation"):
            return False
        if any(word in lowered for word in ("pose", "base", "imu", "ee", "image", "velocity", "effort", "torque", "contact", "force", "rot", "fig", "hand", "valid", "d435", "torso")):
            return False
        return "state" in lowered or "position" in lowered or "joint" in lowered

    candidates = []
    entry_names: dict[str, list[str]] = {}
    for column, feature in features.items():
        if feature.get("dtype") in ("video", "image") or not state_like(column):
            continue
        context = [word for word in words(column.split(".")[-1]) if word not in {"state"}] if column.count(".") >= 2 else []
        entry_names[column], fixed = corrected_names(info, column)
        if fixed:
            report.append(f"{column}: names {_names(feature)} are stored as {entry_names[column]} (KNOWN_NAME_FIXES)")
        for index, name in enumerate(entry_names[column]):
            candidates.append((column, index, name, context))
    assigned, unmatched = match_joints(urdf_joints, candidates, threshold)
    if not assigned:
        raise ValueError("no joint-state column could be matched to the URDF joints")
    columns: list[str] = []
    for joint in urdf_joints:
        if joint in assigned and assigned[joint][0] not in columns:
            columns.append(assigned[joint][0])
    offsets, total = {}, 0
    for column in columns:
        offsets[column] = total
        total += features[column]["shape"][0]
    joints = [joint for joint in urdf_joints if joint in assigned]
    indices = [offsets[assigned[joint][0]] + assigned[joint][1] for joint in joints]
    source: dict[str, Any] = {
        "joint_names": joints,
        "state_key": columns[0] if len(columns) == 1 else columns,
        "state_indices": indices,
        "fps": info.get("fps"),
    }
    for joint in joints:
        column, index = assigned[joint]
        report.append(f"  {joint:32s} <- {column}[{index}] {entry_names[column][index]!r}")
    action_columns = [column.replace("observation.state", "action") for column in columns]
    if all(column in features for column in action_columns):
        source["action_key"] = action_columns[0] if len(action_columns) == 1 else action_columns
        source["action_indices"] = indices
        source["action_type"] = "position"
        report.append(f"actions: {action_columns} (assumed position targets)")
    elif "action" in features and "observation.state" in columns and features["action"]["shape"] == features["observation.state"]["shape"]:
        source["action_key"] = "action"
        source["action_indices"] = indices
        source["action_type"] = "position"
        report.append("actions: action (assumed position targets)")
    for column, feature in features.items():
        lowered = column.lower()
        names = _names(feature)
        if names == [str(name) for name in urdf_joints if name in names] and not names:
            continue
        if "velocity" in lowered and "base" not in lowered and set(names) >= {name for _, _, name, _ in candidates if _ == columns[0]}:
            source["velocity_key"] = column
        if ("effort" in lowered or "torque" in lowered) and "base" not in lowered:
            source["torque_key"] = column
    pose = _find_vector(features, ["x", "y", "z", "qx", "qy", "qz", "qw"], ("base", "pelvis", "root"))
    if pose:
        source.update(
            base_position_key=pose[0], base_position_indices=pose[1][:3],
            base_orientation_key=pose[0], base_orientation_indices=pose[1][3:], base_orientation_format="xyzw",
        )
        report.append(f"base pose: {pose[0]}")
    else:
        orientation = _find_vector(features, ["qx", "qy", "qz", "qw"], ("base", "orientation", "imu"))
        if orientation:
            source.update(base_orientation_key=orientation[0], base_orientation_indices=orientation[1], base_orientation_format="xyzw")
            report.append(f"base orientation: {orientation[0]}")
        position = _find_vector(features, ["x", "y", "z"], ("base_position", "base", "pelvis"))
        if position and "base_position" in position[0]:
            source.update(base_position_key=position[0], base_position_indices=position[1])
    for key, required, prefer in (
        ("base_angular_velocity", ["wx", "wy", "wz"], ("base", "angular", "gyro", "rot")),
        ("base_linear_velocity", ["vx", "vy", "vz"], ("base", "linear")),
        ("projected_gravity", ["gx", "gy", "gz"], ("gravity", "base", "rot")),
        ("imu_acceleration", ["ax", "ay", "az"], ("imu",)),
    ):
        found = _find_vector(features, required, prefer)
        if found and (key != "imu_acceleration" or "imu" in found[0]):
            source[f"{key}_key"] = found[0]
            source[f"{key}_indices"] = found[1]
            report.append(f"{key.replace('_', ' ')}: {found[0]}")
    for column in features:
        if "contact" in column.lower() and "force" not in column.lower():
            source["contact_key"] = column
            report.append(f"contacts: {column} (one entry per contact link, in order)")
            break
    return {"source": source, "report": report, "unmatched": unmatched}
