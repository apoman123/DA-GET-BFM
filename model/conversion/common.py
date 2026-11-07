"""Shared tools for converting robot datasets into the training table format.

Every converter writes the same layout, which the ``huggingface`` data source
reads with ``"path"`` and ``scripts/make_data_config.py`` understands::

    <output>/
      data/chunk-000/file-000.parquet   # one row per frame
      meta/info.json                    # fps, features with names, provenance
      source.json                       # a ready-to-use data-config source

Columns (all optional except the indices and ``observation.state``):

========================================  ==========================================
``episode_index``, ``frame_index``         int
``timestamp``                              s since the episode start
``observation.state``                      joint positions, URDF joint names
``observation.velocity``                   joint velocities
``observation.effort``                     measured joint torques (N m)
``action``                                 commands
``observation.base_position``              base origin in the world (m)
``observation.base_orientation``           world-from-base quaternion (x, y, z, w)
``observation.base_linear_velocity``       base frame (m/s)
``observation.base_angular_velocity``      base frame (rad/s)
``observation.base_linear_acceleration``   base frame, classical, no gravity (m/s^2)
``observation.imu_acceleration``           accelerometer specific force, base frame
``observation.projected_gravity``          unit gravity direction, base frame
``observation.contact``                    per-foot contact flag (0 or 1)
``observation.contact_force``              per-foot force magnitude (N)
========================================  ==========================================
"""

from __future__ import annotations

import io
import json
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

FEATURE_COLUMNS = {
    "state": "observation.state",
    "velocity": "observation.velocity",
    "effort": "observation.effort",
    "action": "action",
    "base_position": "observation.base_position",
    "base_orientation": "observation.base_orientation",
    "base_linear_velocity": "observation.base_linear_velocity",
    "base_angular_velocity": "observation.base_angular_velocity",
    "base_linear_acceleration": "observation.base_linear_acceleration",
    "imu_acceleration": "observation.imu_acceleration",
    "projected_gravity": "observation.projected_gravity",
    "contact": "observation.contact",
    "contact_force": "observation.contact_force",
}
VECTOR_NAMES = {
    "base_position": ["x", "y", "z"],
    "base_orientation": ["qx", "qy", "qz", "qw"],
    "base_linear_velocity": ["vx", "vy", "vz"],
    "base_angular_velocity": ["wx", "wy", "wz"],
    "base_linear_acceleration": ["ax", "ay", "az"],
    "imu_acceleration": ["ax", "ay", "az"],
    "projected_gravity": ["gx", "gy", "gz"],
}


@dataclass
class Episode:
    """One episode of converted signals, all ``[T, ...]`` arrays at a common rate.

    Attributes:
        signals (dict[str, np.ndarray]): Keys from ``FEATURE_COLUMNS``.
        meta (dict): Free-form provenance (source file, segment).
    """

    signals: dict[str, np.ndarray]
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def length(self) -> int:
        """Return the number of frames."""
        return len(next(iter(self.signals.values())))


def resample(time: np.ndarray, values: np.ndarray, new_time: np.ndarray) -> np.ndarray:
    """Linearly interpolate ``[N, D]`` values onto new timestamps."""
    values = np.asarray(values, dtype=np.float64)
    flat = values.reshape(len(values), -1)
    output = np.stack([np.interp(new_time, time, flat[:, column]) for column in range(flat.shape[1])], axis=1)
    return output.reshape(len(new_time), *values.shape[1:])


def resample_quaternion(time: np.ndarray, quaternion: np.ndarray, new_time: np.ndarray) -> np.ndarray:
    """Interpolate quaternions (any component order) with sign continuity."""
    quaternion = np.asarray(quaternion, dtype=np.float64).copy()
    for index in range(1, len(quaternion)):
        if np.dot(quaternion[index], quaternion[index - 1]) < 0:
            quaternion[index] = -quaternion[index]
    output = resample(time, quaternion, new_time)
    return output / np.linalg.norm(output, axis=1, keepdims=True).clip(1e-12)


def uniform_time(start: float, stop: float, rate: float) -> np.ndarray:
    """Return timestamps from start to stop at a fixed rate."""
    count = int(np.floor((stop - start) * rate)) + 1
    return start + np.arange(count) / rate


def split_segments(time: np.ndarray, max_gap: float, min_length: float) -> list[tuple[int, int]]:
    """Split a time series at gaps longer than ``max_gap`` seconds.

    Returns:
        list[tuple[int, int]]: ``(start, stop)`` index ranges at least
            ``min_length`` seconds long.
    """
    breaks = np.nonzero(np.diff(time) > max_gap)[0] + 1
    edges = [0, *breaks.tolist(), len(time)]
    return [
        (start, stop)
        for start, stop in zip(edges[:-1], edges[1:])
        if stop - start > 1 and time[stop - 1] - time[start] >= min_length
    ]


def chunk_episode(episode: Episode, frames: int | None) -> list[Episode]:
    """Cut an episode into pieces of at most ``frames`` frames."""
    if not frames or episode.length <= frames:
        return [episode]
    pieces = []
    for start in range(0, episode.length - frames // 2, frames):
        stop = min(start + frames, episode.length)
        pieces.append(
            Episode(
                {key: value[start:stop] for key, value in episode.signals.items()},
                {**episode.meta, "frames": [start, stop]},
            )
        )
    return pieces


def write_dataset(
    output: str | Path,
    episodes: Iterable[Episode],
    fps: float,
    joint_names: list[str],
    contact_names: list[str] | None = None,
    info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write episodes as a LeRobot-style parquet table with ``meta/info.json``.

    Args:
        output (str | Path): Folder to create.
        episodes (Iterable[Episode]): Episodes with consistent signal keys.
        fps (float): Frame rate.
        joint_names (list[str]): Names of the joint columns.
        contact_names (list[str] | None): Names of contact columns (URDF links).
        info (dict | None): Extra provenance stored in ``info.json``.

    Returns:
        dict: The ``info.json`` content.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    output = Path(output)
    (output / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (output / "meta").mkdir(parents=True, exist_ok=True)
    columns: dict[str, list] = {"episode_index": [], "frame_index": [], "timestamp": []}
    keys: list[str] | None = None
    total = 0
    count = 0
    for index, episode in enumerate(episodes):
        if keys is None:
            keys = sorted(episode.signals)
            for key in keys:
                columns[FEATURE_COLUMNS[key]] = []
        if sorted(episode.signals) != keys:
            raise ValueError("every episode must have the same signals")
        length = episode.length
        columns["episode_index"] += [index] * length
        columns["frame_index"] += list(range(length))
        columns["timestamp"] += (np.arange(length) / fps).tolist()
        for key in keys:
            values = np.asarray(episode.signals[key], dtype=np.float32).reshape(length, -1)
            columns[FEATURE_COLUMNS[key]].append(values)
        total += length
        count += 1
    if not count:
        raise ValueError("no episodes to write")
    arrays = {}
    for name, values in columns.items():
        if name in ("episode_index", "frame_index"):
            arrays[name] = pa.array(values, type=pa.int64())
        elif name == "timestamp":
            arrays[name] = pa.array(values, type=pa.float32())
        else:
            stacked = np.concatenate(values, axis=0)
            arrays[name] = pa.FixedSizeListArray.from_arrays(pa.array(stacked.ravel()), stacked.shape[1])
    pq.write_table(pa.table(arrays), output / "data" / "chunk-000" / "file-000.parquet")
    features = {
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
    }
    for key in keys:
        width = arrays[FEATURE_COLUMNS[key]].type.list_size
        if key in ("state", "velocity", "effort", "action"):
            names = list(joint_names)
        elif key in ("contact", "contact_force"):
            names = list(contact_names or [])
        else:
            names = VECTOR_NAMES[key]
        features[FEATURE_COLUMNS[key]] = {"dtype": "float32", "shape": [width], "names": names}
    content = {
        "codebase_version": "da-get-bfm-1",
        "fps": fps,
        "total_episodes": count,
        "total_frames": total,
        "features": features,
        **(info or {}),
    }
    (output / "meta" / "info.json").write_text(json.dumps(content, indent=2))
    return content


def source_config(
    output: str | Path,
    info: dict[str, Any],
    urdf: str,
    name: str,
    floating_base: bool,
    contact_links: list[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build and save a ``huggingface`` data-config source for a converted dataset.

    Args:
        output (str | Path): Converted dataset folder; ``source.json`` is written there.
        info (dict): ``info.json`` content from ``write_dataset``.
        urdf (str): URDF path to put in the source (edit it if needed).
        name (str): Source name.
        floating_base (bool): Whether the robot's base is free.
        contact_links (list[str] | None): Feet for contact handling.
        extra (dict | None): More source keys (actuator overrides, ...).

    Returns:
        dict: The source.
    """
    features = info["features"]
    joint_names = features["observation.state"]["names"]
    if Path(urdf).exists():
        urdf = str(Path(urdf).resolve())
    source: dict[str, Any] = {
        "type": "huggingface",
        "name": name,
        "path": str(Path(output).resolve()),
        "urdf": urdf,
        "joint_names": joint_names,
        "state_key": "observation.state",
        "fps": info["fps"],
        "floating_base": floating_base,
    }
    optional = {
        "observation.velocity": "velocity_key",
        "observation.effort": "torque_key",
        "action": "action_key",
        "observation.base_position": "base_position_key",
        "observation.base_orientation": "base_orientation_key",
        "observation.base_linear_velocity": "base_linear_velocity_key",
        "observation.base_angular_velocity": "base_angular_velocity_key",
        "observation.imu_acceleration": "imu_acceleration_key",
        "observation.projected_gravity": "projected_gravity_key",
        "observation.contact": "contact_key",
    }
    for column, key in optional.items():
        if column in features:
            source[key] = column
    if "observation.base_orientation" in features:
        source["base_orientation_format"] = "xyzw"
    if contact_links:
        source["contact_links"] = contact_links
    source.update(extra or {})
    Path(output, "source.json").write_text(json.dumps(source, indent=2))
    return source


class HttpRangeFile(io.RawIOBase):
    """A seekable read-only file over HTTP range requests.

    Lets ``zipfile`` read single members of a large remote archive (for
    example one NeuroBEM flight out of a 627 MB zip) without downloading all
    of it.

    Args:
        url (str): File URL; the server must support ``Range`` requests.
        block (int): Bytes fetched per request.
    """

    def __init__(self, url: str, block: int = 1 << 20) -> None:
        super().__init__()
        self.url = url
        self.block = block
        request = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(request, timeout=60) as response:
            self.size = int(response.headers["Content-Length"])
        self.position = 0
        self.cache: dict[int, bytes] = {}

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = 0) -> int:
        base = {0: 0, 1: self.position, 2: self.size}[whence]
        self.position = max(0, min(self.size, base + offset))
        return self.position

    def _fetch(self, index: int) -> bytes:
        if index not in self.cache:
            start = index * self.block
            stop = min(start + self.block, self.size) - 1
            request = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{stop}"})
            with urllib.request.urlopen(request, timeout=120) as response:
                self.cache[index] = response.read()
            if len(self.cache) > 64:
                self.cache.pop(next(iter(self.cache)))
        return self.cache[index]

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            size = self.size - self.position
        size = min(size, self.size - self.position)
        chunks = []
        while size > 0:
            index, offset = divmod(self.position, self.block)
            data = self._fetch(index)[offset : offset + size]
            chunks.append(data)
            self.position += len(data)
            size -= len(data)
        return b"".join(chunks)

    def readinto(self, buffer) -> int:
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)


def quaternion_wxyz_to_xyzw(quaternion: np.ndarray) -> np.ndarray:
    """Reorder quaternions from (w, x, y, z) to (x, y, z, w)."""
    return np.asarray(quaternion)[..., [1, 2, 3, 0]]


def rotation_from_xyzw(quaternion: np.ndarray) -> np.ndarray:
    """Return world-from-base rotation matrices of (x, y, z, w) quaternions."""
    x, y, z, w = (np.asarray(quaternion, dtype=np.float64) / np.linalg.norm(quaternion, axis=-1, keepdims=True)).T
    return np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
            np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
            np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
        ],
        -2,
    )


def to_base_frame(quaternion_xyzw: np.ndarray, world_vectors: np.ndarray) -> np.ndarray:
    """Rotate world-frame vectors into the base frame."""
    rotation = rotation_from_xyzw(quaternion_xyzw)
    return np.einsum("nji,nj->ni", rotation, world_vectors)
