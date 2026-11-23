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

References:
    * The layout follows LeRobotDataset v3.0 (``data/chunk-000/file-000.parquet`` and
      ``meta/info.json``): LeRobot, Cadene et al., 2024, https://github.com/huggingface/lerobot
      (Apache-2.0), ``docs/source/lerobot-dataset-v3.mdx``; paper: Cadene et al., ICLR
      2026, arXiv:2602.22818. Only the layout is followed; no LeRobot code is used.
    * Apache Parquet (https://parquet.apache.org/docs/file-format/), written with
      PyArrow (Apache Arrow, https://arrow.apache.org/, Apache-2.0).
    * ``HttpRangeFile`` uses HTTP range requests: RFC 9110 "HTTP Semantics", Sec. 14,
      Fielding, Nottingham & Reschke (eds.), 2022, DOI 10.17487/RFC9110.
    * Quaternions: ``rotation_from_xyzw`` is eq. (B.12) of Lynch & Park, Modern Robotics,
      Cambridge University Press 2017, App. B.3, DOI 10.1017/9781316661239, which also
      notes that ``q`` and ``-q`` give the same rotation (hence the sign continuity in
      ``resample_quaternion``). ``resample_quaternion`` interpolates the components
      linearly and renormalizes, an approximation of slerp (Shoemake, SIGGRAPH 1985,
      DOI 10.1145/325165.325242) that is close for the short steps between samples.
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
    rows_per_group: int = 200_000,
) -> dict[str, Any]:
    """Write episodes as a LeRobot-style (v3.0 layout) parquet table with ``meta/info.json``.

    The episodes are consumed one at a time and written in row groups of about
    ``rows_per_group`` frames, so a generator of episodes is converted without holding the
    whole dataset in memory. The table is written to a temporary name and renamed when
    complete, so an interrupted conversion leaves no file that looks finished.

    Args:
        output (str | Path): Folder to create.
        episodes (Iterable[Episode]): Episodes with consistent signal keys and widths.
        fps (float): Frame rate.
        joint_names (list[str]): Names of the joint columns.
        contact_names (list[str] | None): Names of contact columns (URDF links).
        info (dict | None): Extra provenance stored in ``info.json``.
        rows_per_group (int): Frames buffered before a row group is written.

    Returns:
        dict: The ``info.json`` content.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    output = Path(output)
    (output / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (output / "meta").mkdir(parents=True, exist_ok=True)
    target = output / "data" / "chunk-000" / "file-000.parquet"
    partial = target.with_name(target.name + ".partial")
    keys: list[str] | None = None
    widths: dict[str, int] = {}
    buffer: list[tuple[int, Episode]] = []
    buffered = total = count = 0
    writer = None

    def flush() -> None:
        nonlocal writer, buffer, buffered
        if not buffer:
            return
        lengths = [episode.length for _, episode in buffer]
        arrays = {
            "episode_index": pa.array(np.repeat([index for index, _ in buffer], lengths), type=pa.int64()),
            "frame_index": pa.array(np.concatenate([np.arange(length) for length in lengths]), type=pa.int64()),
            "timestamp": pa.array(np.concatenate([np.arange(length) / fps for length in lengths]), type=pa.float32()),
        }
        for key in keys:
            stacked = np.concatenate([np.asarray(episode.signals[key], dtype=np.float32).reshape(episode.length, -1) for _, episode in buffer])
            arrays[FEATURE_COLUMNS[key]] = pa.FixedSizeListArray.from_arrays(pa.array(stacked.ravel()), widths[key])
        table = pa.table(arrays)
        if writer is None:
            writer = pq.ParquetWriter(partial, table.schema)
        writer.write_table(table)
        buffer, buffered = [], 0

    try:
        for episode in episodes:
            if keys is None:
                keys = sorted(episode.signals)
                widths = {key: int(np.prod(np.shape(episode.signals[key])[1:], dtype=np.int64)) for key in keys}
            if sorted(episode.signals) != keys:
                raise ValueError("every episode must have the same signals")
            for key in keys:
                if int(np.prod(np.shape(episode.signals[key])[1:], dtype=np.int64)) != widths[key]:
                    raise ValueError(f"{key}: every episode must have the same width")
            buffer.append((count, episode))
            buffered += episode.length
            total += episode.length
            count += 1
            if buffered >= rows_per_group:
                flush()
        if not count:
            raise ValueError("no episodes to write")
        flush()
    except BaseException:
        if writer is not None:
            writer.close()
        partial.unlink(missing_ok=True)
        raise
    writer.close()
    partial.replace(target)
    features = {
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
    }
    for key in keys:
        width = widths[key]
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


HF_FOLDER = "hf"
"""Subfolder of a dataset that holds its Hugging Face ``save_to_disk`` copy."""


def save_hf_dataset(
    source: dict[str, Any],
    target: str | Path | None = None,
    base: str | Path | None = None,
    max_shard_size: str = "1GB",
) -> dict[str, Any]:
    """Save the table of a local ``huggingface`` source as a Hugging Face dataset.

    The copy is written with ``datasets.Dataset.save_to_disk`` (Arrow files of at most
    ``max_shard_size``) and keeps only the columns the source reads
    (``model.training.sources.source_columns``). The ``huggingface`` source opens such a
    folder with ``datasets.load_from_disk``: memory-mapped, with no parquet-to-Arrow copy
    in ``HF_DATASETS_CACHE`` (under the home directory by default). The parquet tables stay
    where they are; the intermediate Arrow cache is built next to the target and deleted.

    Args:
        source (dict): Source with ``path`` (a folder of parquet tables, e.g. the
            ``source.json`` of a converter or a downloaded LeRobot dataset) and optional
            ``data_files``.
        target (str | Path | None): Folder to write (default ``<path>/hf``); replaced if
            it exists.
        base (str | Path | None): Directory that a relative ``path`` is relative to.
        max_shard_size (str): Largest Arrow file.

    Returns:
        dict: The same source reading the copy (``path`` set to it, ``data_files``
            removed).
    """
    import glob
    import shutil
    import tempfile

    import pyarrow.parquet as pq
    from datasets import load_dataset

    from model.training.sources import source_columns

    if "path" not in source:
        raise ValueError("only a source with a local path can be saved")
    path = Path(source["path"])
    if not path.is_absolute() and base is not None:
        path = Path(base) / path
    pattern = source.get("data_files") or "data/*/*.parquet"
    patterns = [pattern] if isinstance(pattern, str) else list(pattern)
    files = sorted({name for item in patterns for name in glob.glob(str(item if Path(item).is_absolute() else path / item))})
    if not files:
        raise FileNotFoundError(f"no parquet files match {patterns} in {path}")
    available = set(pq.read_schema(files[0]).names)
    columns = [column for column in source_columns(source) if column in available]
    frame_key = source.get("frame_key", "frame_index")
    missing = sorted(set(source_columns(source)) - available - {frame_key})
    if missing:
        raise ValueError(f"{path}: the source reads columns the table does not have: {missing}")
    target = Path(target) if target is not None else path / HF_FOLDER
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".partial")
    shutil.rmtree(partial, ignore_errors=True)
    with tempfile.TemporaryDirectory(dir=target.parent, prefix=".hf-cache-") as cache:
        table = load_dataset("parquet", data_files=files, split="train", columns=columns, cache_dir=cache)
        table.save_to_disk(str(partial), max_shard_size=max_shard_size)
        del table
    shutil.rmtree(target, ignore_errors=True)
    partial.replace(target)
    saved = {key: value for key, value in source.items() if key not in ("data_files", "repo_id", "revision", "split")}
    saved["path"] = str(target.resolve())
    return saved


class HttpRangeFile(io.RawIOBase):
    """A seekable read-only file over HTTP range requests (RFC 9110, Sec. 14).

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
    """Return world-from-base rotation matrices of (x, y, z, w) quaternions (Lynch & Park eq. B.12)."""
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
