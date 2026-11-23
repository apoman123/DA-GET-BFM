"""Processed physics episodes on disk as Hugging Face ``datasets`` Arrow files.

``build_physics_datasets`` processes every trajectory once (``process_episode``) and
writes the result frame by frame to memory-mapped Arrow files, so the training set can
be larger than the RAM: only the frames a batch needs are read. A cache entry
``<cache_dir>/<key>/`` holds:

* ``frames-00000.arrow``, ``frames-00001.arrow``, ...: one row per frame, one list column
  per per-frame tensor of ``process_episode`` (flattened; shapes are in
  ``episodes.json``). Each file is a Hugging Face dataset:
  ``datasets.Dataset.from_file(path)`` opens it, ``ProcessedEpisodes.dataset()`` all of
  them;
* ``episodes.json``: for every episode its embodiment, shard, first row, number of
  frames, tensor layout and scalars (``dt``);
* ``records.pt``: the embodiment records, including the data scales of
  ``finalize_acceleration_scales``;
* ``complete``: written last; an entry without it is rebuilt.

``MemoryEpisodes`` gives the same interface over episodes held in memory (the physics
replay of RL, and tests).

References:
    * Hugging Face ``datasets`` (memory-mapped Arrow tables, ``Dataset.from_file``):
      Lhoest et al., "Datasets: A Community Library for Natural Language Processing",
      EMNLP 2021 System Demonstrations, pp. 175-184, DOI 10.18653/v1/2021.emnlp-demo.21;
      https://github.com/huggingface/datasets (Apache-2.0).
    * Apache Arrow columnar format and IPC streaming format, written with PyArrow:
      https://arrow.apache.org/docs/format/Columnar.html (Apache-2.0). No code of either
      project is copied.
"""

from __future__ import annotations

import json
import shutil
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch

FORMAT = "da-get-bfm-frames-1"
SHARD_BYTES = 1 << 30
"""Start a new Arrow file after about this many bytes (also keeps list offsets in int32)."""
LAZY_CACHE = 8
"""Episodes whose loaded columns are kept by ``ProcessedEpisodes.__getitem__``."""

_ARROW_TYPES = {
    "torch.float16": "float16",
    "torch.float32": "float32",
    "torch.float64": "float64",
    "torch.bool": "bool_",
    "torch.int8": "int8",
    "torch.int16": "int16",
    "torch.int32": "int32",
    "torch.int64": "int64",
    "torch.uint8": "uint8",
}
_TORCH_TYPES = {name: getattr(torch, name.split(".")[1]) for name in _ARROW_TYPES}


def _split_episode(episode: Mapping[str, torch.Tensor]) -> tuple[int, dict[str, torch.Tensor], dict[str, float]]:
    """Separate per-frame tensors ``[T, ...]`` from scalars; statistics (``_*``) are dropped."""
    steps = int(episode["applied_torque"].shape[0])
    frames, scalars = {}, {}
    for key, value in episode.items():
        if key.startswith("_") or not torch.is_tensor(value):
            continue
        if value.dim() == 0:
            scalars[key] = float(value)
            continue
        if value.shape[0] != steps:
            raise ValueError(f"episode key {key!r} has {value.shape[0]} frames, expected {steps}")
        frames[key] = value
    return steps, frames, scalars


class EpisodeWriter:
    """Write processed episodes to a new cache entry, one Arrow row per frame.

    Args:
        path (str | Path): Folder of the entry; an existing folder is replaced.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if self.path.exists():
            shutil.rmtree(self.path)
        self.path.mkdir(parents=True)
        self.episodes: list[dict[str, Any]] = []
        self.layouts: list[dict[str, Any]] = []
        self._layout_ids: dict[tuple, int] = {}
        self._writer = None
        self._sink = None
        self._schema = None
        self._shard = -1
        self._rows = 0
        self._bytes = 0

    def _open_shard(self, schema) -> None:
        """Close the current file and start the next one."""
        import pyarrow as pa

        self._close_shard()
        self._shard += 1
        self._sink = pa.OSFile(str(self.path / f"frames-{self._shard:05d}.arrow"), "wb")
        self._writer = pa.ipc.new_stream(self._sink, schema)
        self._schema = schema
        self._rows = 0
        self._bytes = 0

    def _close_shard(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._sink.close()
        self._writer = self._sink = None

    def add(self, embodiment: int, episode: Mapping[str, torch.Tensor]) -> None:
        """Append one processed episode.

        Args:
            embodiment (int): Index of its record.
            episode (Mapping[str, torch.Tensor]): ``process_episode`` output.
        """
        import pyarrow as pa

        steps, frames, scalars = _split_episode(episode)
        keys = sorted(frames)
        layout = {key: {"shape": list(frames[key].shape[1:]), "dtype": str(frames[key].dtype)} for key in keys}
        signature = tuple((key, tuple(entry["shape"]), entry["dtype"]) for key, entry in layout.items())
        if signature not in self._layout_ids:
            self._layout_ids[signature] = len(self.layouts)
            self.layouts.append(layout)
        columns = {}
        for key in keys:
            value = frames[key].detach().cpu().contiguous()
            dtype = str(value.dtype)
            if dtype not in _ARROW_TYPES:
                raise TypeError(f"episode key {key!r} has unsupported dtype {dtype}")
            width = int(np.prod(value.shape[1:], dtype=np.int64))
            flat = value.reshape(steps * width).numpy()
            offsets = pa.array(np.arange(steps + 1, dtype=np.int32) * width, type=pa.int32())
            columns[key] = pa.ListArray.from_arrays(offsets, pa.array(flat, type=getattr(pa, _ARROW_TYPES[dtype])()))
        table = pa.table(columns)
        if self._writer is None or table.schema != self._schema or (self._rows and self._bytes + table.nbytes > SHARD_BYTES):
            self._open_shard(table.schema)
        self._writer.write_table(table)
        self.episodes.append(
            {
                "embodiment": int(embodiment),
                "shard": self._shard,
                "row": self._rows,
                "steps": steps,
                "variants": int(episode["applied_torque"].shape[1]),
                "layout": self._layout_ids[signature],
                "scalars": scalars,
            }
        )
        self._rows += steps
        self._bytes += table.nbytes

    def close(self, records: Sequence[Any]) -> None:
        """Finish the files, save the records and mark the entry complete.

        Args:
            records (Sequence): Embodiment records the episodes index.
        """
        self._close_shard()
        torch.save(list(records), self.path / "records.pt")
        meta = {"format": FORMAT, "shards": self._shard + 1, "layouts": self.layouts, "episodes": self.episodes}
        (self.path / "episodes.json").write_text(json.dumps(meta))
        (self.path / "complete").write_text(FORMAT)


def entry_complete(path: str | Path) -> bool:
    """Return whether a cache entry was fully written in the current format."""
    marker = Path(path) / "complete"
    return marker.exists() and marker.read_text().strip() == FORMAT


class _Shards:
    """Memory-mapped Arrow tables of a cache entry.

    They are mapped when the entry is opened, so a view keeps working if the cache folder
    is removed later (on POSIX systems); after pickling (a DataLoader worker started with
    ``spawn``) they are mapped again on first use.
    """

    def __init__(self, path: Path, count: int) -> None:
        self.path, self.count = path, count
        self._tables: list | None = None
        self.table(0) if count else None

    def __getstate__(self) -> dict[str, Any]:
        return {"path": self.path, "count": self.count, "_tables": None}

    def table(self, shard: int):
        """Return the PyArrow table of one shard."""
        if self._tables is None:
            from datasets import Dataset

            self._tables = [Dataset.from_file(str(self.path / f"frames-{index:05d}.arrow")).data.table for index in range(self.count)]
        return self._tables[shard]


class LazyEpisode(Mapping):
    """Read-only mapping of one stored episode whose columns load on first access."""

    def __init__(self, store: "ProcessedEpisodes", episode: int) -> None:
        self._store, self._episode = store, episode
        self._loaded: dict[str, torch.Tensor] = {}

    def __getitem__(self, key: str) -> torch.Tensor:
        if key not in self._loaded:
            scalars = self._store._meta(self._episode)["scalars"]
            if key in scalars:
                self._loaded[key] = torch.tensor(scalars[key], dtype=torch.float32)
            elif key in self._store._layout(self._episode):
                self._loaded[key] = self._store._read(self._episode, 0, self._store._meta(self._episode)["steps"], [key])[key]
            else:
                raise KeyError(key)
        return self._loaded[key]

    def __iter__(self) -> Iterator[str]:
        yield from self._store._layout(self._episode)
        yield from self._store._meta(self._episode)["scalars"]

    def __len__(self) -> int:
        return len(self._store._layout(self._episode)) + len(self._store._meta(self._episode)["scalars"])


class ProcessedEpisodes(Sequence):
    """Episodes of a cache entry, optionally a subset (train or evaluation split).

    Indexing gives ``(embodiment, episode)`` pairs like a list of processed episodes;
    the episode is a ``LazyEpisode``. Training reads frames with ``frames``.

    Args:
        path (str | Path): Complete cache entry.
        ids (Sequence[int] | None): Episodes of the entry in this view; all by default.
    """

    def __init__(self, path: str | Path, ids: Sequence[int] | None = None, _shared: tuple | None = None) -> None:
        self.path = Path(path)
        if _shared is None:
            if not entry_complete(self.path):
                raise FileNotFoundError(f"{self.path} is not a complete processed-episode entry")
            meta = json.loads((self.path / "episodes.json").read_text())
            _shared = (meta, _Shards(self.path, int(meta["shards"])))
        self._meta_all, self._shards = _shared
        self.ids = list(range(len(self._meta_all["episodes"]))) if ids is None else [int(index) for index in ids]
        self._lazy: OrderedDict[int, LazyEpisode] = OrderedDict()

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_lazy"] = OrderedDict()
        return state

    def records(self) -> list[Any]:
        """Return the saved embodiment records."""
        return torch.load(self.path / "records.pt", weights_only=False)

    def subset(self, ids: Sequence[int]) -> "ProcessedEpisodes":
        """Return a view of some episodes of this view (indices relative to it)."""
        return ProcessedEpisodes(self.path, [self.ids[int(index)] for index in ids], (self._meta_all, self._shards))

    def _meta(self, index: int) -> dict[str, Any]:
        return self._meta_all["episodes"][self.ids[index]]

    def _layout(self, index: int) -> dict[str, Any]:
        return self._meta_all["layouts"][self._meta(index)["layout"]]

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[position] for position in range(*index.indices(len(self)))]
        index = int(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        episode = self._lazy.pop(index, None) or LazyEpisode(self, index)
        self._lazy[index] = episode
        while len(self._lazy) > LAZY_CACHE:
            self._lazy.popitem(last=False)
        return self.embodiment(index), episode

    def embodiment(self, index: int) -> int:
        """Return the record index of an episode."""
        return int(self._meta(index)["embodiment"])

    def steps(self, index: int) -> int:
        """Return the number of frames of an episode."""
        return int(self._meta(index)["steps"])

    def variants(self, index: int) -> int:
        """Return the number of action variants per frame."""
        return int(self._meta(index)["variants"])

    def keys(self, index: int) -> set[str]:
        """Return the tensor and scalar keys of an episode."""
        return set(self._layout(index)) | set(self._meta(index)["scalars"])

    def scalar(self, index: int, key: str, default: float | None = None) -> float | None:
        """Return a scalar of an episode (``dt``)."""
        return self._meta(index)["scalars"].get(key, default)

    def frames(self, index: int, start: int, stop: int, keys: Sequence[str]) -> dict[str, torch.Tensor]:
        """Read frames ``start:stop`` of an episode for the given keys.

        Args:
            index (int): Episode in this view.
            start (int): First frame.
            stop (int): Frame after the last.
            keys (Sequence[str]): Per-frame keys to read.

        Returns:
            dict[str, torch.Tensor]: ``[stop - start, ...]`` tensors.
        """
        return self._read(index, start, stop, keys)

    def _read(self, index: int, start: int, stop: int, keys: Sequence[str]) -> dict[str, torch.Tensor]:
        meta, layout = self._meta(index), self._layout(index)
        table = self._shards.table(meta["shard"])
        count = stop - start
        output = {}
        for key in keys:
            entry = layout[key]
            column = table.column(key).slice(meta["row"] + start, count)
            values = (column.chunk(0) if column.num_chunks == 1 else column.combine_chunks()).flatten()
            array = np.array(values.to_numpy(zero_copy_only=False), copy=True)
            output[key] = torch.from_numpy(array).to(_TORCH_TYPES[entry["dtype"]]).reshape(count, *entry["shape"])
        return output

    def dataset(self):
        """Return every frame of the entry as one ``datasets.Dataset`` (shards concatenated)."""
        from datasets import Dataset, concatenate_datasets

        shards = [Dataset.from_file(str(self.path / f"frames-{index:05d}.arrow")) for index in range(self._shards.count)]
        return shards[0] if len(shards) == 1 else concatenate_datasets(shards)


class MemoryEpisodes(Sequence):
    """The ``ProcessedEpisodes`` interface over ``(embodiment, episode)`` pairs in memory.

    Args:
        episodes (Sequence): ``(embodiment, processed episode)`` pairs.
    """

    def __init__(self, episodes: Sequence[tuple[int, Mapping[str, torch.Tensor]]]) -> None:
        self.items = list(episodes)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]

    def embodiment(self, index: int) -> int:
        return int(self.items[index][0])

    def steps(self, index: int) -> int:
        return int(self.items[index][1]["applied_torque"].shape[0])

    def variants(self, index: int) -> int:
        return int(self.items[index][1]["applied_torque"].shape[1])

    def keys(self, index: int) -> set[str]:
        return set(self.items[index][1])

    def scalar(self, index: int, key: str, default: float | None = None) -> float | None:
        episode = self.items[index][1]
        return float(episode[key]) if key in episode else default

    def frames(self, index: int, start: int, stop: int, keys: Sequence[str]) -> dict[str, torch.Tensor]:
        episode = self.items[index][1]
        return {key: episode[key][start:stop] for key in keys}


def as_episode_store(episodes) -> ProcessedEpisodes | MemoryEpisodes:
    """Return ``episodes`` as a store: a ``ProcessedEpisodes`` view is kept, a sequence is wrapped."""
    if isinstance(episodes, (ProcessedEpisodes, MemoryEpisodes)):
        return episodes
    return MemoryEpisodes(episodes)


class SampleIndex(Sequence):
    """``(episode, frame, variant)`` of every sample, computed from per-episode counts.

    It replaces a list of tuples, which needs about 100 bytes per sample. ``variant`` is
    -1 when a sample holds all variants of its frame (``group_variants``).

    Args:
        steps (Sequence[int]): Frames per episode.
        per_step (Sequence[int]): Samples per frame of each episode.
        grouped (bool): Whether samples hold all variants.
    """

    def __init__(self, steps: Sequence[int], per_step: Sequence[int], grouped: bool) -> None:
        self.steps = np.asarray(steps, dtype=np.int64)
        self.per_step = np.asarray(per_step, dtype=np.int64)
        self.grouped = grouped
        self.starts = np.concatenate(([0], np.cumsum(self.steps * self.per_step)))

    def __len__(self) -> int:
        return int(self.starts[-1])

    def __getitem__(self, item):
        if isinstance(item, slice):
            return [self[position] for position in range(*item.indices(len(self)))]
        item = int(item)
        if item < 0:
            item += len(self)
        if not 0 <= item < len(self):
            raise IndexError(item)
        episode = int(np.searchsorted(self.starts, item, side="right") - 1)
        local = item - int(self.starts[episode])
        per_step = int(self.per_step[episode])
        step, variant = divmod(local, per_step)
        return episode, step, -1 if self.grouped else variant

    def __iter__(self) -> Iterator[tuple[int, int, int]]:
        for episode, (steps, per_step) in enumerate(zip(self.steps.tolist(), self.per_step.tolist())):
            for step in range(steps):
                if self.grouped:
                    yield episode, step, -1
                else:
                    for variant in range(per_step):
                        yield episode, step, variant

    def episodes(self) -> np.ndarray:
        """Return the episode of every sample (vectorized)."""
        return np.repeat(np.arange(len(self.steps)), self.steps * self.per_step)
