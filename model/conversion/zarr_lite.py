"""Read zarr v2 arrays from a directory or a tar archive without the zarr package.

Only what dataset exports such as GrandTour need: ``.zarray`` metadata, C-order
chunks, and blosc, zstd, gzip, or no compression (decoded with ``numcodecs``).
"""

from __future__ import annotations

import itertools
import json
import tarfile
from pathlib import Path

import numpy as np


class _Source:
    """Read files from a directory or from members of a tar archive."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.tar = tarfile.open(self.path) if self.path.is_file() else None
        self.members = (
            {member.name.rstrip("/"): member for member in self.tar.getmembers()} if self.tar else None
        )

    def read(self, name: str) -> bytes | None:
        if self.tar is None:
            target = self.path / name
            return target.read_bytes() if target.exists() else None
        member = self.members.get(name)
        if member is None:
            return None
        handle = self.tar.extractfile(member)
        return None if handle is None else handle.read()

    def names(self) -> list[str]:
        if self.tar is None:
            return [str(item.relative_to(self.path)) for item in self.path.rglob("*")]
        return list(self.members)


def _codec(config: dict | None):
    if config is None:
        return None
    from numcodecs import get_codec

    return get_codec(config)


class ZarrGroup:
    """A read-only zarr v2 group.

    Args:
        path (str | Path): Directory or ``.tar`` file containing the group.
        prefix (str | None): Group path inside the archive; detected when None.
    """

    def __init__(self, path: str | Path, prefix: str | None = None) -> None:
        self.source = _Source(path)
        if prefix is None:
            groups = sorted(name for name in self.source.names() if name.endswith(".zgroup"))
            prefix = groups[0][: -len(".zgroup")].rstrip("/") if groups else ""
        self.prefix = prefix

    def _name(self, *parts: str) -> str:
        return "/".join(part for part in (self.prefix, *parts) if part)

    @property
    def attrs(self) -> dict:
        """Return the group's ``.zattrs``."""
        raw = self.source.read(self._name(".zattrs"))
        return json.loads(raw) if raw else {}

    def arrays(self) -> list[str]:
        """Return the names of arrays directly in the group."""
        base = self._name()
        names = []
        for name in self.source.names():
            if name.endswith("/.zarray"):
                parent = name[: -len("/.zarray")]
                if parent.rsplit("/", 1)[0] == base or (not base and "/" not in parent):
                    names.append(parent.rsplit("/", 1)[-1])
        return sorted(names)

    def __contains__(self, name: str) -> bool:
        return self.source.read(self._name(name, ".zarray")) is not None

    def __getitem__(self, name: str) -> np.ndarray:
        """Read a whole array."""
        meta = json.loads(self.source.read(self._name(name, ".zarray")))
        if meta.get("order", "C") != "C":
            raise ValueError("only C-order zarr arrays are supported")
        shape, chunks = meta["shape"], meta["chunks"]
        dtype = np.dtype(meta["dtype"])
        codec = _codec(meta.get("compressor"))
        filters = [_codec(item) for item in meta.get("filters") or []]
        separator = meta.get("dimension_separator", ".")
        output = np.full(shape, meta.get("fill_value") or 0, dtype=dtype)
        grid = [range((size + chunk - 1) // chunk) for size, chunk in zip(shape, chunks)]
        for index in itertools.product(*grid) if shape else [()]:
            key = separator.join(str(item) for item in index) if index else "0"
            raw = self.source.read(self._name(name, key))
            if raw is None:
                continue
            data = codec.decode(raw) if codec is not None else raw
            for item in reversed(filters):
                data = item.decode(data)
            block = np.frombuffer(data, dtype=dtype).reshape(chunks)
            slices = tuple(slice(i * c, min((i + 1) * c, s)) for i, c, s in zip(index, chunks, shape))
            output[slices] = block[tuple(slice(0, s.stop - s.start) for s in slices)]
        return output
