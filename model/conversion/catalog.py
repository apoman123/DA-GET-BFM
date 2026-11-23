"""Public datasets that can be downloaded into chosen folders and converted.

``scripts/download_datasets.py`` downloads entries of ``CATALOG`` and, unless told not to,
converts them; ``scripts/convert_datasets.py`` converts what is already downloaded, without
the network. Both use the same folder for a dataset, ``<root>/<name>`` unless another is
given::

    <folder>/
      raw/            downloaded files, at their paths in the source repository
                      (``raw_root`` moves this folder elsewhere)
      data/, meta/    the converted table (``model.conversion.datasets``)
      source.json     its data-config source, written by the converter
      hf/             the Hugging Face ``save_to_disk`` copy (``hf_dataset``)
      sources.json    every source of the dataset: the ``hf/`` copies when they exist
      catalog.json    what was converted, from which files, with which selection

* **Converted datasets** go through the converters of ``model.conversion.datasets``.
  Robot models come from ``scripts/download_urdfs.py``.
* **Unitree's G1 LeRobot sets** need no conversion: the ``huggingface`` source reads
  them. Each Hub repository is downloaded without videos (the parquet tables and
  ``meta/``) into ``raw/<repository>/``; the repositories are grouped by the schema of
  their ``meta/info.json`` (checked on 2026-11-23: 55 with the split ``lower_body`` and
  arm vectors, 4 with a 36-entry ``robot_q_current``, 18 ``G1_WBT_SMPL_*`` with the same
  vector), and every repository gets its own source. Their Hugging Face copies go to
  ``hf/<repository>/``.

A selection narrows a dataset: ``patterns`` are globs over the repository paths of the
files (over repository names for the Unitree sets, over the archive members for ``xwbc``,
over the CSV members of the zip for ``neurobem``), and ``limit`` keeps the first files,
repositories or members in sorted order. The same selection picks the files to download
and the files to convert.

The licenses are those of ``docs/data.md``; the datasets belong to their creators.
"""

from __future__ import annotations

import fnmatch
import json
import shutil
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Callable

from model.conversion.common import save_hf_dataset
from model.conversion.datasets import (
    AMASS_REPOSITORY,
    AMASS_REVISION,
    G1_FEET,
    G1_JOINTS,
    GRANDTOUR_REPOSITORY,
    LAFAN1_REPOSITORY,
    LOCOMUJOCO_REPOSITORY,
    MIPO_REPOSITORY,
    NEUROBEM_URL,
    XWBC_REPOSITORY,
    XWBC_REVISION,
    convert_amass_g1,
    convert_grandtour,
    convert_lafan1,
    convert_locomujoco,
    convert_mipo,
    convert_neurobem,
    convert_xwbc,
)

ROOT = Path(__file__).resolve().parents[2]
URDF_DIR = ROOT / "assets" / "urdf"
G1_URDF = "g1_29dof_rev_1_0.urdf"
UNITREE_AUTHOR = "unitreerobotics"
UNITREE_DEX3_REPOSITORY = "unitreerobotics/G1_Dex3_ToastedBread_Dataset"
_IGNORED = (".cache",)  # metadata that huggingface_hub keeps in a local_dir


@dataclass(frozen=True)
class Selection:
    """Which part of a dataset to download and convert.

    Attributes:
        patterns (tuple[str, ...]): Globs (``fnmatch``, where ``*`` also matches ``/``);
            empty keeps everything.
        limit (int | None): Keep only the first files, repositories or members.
    """

    patterns: tuple[str, ...] = ()
    limit: int | None = None

    def apply(self, names: list[str]) -> list[str]:
        """Return the selected names, sorted."""
        names = sorted(names)
        if self.patterns:
            names = [name for name in names if any(fnmatch.fnmatch(name, pattern) for pattern in self.patterns)]
        return names[: self.limit] if self.limit else names


@dataclass(frozen=True)
class RemoteFile:
    """One file to download.

    Attributes:
        name (str): Path in the repository, or the file name of a URL.
        folder (str): Subfolder of ``raw/`` (the repository of a multi-repository dataset).
        repo (str | None): Hub dataset repository.
        revision (str | None): Hub revision (the default branch when None).
        url (str | None): Direct URL instead of a Hub file.
        size (int | None): Bytes, when known.
    """

    name: str
    folder: str = ""
    repo: str | None = None
    revision: str | None = None
    url: str | None = None
    size: int | None = None

    @property
    def local(self) -> str:
        """Return the path under ``raw/``."""
        return f"{self.folder}/{self.name}" if self.folder else self.name


@dataclass(frozen=True)
class CatalogEntry:
    """A dataset of the catalog.

    Attributes:
        name (str): Name on the command line and default folder name.
        description (str): One line.
        size (str): Download size of the whole dataset (checked on 2026-11-23).
        license (str): License summary (``docs/data.md`` has the details).
        urdfs (tuple[str, ...]): Robot models of ``assets/urdf`` the sources use.
        remote (Callable): ``remote(selection) -> list[RemoteFile]`` (needs the network).
        local (Callable): ``local(raw, selection) -> list[Path]``: downloaded files to convert.
        convert (Callable): ``convert(files, raw, folder, selection, urdf_dir)`` returns
            ``[(source, hf_target)]``.
    """

    name: str
    description: str
    size: str
    license: str
    urdfs: tuple[str, ...]
    remote: Callable[[Selection], list[RemoteFile]]
    local: Callable[[Path, Selection], list[Path]]
    convert: Callable[..., list[tuple[dict[str, Any], Path]]]


# ------------------------------------------------------------------------ Hub access


def _hub_tree(repo: str, revision: str | None = None) -> dict[str, int]:
    """Return ``{path: size}`` of every file of a Hub dataset repository."""
    from huggingface_hub import HfApi

    files = HfApi().list_repo_tree(repo, repo_type="dataset", revision=revision, recursive=True)
    return {item.path: item.size for item in files if hasattr(item, "size")}


def _hub_text(repo: str, name: str, revision: str | None = None) -> str:
    """Return a small Hub file as text, without caching it on disk."""
    from huggingface_hub import HfFileSystem

    prefix = f"datasets/{repo}" + (f"@{revision}" if revision else "")
    return HfFileSystem().read_text(f"{prefix}/{name}")


def _unitree_repositories() -> list[str]:
    """Return the ``unitreerobotics/G1_WBT_*`` dataset repositories."""
    from huggingface_hub import HfApi

    return sorted(item.id for item in HfApi().list_datasets(author=UNITREE_AUTHOR, search="G1_WBT"))


def _fetch_hub(file: RemoteFile, raw: Path) -> Path:
    from huggingface_hub import hf_hub_download

    local_dir = raw / file.folder if file.folder else raw
    return Path(hf_hub_download(file.repo, file.name, repo_type="dataset", revision=file.revision, local_dir=local_dir))


def _fetch_url(file: RemoteFile, raw: Path) -> Path:
    target = raw / file.local
    partial = target.with_name(target.name + ".partial")
    target.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(file.url, timeout=120) as response, open(partial, "wb") as handle:
        shutil.copyfileobj(response, handle, length=1 << 20)
    partial.replace(target)
    return target


def _transient(error: Exception) -> bool:
    """Whether a download error may go away on retry (not a missing repository, revision or file)."""
    from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError, RevisionNotFoundError

    if isinstance(error, (EntryNotFoundError, RepositoryNotFoundError, RevisionNotFoundError)):
        return False
    return isinstance(error, OSError) or type(error).__module__.split(".")[0] in ("httpx", "httpx2", "httpcore")


def _retry(function, *arguments, attempts: int = 4, wait: float = 5.0):
    """Call ``function`` and retry transient network errors, waiting 5, 10, 20 s."""
    import time

    for attempt in range(attempts):
        try:
            return function(*arguments)
        except Exception as error:
            if attempt == attempts - 1 or not _transient(error):
                raise
            time.sleep(wait * 2**attempt)


def _url_size(url: str) -> int | None:
    def head():
        request = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.headers.get("Content-Length")

    length = _retry(head)
    return int(length) if length else None


def fetch(file: RemoteFile, raw: Path) -> Path:
    """Download one file into ``raw`` unless a file of the same size is there (with retries)."""
    target = raw / file.local
    if target.is_file() and (file.size is None or target.stat().st_size == file.size):
        return target
    return _retry(_fetch_url if file.url else _fetch_hub, file, raw)


# ----------------------------------------------------------------- local files


def _local_names(raw: Path) -> list[str]:
    """Return the files under ``raw`` as POSIX paths relative to it (``.cache`` and partial files skipped)."""
    if not raw.is_dir():
        return []
    names = []
    for path in raw.rglob("*"):
        relative = PurePosixPath(path.relative_to(raw).as_posix())
        if path.is_file() and relative.parts[0] not in _IGNORED and not relative.name.endswith((".partial", ".incomplete", ".lock")):
            names.append(str(relative))
    return names


def _matching(names: list[str], globs: tuple[str, ...]) -> list[str]:
    return [name for name in names if any(fnmatch.fnmatch(name, pattern) for pattern in globs)]


@dataclass(frozen=True)
class HubFiles:
    """Files of one Hub dataset repository that match ``globs``.

    With ``select`` false the selection is not applied to the files (one archive whose
    members the converter selects).
    """

    repo: str
    globs: tuple[str, ...]
    revision: str | None = None
    select: bool = True

    def remote(self, selection: Selection) -> list[RemoteFile]:
        tree = _hub_tree(self.repo, self.revision)
        names = _matching(list(tree), self.globs)
        names = selection.apply(names) if self.select else sorted(names)
        return [RemoteFile(name, repo=self.repo, revision=self.revision, size=tree[name]) for name in names]

    def local(self, raw: Path, selection: Selection) -> list[Path]:
        names = _matching(_local_names(raw), self.globs)
        names = selection.apply(names) if self.select else sorted(names)
        return [raw / name for name in names]


@dataclass(frozen=True)
class UrlFile:
    """One file at a URL."""

    url: str
    name: str

    def remote(self, selection: Selection) -> list[RemoteFile]:
        return [RemoteFile(self.name, url=self.url, size=_url_size(self.url))]

    def local(self, raw: Path, selection: Selection) -> list[Path]:
        return [raw / self.name] if (raw / self.name).is_file() else []


# ----------------------------------------------------------- Unitree LeRobot sets

_LOWER = "observation.state.lower_body"
_ARMS = ("observation.state.left_arm", "observation.state.right_arm")
_Q36 = "observation.state.robot_q_current"
G1_END_EFFECTORS = ["left_wrist_yaw_joint", "right_wrist_yaw_joint", "left_ankle_roll_joint", "right_ankle_roll_joint"]
# The source of configs/data/g1_wbt_floating.json (tests/test_catalog.py checks that they agree).
UNITREE_WBT_SOURCE: dict[str, Any] = {
    "type": "huggingface",
    "floating_base": True,
    "joint_names": list(G1_JOINTS),
    "state_key": [_LOWER, *_ARMS],
    "action_key": ["action.lower_body", "action.left_arm", "action.right_arm"],
    "action_type": "position",
    "base_position_key": "observation.state.state_base_pose",
    "base_position_indices": [0, 1, 2],
    "base_orientation_key": "observation.state.state_base_pose",
    "base_orientation_indices": [3, 4, 5, 6],
    "base_orientation_format": "xyzw",
    "base_angular_velocity_key": "observation.state.state_base_rot",
    "base_angular_velocity_indices": [3, 4, 5],
    "contact_links": list(G1_FEET),
    "surface_contacts": True,
    "contact_height_tolerance": 0.03,
    "fps": 30,
    "smoothing_window": 3,
    "end_effectors": G1_END_EFFECTORS,
    "actuator_defaults": {"kp": 80.0, "kd": 2.0},
}
# robot_q_current: root x, y, z, quaternion w, x, y, z, then the 29 joints in URDF order.
# Checked on files of G1_WBT_SMPL_Inspire_Pick_Vegetables and G1_WBT_Inspire_Take_Plates_Into_Dishwasher
# (docs/data.md#g1-joint-order-check). robot_q_desired is not used: that it is the commanded
# joint target is not confirmed.
UNITREE_Q36_SOURCE: dict[str, Any] = {
    "type": "huggingface",
    "floating_base": True,
    "joint_names": list(G1_JOINTS),
    "state_key": _Q36,
    "state_indices": list(range(7, 36)),
    "base_position_key": _Q36,
    "base_position_indices": [0, 1, 2],
    "base_orientation_key": _Q36,
    "base_orientation_indices": [3, 4, 5, 6],
    "base_orientation_format": "wxyz",
    "contact_links": list(G1_FEET),
    "surface_contacts": True,
    "contact_height_tolerance": 0.03,
    "fps": 30,
    "smoothing_window": 3,
    "end_effectors": G1_END_EFFECTORS,
}
_G1_ARMS = [joint for joint in G1_JOINTS if "shoulder" in joint or "elbow" in joint or "wrist" in joint]
# The source of configs/data/g1_hub_plus_synthetic.json.
UNITREE_DEX3_SOURCE: dict[str, Any] = {
    "type": "huggingface",
    "joint_names": _G1_ARMS,
    "state_key": "observation.state",
    "state_indices": list(range(14)),
    "action_key": "action",
    "action_indices": list(range(14)),
    "action_type": "position",
    "fps": 30,
    "smoothing_window": 3,
    "end_effectors": ["left_wrist_yaw_joint", "right_wrist_yaw_joint"],
    "actuator_defaults": {"kp": 60.0, "kd": 1.5},
}


def unitree_schema(repo: str, info: dict[str, Any]) -> str | None:
    """Return ``"wbt"``, ``"q36"``, ``"smpl"``, ``"dex3"`` or None from a ``meta/info.json``."""
    features = info.get("features", {})
    if repo.rsplit("/", 1)[-1] == UNITREE_DEX3_REPOSITORY.rsplit("/", 1)[-1]:
        return "dex3"
    if all(key in features for key in (_LOWER, *_ARMS, "observation.state.state_base_pose")):
        return "wbt"
    if features.get(_Q36, {}).get("shape") == [36]:
        return "smpl" if "_SMPL_" in repo else "q36"
    return None


def _is_data_file(name: str) -> bool:
    parts = PurePosixPath(name).parts
    return name.endswith(".parquet") and "data" in parts[:-1] and "meta" not in parts


def _is_meta_file(name: str) -> bool:
    return "meta" in PurePosixPath(name).parts[:-1]


def _info_path(names: list[str]) -> str | None:
    candidates = sorted((name for name in names if PurePosixPath(name).parts[-2:] == ("meta", "info.json")), key=len)
    return candidates[0] if candidates else None


@dataclass(frozen=True)
class UnitreeRepositories:
    """The Unitree repositories of one schema, downloaded as they are (no videos)."""

    schema: str

    def _candidates(self, names: list[str], selection: Selection) -> list[str]:
        """Repository names that may have this schema, filtered by the patterns (the limit counts matches)."""
        if self.schema == "dex3":
            names = [name for name in names if name == UNITREE_DEX3_REPOSITORY.rsplit("/", 1)[-1]]
        else:
            names = [name for name in names if ("_SMPL_" in name) == (self.schema == "smpl")]
        return Selection(selection.patterns).apply(names)

    def remote(self, selection: Selection) -> list[RemoteFile]:
        repositories = [UNITREE_DEX3_REPOSITORY] if self.schema == "dex3" else _unitree_repositories()
        repos = {repo.rsplit("/", 1)[-1]: repo for repo in repositories}
        files, count = [], 0
        for short in self._candidates(list(repos), selection):
            if selection.limit and count >= selection.limit:
                break
            tree = _hub_tree(repos[short])
            info = _info_path(list(tree))
            if info is None or unitree_schema(short, json.loads(_hub_text(repos[short], info))) != self.schema:
                continue
            names = [name for name in tree if _is_data_file(name) or _is_meta_file(name)]
            files += [RemoteFile(name, folder=short, repo=repos[short], size=tree[name]) for name in sorted(names)]
            count += 1
        return files

    def local(self, raw: Path, selection: Selection) -> list[Path]:
        if not raw.is_dir():
            return []
        folders = [path.name for path in raw.iterdir() if path.is_dir() and path.name not in _IGNORED]
        files, count = [], 0
        for short in self._candidates(folders, selection):
            if selection.limit and count >= selection.limit:
                break
            names = _local_names(raw / short)
            info = _info_path(names)
            if info is None or unitree_schema(short, json.loads((raw / short / info).read_text())) != self.schema:
                continue
            files += [raw / short / name for name in sorted(names) if _is_data_file(name)]
            count += 1
        return files

    def convert(self, files: list[Path], raw: Path, folder: Path, selection: Selection, urdf_dir: Path):
        template = {"wbt": UNITREE_WBT_SOURCE, "q36": UNITREE_Q36_SOURCE, "smpl": UNITREE_Q36_SOURCE, "dex3": UNITREE_DEX3_SOURCE}[self.schema]
        output = []
        for short in sorted({file.relative_to(raw).parts[0] for file in files}):
            repo_dir = raw / short
            info_name = _info_path(_local_names(repo_dir))
            info = json.loads((repo_dir / info_name).read_text())
            prefix = str(PurePosixPath(info_name).parent.parent)
            source = json.loads(json.dumps(template))
            source.update({
                "name": short.lower(),
                "path": str(repo_dir.resolve()),
                "data_files": ("" if prefix == "." else prefix + "/") + "data/*/*.parquet",
                "urdf": str((urdf_dir / G1_URDF).resolve()),
                "fps": info.get("fps", source["fps"]),
            })
            features = info.get("features", {})
            for key in ("action_key", "base_angular_velocity_key"):
                if key in source and not all(column in features for column in ([source[key]] if isinstance(source[key], str) else source[key])):
                    for drop in [name for name in source if name.startswith(key.removesuffix("_key"))]:
                        source.pop(drop)
            output.append((source, folder / "hf" / short))
        return output


# -------------------------------------------------------------------- converters


def _urdf(urdf_dir: Path, name: str) -> str:
    return str((urdf_dir / name).resolve())


def _files_converter(converter, urdf: str, **fixed):
    """Adapter for a converter that takes the list of local files."""

    def convert(files, raw, folder, selection, urdf_dir):
        return [(converter(folder, files=files, urdf=_urdf(urdf_dir, urdf), **fixed), folder / "hf")]

    return convert


def _convert_grandtour(files, raw, folder, selection, urdf_dir):
    return [(convert_grandtour(folder, archives=files, urdf=_urdf(urdf_dir, "anymal_d.urdf")), folder / "hf")]


def _convert_neurobem(files, raw, folder, selection, urdf_dir):
    source = convert_neurobem(folder, source=files[0], patterns=list(selection.patterns) or None, max_flights=selection.limit)
    return [(source, folder / "hf")]


def _convert_locomujoco(files, raw, folder, selection, urdf_dir):
    output = []
    for collection in ("DefaultDatasets", "Lafan1"):
        group = [file for file in files if file.relative_to(raw).parts[0] == collection]
        if group:
            target = folder / collection.lower()
            source = convert_locomujoco(target, robot="UnitreeG1", collection=collection, files=group, urdf=_urdf(urdf_dir, G1_URDF))
            source["name"] = f"locomujoco_g1_{collection.lower()}"  # the two collections need different names
            (target / "source.json").write_text(json.dumps(source, indent=2))
            output.append((source, target / "hf"))
    return output


def _xwbc(collection: str):
    def convert(files, raw, folder, selection, urdf_dir):
        source = convert_xwbc(
            folder, robot="unitree_g1_29dof", collection=collection, archive=files[0],
            patterns=list(selection.patterns) or None, max_files=selection.limit,
            urdf=_urdf(urdf_dir, G1_URDF), contact_links=list(G1_FEET),
        )
        return [(source, folder / "hf")]

    return convert


def _hub(repo, globs, revision=None, select=True):
    return HubFiles(repo, tuple(globs), revision, select)


def _entry(name, description, size, license, urdfs, files, convert) -> CatalogEntry:
    return CatalogEntry(name, description, size, license, tuple(urdfs), files.remote, files.local, convert)


_unitree = {schema: UnitreeRepositories(schema) for schema in ("wbt", "q36", "smpl", "dex3")}
CATALOG: dict[str, CatalogEntry] = {entry.name: entry for entry in (
    _entry("grandtour", "GrandTour ANYmal D: measured joint torques, foot contact states, base state (48 missions)",
           "5.3 GB", "data CC BY-SA 4.0 (website), MIT on the card", ["anymal_d.urdf"],
           _hub(GRANDTOUR_REPOSITORY, ["*/data/anymal_state_state_estimator.tar"]), _convert_grandtour),
    _entry("mipo", "MIPO Unitree Go1: measured torques, IMU, foot forces (84 sequences)",
           "0.53 GB", "AGPL-3.0", ["go1.urdf"],
           _hub(MIPO_REPOSITORY, ["*.h5"]), _files_converter(convert_mipo, "go1.urdf")),
    _entry("neurobem", "NeuroBEM quadrotor flights; the propeller model is identified from the data",
           "0.63 GB", "none stated for the data", [],
           UrlFile(NEUROBEM_URL, "processed_data.zip"), _convert_neurobem),
    _entry("locomujoco_g1", "LocoMuJoCo Unitree G1 motions: 19 default and 40 LAFAN1 (floating base, named joints)",
           "0.10 GB", "CC BY-NC-ND 4.0", [G1_URDF],
           _hub(LOCOMUJOCO_REPOSITORY, ["DefaultDatasets/mocap/UnitreeG1/*.npz", "Lafan1/mocap/UnitreeG1/*.npz"]), _convert_locomujoco),
    _entry("amass_g1", "AMASS retargeted to G1 by fleaven (17,717 clips; wrist pitch and yaw are 0)",
           "6.95 GB", "AMASS terms: non-commercial research", [G1_URDF],
           _hub(AMASS_REPOSITORY, ["g1/*.npy"], AMASS_REVISION), _files_converter(convert_amass_g1, G1_URDF)),
    _entry("xwbc_g1_amass", "AMASS retargeted to G1 by xwbc: every joint moves, base twist (one archive)",
           "8.10 GB", "AMASS terms: non-commercial research", [G1_URDF],
           _hub(XWBC_REPOSITORY, ["amass/unitree_g1_29dof.tar.gz"], XWBC_REVISION, select=False), _xwbc("amass")),
    _entry("xwbc_g1_lafan", "LAFAN1 retargeted to G1 by xwbc: every joint moves, base twist (one archive)",
           "0.84 GB", "LAFAN1: CC BY-NC-ND 4.0", [G1_URDF],
           _hub(XWBC_REPOSITORY, ["lafan/unitree_g1_29dof.tar.gz"], XWBC_REVISION, select=False), _xwbc("lafan")),
    _entry("lafan1_g1", "LAFAN1 retargeted to G1 by Unitree (40 clips, 30 Hz)",
           "0.08 GB", "LAFAN1: CC BY-NC-ND 4.0; retargeting BSD-3-Clause", [G1_URDF],
           _hub(LAFAN1_REPOSITORY, ["g1/*.csv"]), _files_converter(convert_lafan1, G1_URDF, robot="g1")),
    _entry("lafan1_h1", "LAFAN1 retargeted to H1 by Unitree (40 clips, 30 Hz)",
           "0.08 GB", "LAFAN1: CC BY-NC-ND 4.0; retargeting BSD-3-Clause", ["h1.urdf"],
           _hub(LAFAN1_REPOSITORY, ["h1/*.csv"]), _files_converter(convert_lafan1, "h1.urdf", robot="h1")),
    _entry("g1_wbt", "Unitree G1 whole-body teleoperation, 55 G1_WBT_* repositories (parquet only, no videos)",
           "13.4 GB", "Apache-2.0", [G1_URDF], _unitree["wbt"], _unitree["wbt"].convert),
    _entry("g1_wbt_q36", "Unitree G1_WBT_Inspire_* repositories with the 36-entry robot_q_current (4)",
           "0.45 GB", "Apache-2.0", [G1_URDF], _unitree["q36"], _unitree["q36"].convert),
    _entry("g1_wbt_smpl", "Unitree G1_WBT_SMPL_*: real G1 whole-body motion with root pose (18 repositories)",
           "3.8 GB", "Apache-2.0", [G1_URDF], _unitree["smpl"], _unitree["smpl"].convert),
    _entry("g1_dex3", "Unitree G1_Dex3_ToastedBread_Dataset: G1 arms (fixed base)",
           "0.08 GB", "Apache-2.0", [G1_URDF], _unitree["dex3"], _unitree["dex3"].convert),
)}


# ----------------------------------------------------------------------- folders


@dataclass(frozen=True)
class Layout:
    """Where the folders of the datasets are.

    Attributes:
        root (Path): Parent of the dataset folders.
        paths (dict[str, Path]): Folders chosen per dataset.
        raw_root (Path | None): Parent of the download folders (default ``<folder>/raw``).
    """

    root: Path = Path("data")
    paths: dict[str, Path] = field(default_factory=dict)
    raw_root: Path | None = None

    def folder(self, name: str) -> Path:
        return Path(self.paths.get(name, self.root / name))

    def raw(self, name: str) -> Path:
        return Path(self.raw_root) / name if self.raw_root else self.folder(name) / "raw"


def missing_urdfs(entry: CatalogEntry, urdf_dir: Path = URDF_DIR) -> list[str]:
    """Return the robot models an entry needs that are not in ``urdf_dir``."""
    return [name for name in entry.urdfs if not (urdf_dir / name).is_file()]


# ---------------------------------------------------------------------- download


def plan_download(entry: CatalogEntry, raw: Path, selection: Selection) -> tuple[list[RemoteFile], list[RemoteFile]]:
    """Return ``(all selected files, files still to download)``."""
    files = entry.remote(selection)
    missing = [file for file in files if not ((raw / file.local).is_file() and (file.size is None or (raw / file.local).stat().st_size == file.size))]
    return files, missing


def download_files(files: list[RemoteFile], raw: Path, workers: int = 8, log: Callable[[str], None] = print) -> None:
    """Download files into ``raw`` with a few threads, reporting progress."""
    if not files:
        return
    raw.mkdir(parents=True, exist_ok=True)
    total = sum(file.size or 0 for file in files)
    done = size = 0
    step = max(1, len(files) // 50)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(fetch, file, raw): file for file in files}
        for future in as_completed(futures):
            future.result()
            done += 1
            size += futures[future].size or 0
            if done % step == 0 or done == len(files):
                log(f"  {done}/{len(files)} files, {size / 1e9:.2f}/{total / 1e9:.2f} GB")


# ----------------------------------------------------------------------- convert


def _selection_record(selection: Selection) -> dict[str, Any]:
    return {"patterns": list(selection.patterns), "limit": selection.limit}


def convert(
    entry: CatalogEntry,
    folder: Path,
    raw: Path,
    selection: Selection = Selection(),
    hf_dataset: bool = True,
    force: bool = False,
    urdf_dir: Path = URDF_DIR,
    log: Callable[[str], None] = print,
) -> list[dict[str, Any]]:
    """Convert the downloaded files of a dataset and write ``sources.json``.

    The conversion is skipped when ``catalog.json`` records the same files and selection
    (``force`` redoes it). With ``hf_dataset`` every source also gets a Hugging Face
    ``save_to_disk`` copy (``model.conversion.common.save_hf_dataset``), and
    ``sources.json`` lists the copies.

    Returns:
        list[dict]: The sources written to ``sources.json``.
    """
    folder.mkdir(parents=True, exist_ok=True)
    state_path = folder / "catalog.json"
    state = json.loads(state_path.read_text()) if state_path.is_file() else None
    files = entry.local(raw, selection)
    names = [file.relative_to(raw).as_posix() for file in files]
    if not files and (state is None or force):
        raise FileNotFoundError(f"{entry.name}: nothing downloaded in {raw}; run scripts/download_datasets.py --dataset {entry.name}")
    current = state is not None and (not files or (state.get("files") == names and state.get("selection") == _selection_record(selection)))
    if force or not current:
        missing = missing_urdfs(entry, urdf_dir)
        if missing:
            raise FileNotFoundError(f"{entry.name}: robot models missing in {urdf_dir}: {missing}; run scripts/download_urdfs.py")
        state_path.unlink(missing_ok=True)
        log(f"{entry.name}: converting {len(files)} files")
        pairs = entry.convert(files, raw, folder, selection, urdf_dir)
        for _, target in pairs:
            shutil.rmtree(target, ignore_errors=True)
        state = {
            "dataset": entry.name,
            "files": names,
            "selection": _selection_record(selection),
            "sources": [source for source, _ in pairs],
            "hf_targets": [str(target) for _, target in pairs],
        }
        state_path.write_text(json.dumps(state, indent=2))
    else:
        log(f"{entry.name}: already converted ({len(state['files'])} files)")
    sources = state["sources"]
    if hf_dataset:
        from model.training.sources import is_saved_dataset

        copies = []
        for source, target in zip(state["sources"], state["hf_targets"]):
            target = Path(target)
            if force or not is_saved_dataset(target):
                log(f"{entry.name}: writing the Hugging Face dataset {target}")
                copies.append(save_hf_dataset(source, target))
            else:
                copies.append({**{k: v for k, v in source.items() if k not in ("data_files", "repo_id", "revision", "split")}, "path": str(target.resolve())})
        sources = copies
    gone = [source["path"] for source in sources if "path" in source and not Path(source["path"]).exists()]
    if gone:
        raise FileNotFoundError(f"{entry.name}: the sources read {gone}, which no longer exist (deleted downloads?); "
                                "use --hf_dataset, or download again with --redownload")
    (folder / "sources.json").write_text(json.dumps(sources, indent=2))
    return sources


def reads_raw(sources: list[dict[str, Any]], raw: Path) -> bool:
    """Whether any source reads files under ``raw``."""
    raw = raw.resolve()
    return any(Path(source["path"]).resolve().is_relative_to(raw) for source in sources if "path" in source)


# --------------------------------------------------------------- command line


def _pairs(values: list[str], option: str) -> dict[str, str]:
    output = {}
    for value in values:
        name, separator, rest = value.partition("=")
        if not separator or name not in CATALOG:
            raise SystemExit(f"{option} {value!r}: expected NAME=VALUE with NAME one of {sorted(CATALOG)}")
        output[name] = rest
    return output


def _patterns(values: list[str], names: list[str]) -> dict[str, list[str]]:
    output: dict[str, list[str]] = {name: [] for name in names}
    for value in values:
        name, separator, rest = value.partition("=")
        targets = [name] if separator and name in CATALOG else names
        for target in targets:
            output.setdefault(target, []).append(rest if separator and name in CATALOG else value)
    return output


def _free_bytes(path: Path) -> int:
    path = path.resolve()
    while not path.exists():
        path = path.parent
    return shutil.disk_usage(path).free


def _fetch_urdfs(urdf_dir: Path, log: Callable[[str], None]) -> None:
    """Run ``scripts/download_urdfs.py`` (it skips files that are present)."""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "download_urdfs.py"), "--output", str(urdf_dir)],
        capture_output=True, text=True, check=False,
    )
    for line in result.stdout.splitlines():
        if not line.startswith("exists"):
            log(line)
    if result.returncode:
        raise RuntimeError(f"scripts/download_urdfs.py failed:\n{result.stderr}")


def build_parser(description: str, download: bool):
    """Return the argument parser of ``scripts/download_datasets.py`` or ``scripts/convert_datasets.py``."""
    import argparse

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--dataset", "-d", action="append", default=[], metavar="NAME",
                        help="Dataset to process; repeatable or comma-separated (see --list).")
    parser.add_argument("--all", action="store_true", help="Every dataset of the catalog.")
    parser.add_argument("--list", action="store_true", help="Print the catalog and exit.")
    parser.add_argument("--root", default="data", help="Parent folder of the dataset folders (default: data).")
    parser.add_argument("--path", action="append", default=[], metavar="NAME=DIR",
                        help="Folder of one dataset instead of <root>/<NAME>; repeatable.")
    parser.add_argument("--raw_root", metavar="DIR",
                        help="Keep the downloads in <DIR>/<NAME> instead of <folder>/raw.")
    parser.add_argument("--limit", type=int, metavar="N",
                        help="Only the first N files (repositories of the Unitree sets, clips of xwbc, flights of neurobem).")
    parser.add_argument("--pattern", action="append", default=[], metavar="[NAME=]GLOB",
                        help="Keep files whose repository path matches GLOB (repository names of the Unitree sets, "
                             "archive members of xwbc, CSV members of neurobem); without NAME= for every dataset; repeatable.")
    parser.add_argument("--hf_dataset", action=argparse.BooleanOptionalAction, default=True,
                        help="Also save every source as a Hugging Face dataset (save_to_disk, in hf/) and list those in "
                             "sources.json (default: on).")
    parser.add_argument("--force", action="store_true", help="Convert again even if catalog.json says it is done.")
    parser.add_argument("--delete_raw", action="store_true",
                        help="Delete the downloads after converting (refused while a source reads them).")
    parser.add_argument("--urdf_dir", default=str(URDF_DIR), help="Folder of the robot models (default: assets/urdf).")
    if download:
        parser.add_argument("--convert", action=argparse.BooleanOptionalAction, default=True,
                            help="Convert after downloading (default: on).")
        parser.add_argument("--redownload", action="store_true", help="Download files again even if present.")
        parser.add_argument("--workers", type=int, default=8, help="Parallel downloads (default: 8).")
        parser.add_argument("--dry_run", action="store_true", help="List what would be downloaded and exit.")
    return parser


def main(argv: list[str] | None = None, download: bool = True, description: str = "") -> int:
    """Download and/or convert catalog datasets from the command line; return the exit status."""
    parser = build_parser(description, download)
    args = parser.parse_args(argv)
    if args.list:
        for entry in CATALOG.values():
            print(f"{entry.name:15s} {entry.size:>8s}  {entry.description}\n{'':25s}license: {entry.license}")
        return 0
    names = list(CATALOG) if args.all else [name.strip() for value in args.dataset for name in value.split(",") if name.strip()]
    unknown = sorted(set(names) - set(CATALOG))
    if unknown:
        parser.error(f"unknown datasets {unknown}; see --list")
    if not names:
        parser.error("choose datasets with --dataset NAME (see --list) or --all")
    names = list(dict.fromkeys(names))
    layout = Layout(Path(args.root), {name: Path(path) for name, path in _pairs(args.path, "--path").items()},
                    Path(args.raw_root) if args.raw_root else None)
    patterns = _patterns(args.pattern, names)
    urdf_dir = Path(args.urdf_dir)
    converting = not download or args.convert
    dry_run = download and args.dry_run
    if download and converting and not dry_run and any(CATALOG[name].urdfs for name in names):
        _fetch_urdfs(urdf_dir, print)
    done, failed = [], []
    total_missing = 0
    for name in names:
        entry = CATALOG[name]
        selection = Selection(tuple(patterns.get(name, ())), args.limit)
        folder, raw = layout.folder(name), layout.raw(name)
        try:
            if download:
                if not args.redownload and not args.force and (folder / "catalog.json").is_file() and not raw.exists():
                    print(f"{name}: converted earlier and its downloads were deleted; skipped (--redownload fetches them again)")
                    continue
                files, missing = plan_download(entry, raw, selection)
                if args.redownload:
                    missing = files
                    for file in files:
                        (raw / file.local).unlink(missing_ok=True)
                size = sum(file.size or 0 for file in files)
                need = sum(file.size or 0 for file in missing)
                total_missing += need
                print(f"{name}: {len(files)} file(s), {size / 1e9:.2f} GB; {len(missing)} to download "
                      f"({need / 1e9:.2f} GB) into {raw}")
                if dry_run:
                    continue
                if not files:
                    raise FileNotFoundError("the selection matches no files")
                if need > _free_bytes(raw):
                    raise OSError(f"{need / 1e9:.2f} GB needed but {_free_bytes(raw) / 1e9:.2f} GB free at {raw}")
                download_files(missing, raw, args.workers)
            if not converting:
                continue
            sources = convert(entry, folder, raw, selection, args.hf_dataset, args.force, urdf_dir)
            if args.delete_raw and raw.exists():
                if reads_raw(sources, raw):
                    print(f"{name}: its sources read {raw}, which is kept (use --hf_dataset to make a copy first)")
                else:
                    shutil.rmtree(raw)
                    print(f"{name}: deleted {raw}")
            done.append((name, folder / "sources.json", len(sources)))
        except Exception as error:  # report and go on with the other datasets
            failed.append((name, error))
            print(f"{name}: FAILED: {type(error).__name__}: {error}")
    if dry_run:
        print(f"total to download: {total_missing / 1e9:.2f} GB")
    for name, path, count in done:
        print(f"{name}: {count} source(s) in {path}")
    return 1 if failed else 0


__all__ = [
    "CATALOG",
    "CatalogEntry",
    "Layout",
    "RemoteFile",
    "Selection",
    "UNITREE_DEX3_SOURCE",
    "UNITREE_Q36_SOURCE",
    "UNITREE_WBT_SOURCE",
    "convert",
    "download_files",
    "fetch",
    "missing_urdfs",
    "plan_download",
    "reads_raw",
    "build_parser",
    "main",
    "unitree_schema",
]
