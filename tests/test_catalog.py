"""Test the dataset catalog: download and convert scripts, folders, Hugging Face copies, streamed tables."""

import contextlib
import io
import json
import shutil
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from model.conversion import catalog
from model.conversion.common import Episode, save_hf_dataset, write_dataset
from model.conversion.datasets import G1_JOINTS, LAFAN1_REPOSITORY
from model.embodiment import JointFeatureConfig
from model.training.dataset import load_source
from model.training.sources import is_saved_dataset, source_columns

ROOT = Path(__file__).resolve().parents[1]
G1_URDF = ROOT / "assets" / "urdf" / catalog.G1_URDF
NEEDS_G1_URDF = unittest.skipUnless(G1_URDF.exists(), "needs the robot URDFs: uv run python scripts/download_urdfs.py")
FEATURES = JointFeatureConfig.da_get_zero()


def lafan_csv(path: Path, frames: int = 40, offset: float = 0.0) -> None:
    """Write a LAFAN1-on-G1 CSV: root x, y, z, quaternion x, y, z, w, then 29 joints."""
    table = np.zeros((frames, 36))
    table[:, 2] = 0.75
    table[:, 6] = 1.0
    table[:, 7 + G1_JOINTS.index("left_elbow_joint")] = np.linspace(0.0, 0.5, frames) + offset
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, table, delimiter=",")


def lerobot_repo(folder: Path, columns: dict[str, np.ndarray], episodes: int = 2, frames: int = 40, prefix: str = "") -> None:
    """Write a small LeRobot-style repository: ``meta/info.json``, one parquet file and a video to skip."""
    base = folder / prefix if prefix else folder
    (base / "meta").mkdir(parents=True)
    (base / "data" / "chunk-000").mkdir(parents=True)
    (folder / "videos").mkdir(parents=True, exist_ok=True)
    (folder / "videos" / "camera.mp4").write_bytes(b"not a video")
    count = episodes * frames
    arrays = {
        "episode_index": pa.array(np.repeat(np.arange(episodes), frames)),
        "frame_index": pa.array(np.tile(np.arange(frames), episodes)),
        "timestamp": pa.array(np.tile(np.arange(frames) / 30.0, episodes).astype(np.float32)),
    }
    features = {}
    for name, row in columns.items():
        values = np.tile(np.asarray(row, dtype=np.float32), (count, 1))
        arrays[name] = pa.array(list(values))
        features[name] = {"dtype": "float32", "shape": [values.shape[1]], "names": None}
    pq.write_table(pa.table(arrays), base / "data" / "chunk-000" / "file-000.parquet")
    (base / "meta" / "info.json").write_text(json.dumps({"fps": 30, "robot_type": "unitree_g1", "features": features}))


class FakeHub:
    """Serve ``<remote>/<repo>/<path>`` as Hub files and record what is fetched."""

    def __init__(self, remote: Path) -> None:
        self.remote = remote
        self.fetched: list[str] = []

    def tree(self, repo, revision=None):
        base = self.remote / repo
        return {path.relative_to(base).as_posix(): path.stat().st_size for path in base.rglob("*") if path.is_file()}

    def fetch(self, file, raw):
        target = raw / file.local
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(self.remote / file.repo / file.name, target)
        self.fetched.append(file.local)
        return target

    def text(self, repo, name, revision=None):
        return (self.remote / repo / name).read_text()

    def repositories(self):
        return sorted(f"{catalog.UNITREE_AUTHOR}/{path.name}" for path in (self.remote / catalog.UNITREE_AUTHOR).iterdir())

    @contextlib.contextmanager
    def active(self):
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(catalog, "_hub_tree", self.tree))
            stack.enter_context(mock.patch.object(catalog, "_fetch_hub", self.fetch))
            stack.enter_context(mock.patch.object(catalog, "_hub_text", self.text))
            stack.enter_context(mock.patch.object(catalog, "_unitree_repositories", self.repositories))
            stack.enter_context(mock.patch.object(catalog, "_fetch_urdfs", lambda *arguments: None))
            yield self


def run(arguments: list[str], download: bool = True) -> tuple[int, str]:
    output = io.StringIO()
    with contextlib.redirect_stdout(output), contextlib.redirect_stderr(io.StringIO()):
        status = catalog.main(arguments, download=download)
    return status, output.getvalue()


class StreamedTableTest(unittest.TestCase):
    """``write_dataset`` streams row groups and never leaves a half-written table."""

    def test_row_groups_and_failure(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            def episodes(fail: bool = False):
                for index in range(5):
                    if fail and index == 3:
                        raise RuntimeError("source broke")
                    yield Episode({"state": np.full((7, 2), index, dtype=np.float64), "base_position": np.zeros((7, 3))})

            info = write_dataset(Path(folder) / "out", episodes(), 10.0, ["a", "b"], rows_per_group=10)
            parquet = pq.ParquetFile(Path(folder) / "out" / "data" / "chunk-000" / "file-000.parquet")
            self.assertGreater(parquet.num_row_groups, 1)
            table = parquet.read().to_pydict()
            self.assertEqual((info["total_episodes"], info["total_frames"]), (5, 35))
            self.assertEqual(table["episode_index"], [index for index in range(5) for _ in range(7)])
            self.assertEqual(table["frame_index"][:8], [0, 1, 2, 3, 4, 5, 6, 0])
            self.assertAlmostEqual(table["timestamp"][6], 0.6, places=6)
            self.assertEqual(table["observation.state"][-1], [4.0, 4.0])
            with self.assertRaises(RuntimeError):
                write_dataset(Path(folder) / "broken", episodes(fail=True), 10.0, ["a", "b"], rows_per_group=10)
            self.assertEqual(list((Path(folder) / "broken" / "data" / "chunk-000").iterdir()), [])
            with self.assertRaises(ValueError):
                write_dataset(Path(folder) / "none", iter(()), 10.0, ["a", "b"])


class CatalogTest(unittest.TestCase):
    """The catalog's download, conversion, folders and Hugging Face copies, on a fake Hub."""

    def setUp(self) -> None:
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)
        self.hub = FakeHub(self.root / "remote")

    def tearDown(self) -> None:
        self.folder.cleanup()

    def test_selection_and_options(self) -> None:
        self.assertEqual(catalog.Selection(("*b*", "*c*"), 1).apply(["c", "ab", "x"]), ["ab"])
        self.assertEqual(catalog._patterns(["lafan1_g1=*a*", "*b*"], ["lafan1_g1", "mipo"]),
                         {"lafan1_g1": ["*a*", "*b*"], "mipo": ["*b*"]})
        with self.assertRaises(SystemExit):
            catalog._pairs(["nothing=here"], "--path")
        layout = catalog.Layout(Path("data"), {"mipo": Path("/big/mipo")}, Path("/raw"))
        self.assertEqual((layout.folder("mipo"), layout.raw("mipo")), (Path("/big/mipo"), Path("/raw/mipo")))
        self.assertEqual(catalog.Layout().raw("mipo"), Path("data/mipo/raw"))
        status, text = run(["--list"])
        self.assertEqual(status, 0)
        for name in catalog.CATALOG:
            self.assertIn(name, text)
        with self.assertRaises(SystemExit):
            run(["--dataset", "unknown"])

    def test_url_file(self) -> None:
        """A URL entry (NeuroBEM) is sized, downloaded through a partial file, and found offline."""
        served = self.root / "served" / "processed_data.zip"
        served.parent.mkdir()
        with zipfile.ZipFile(served, "w") as archive:
            archive.writestr("flight.csv", "t\n0\n")
        entry = catalog.UrlFile(served.as_uri(), "processed_data.zip")
        files = entry.remote(catalog.Selection())
        self.assertEqual(files[0].size, served.stat().st_size)
        raw = self.root / "raw"
        self.assertEqual(entry.local(raw, catalog.Selection()), [])
        catalog.download_files(files, raw, log=lambda line: None)
        self.assertEqual((raw / "processed_data.zip").read_bytes(), served.read_bytes())
        self.assertEqual(entry.local(raw, catalog.Selection()), [raw / "processed_data.zip"])
        self.assertEqual(list(raw.glob("*.partial")), [])
        self.assertEqual(catalog.plan_download(catalog.CatalogEntry("x", "", "", "", (), entry.remote, entry.local, None), raw,
                                               catalog.Selection())[1], [])

    def test_templates_match_the_configs(self) -> None:
        """The Unitree sources are those of the shipped data configs."""
        for config, template, ignored in (
            ("g1_wbt_floating.json", catalog.UNITREE_WBT_SOURCE, {"name", "repo_id", "data_files", "urdf", "max_episodes"}),
            ("g1_hub_plus_synthetic.json", catalog.UNITREE_DEX3_SOURCE, {"name", "repo_id", "data_files", "urdf", "max_episodes"}),
        ):
            source = json.loads((ROOT / "configs" / "data" / config).read_text())["sources"][0]
            self.assertEqual({key: value for key, value in source.items() if key not in ignored}, template, config)
        self.assertEqual(catalog.unitree_schema("x/G1_WBT_SMPL_A", {"features": {catalog._Q36: {"shape": [36]}}}), "smpl")
        self.assertEqual(catalog.unitree_schema("x/G1_WBT_Inspire_A", {"features": {catalog._Q36: {"shape": [36]}}}), "q36")
        self.assertIsNone(catalog.unitree_schema("x/G1_WBT_A", {"features": {}}))

    @NEEDS_G1_URDF
    def test_download_convert_and_hf_copy(self) -> None:
        for name, offset in (("a", 0.0), ("b", 0.1), ("c", 0.2)):
            lafan_csv(self.root / "remote" / LAFAN1_REPOSITORY / "g1" / f"{name}.csv", offset=offset)
        data = self.root / "data"
        folder = data / "lafan1_g1"
        with self.hub.active():
            status, text = run(["--dataset", "lafan1_g1", "--root", str(data), "--limit", "2"])
            self.assertEqual(status, 0, text)
            self.assertEqual(sorted(self.hub.fetched), ["g1/a.csv", "g1/b.csv"])
            state = json.loads((folder / "catalog.json").read_text())
            self.assertEqual(state["files"], ["g1/a.csv", "g1/b.csv"])
            sources = json.loads((folder / "sources.json").read_text())
            self.assertEqual(len(sources), 1)
            self.assertTrue(is_saved_dataset(Path(sources[0]["path"])))
            self.assertEqual(Path(sources[0]["path"]), (folder / "hf").resolve())
            from datasets import load_from_disk

            # Only the columns the source reads are copied (the converted table also has timestamp).
            self.assertEqual(sorted(load_from_disk(sources[0]["path"]).column_names), source_columns(sources[0]))
            _, copy = load_source(sources[0], None, FEATURES)
            _, table = load_source(state["sources"][0], None, FEATURES)
            self.assertEqual(len(copy), 2)
            for left, right in zip(copy, table):
                self.assertTrue(torch.equal(left.positions, right.positions))
            # Nothing is downloaded or converted again.
            self.hub.fetched.clear()
            status, text = run(["--dataset", "lafan1_g1", "--root", str(data), "--limit", "2"])
            self.assertEqual((status, self.hub.fetched), (0, []))
            self.assertIn("already converted", text)
            # The conversion script alone, without the copies: sources.json lists the table.
            status, text = run(["--dataset", "lafan1_g1", "--root", str(data), "--limit", "2", "--no-hf_dataset", "--delete_raw"], download=False)
            self.assertEqual(status, 0, text)
            self.assertEqual(Path(json.loads((folder / "sources.json").read_text())[0]["path"]), folder.resolve())
            self.assertFalse((folder / "raw").exists())
            status, text = run(["--dataset", "lafan1_g1", "--root", str(data)])
            self.assertEqual((status, self.hub.fetched), (0, []))
            self.assertIn("downloads were deleted", text)
            # A larger selection is converted again from the files present.
            status, text = run(["--dataset", "lafan1_g1", "--root", str(data), "--redownload"])
            self.assertEqual(status, 0, text)
            self.assertEqual(json.loads((folder / "catalog.json").read_text())["files"], ["g1/a.csv", "g1/b.csv", "g1/c.csv"])
            status, text = run(["--dataset", "lafan1_g1", "--root", str(self.root / "empty")], download=False)
            self.assertEqual(status, 1)
            self.assertIn("nothing downloaded", text)

    @NEEDS_G1_URDF
    def test_paths_patterns_and_dry_run(self) -> None:
        for name in ("walk", "run", "dance"):
            lafan_csv(self.root / "remote" / LAFAN1_REPOSITORY / "g1" / f"{name}.csv")
        folder, raw_root = self.root / "chosen", self.root / "downloads"
        arguments = ["-d", "lafan1_g1", "--path", f"lafan1_g1={folder}", "--raw_root", str(raw_root), "--pattern", "lafan1_g1=*/[rw]*"]
        with self.hub.active():
            status, text = run([*arguments, "--dry_run"])
            self.assertEqual((status, self.hub.fetched), (0, []))
            self.assertIn("2 file(s)", text)
            self.assertIn(str(raw_root / "lafan1_g1"), text)
            self.assertFalse(folder.exists())
            status, text = run([*arguments, "--no-convert"])
            self.assertEqual(status, 0, text)
            self.assertEqual(sorted(self.hub.fetched), ["g1/run.csv", "g1/walk.csv"])
            self.assertTrue((raw_root / "lafan1_g1" / "g1" / "run.csv").is_file())
            self.assertFalse((folder / "sources.json").exists())
            status, text = run(arguments, download=False)
            self.assertEqual(status, 0, text)
            sources = json.loads((folder / "sources.json").read_text())
            self.assertEqual(Path(sources[0]["path"]), (folder / "hf").resolve())

    @NEEDS_G1_URDF
    def test_unitree_sets(self) -> None:
        """Repositories are sorted by schema, downloaded without videos, and get loadable sources."""
        unitree = self.root / "remote" / catalog.UNITREE_AUTHOR
        q36 = [0.0, 0.0, 0.75, 1.0, 0.0, 0.0, 0.0] + [0.0] * 29
        lerobot_repo(unitree / "G1_WBT_Brainco_Task", {
            "observation.state.lower_body": [0.0] * 15, "observation.state.left_arm": [0.1] * 7,
            "observation.state.right_arm": [0.2] * 7, "observation.state.state_base_pose": [0, 0, 0.75, 0, 0, 0, 1],
            "action.lower_body": [0.0] * 15, "action.left_arm": [0.1] * 7, "action.right_arm": [0.2] * 7,
        })
        lerobot_repo(unitree / "G1_WBT_Inspire_Task_MainCamOnly", {catalog._Q36: q36}, prefix="G1_WB_Dex5_Task")
        lerobot_repo(unitree / "G1_WBT_SMPL_Brainco_Task", {catalog._Q36: q36})
        data = self.root / "data"
        with self.hub.active():
            status, text = run(["-d", "g1_wbt,g1_wbt_q36,g1_wbt_smpl", "--root", str(data)])
        self.assertEqual(status, 0, text)
        self.assertFalse(any(name.endswith(".mp4") for name in self.hub.fetched))
        wbt = json.loads((data / "g1_wbt" / "sources.json").read_text())
        q36_sources = json.loads((data / "g1_wbt_q36" / "sources.json").read_text())
        smpl = json.loads((data / "g1_wbt_smpl" / "sources.json").read_text())
        self.assertEqual([source["name"] for source in wbt + q36_sources + smpl],
                         ["g1_wbt_brainco_task", "g1_wbt_inspire_task_maincamonly", "g1_wbt_smpl_brainco_task"])
        self.assertNotIn("base_angular_velocity_key", wbt[0])  # no state_base_rot in this repository
        self.assertEqual(wbt[0]["action_key"], catalog.UNITREE_WBT_SOURCE["action_key"])
        table = json.loads((data / "g1_wbt_q36" / "catalog.json").read_text())["sources"][0]
        self.assertEqual(table["data_files"], "G1_WB_Dex5_Task/data/*/*.parquet")
        self.assertEqual(Path(q36_sources[0]["path"]), (data / "g1_wbt_q36" / "hf" / "G1_WBT_Inspire_Task_MainCamOnly").resolve())
        for source in wbt + q36_sources + smpl:
            records, trajectories = load_source(source, None, FEATURES)
            self.assertEqual(records[0].graph.num_joints, 29)
            self.assertEqual(len(trajectories), 2)
            torch.testing.assert_close(trajectories[0].base_orientation[0], torch.eye(3, dtype=trajectories[0].base_orientation.dtype))
            self.assertAlmostEqual(float(trajectories[0].base_position[0, 2]), 0.75, places=5)
        left_elbow = G1_JOINTS.index("left_elbow_joint")
        _, trajectories = load_source(wbt[0], None, FEATURES)
        self.assertAlmostEqual(float(trajectories[0].positions[0, left_elbow]), 0.1, places=5)
        with self.assertRaises(ValueError):
            save_hf_dataset({**wbt[0], "path": str(data / "g1_wbt" / "raw" / "G1_WBT_Brainco_Task"), "state_key": "missing"})
        # Without the copies the sources read raw/, which is then kept; with them it can go.
        status, text = run(["-d", "g1_wbt", "--root", str(data), "--no-hf_dataset", "--delete_raw"], download=False)
        self.assertEqual(status, 0, text)
        self.assertIn("is kept", text)
        status, text = run(["-d", "g1_wbt", "--root", str(data), "--delete_raw"], download=False)
        self.assertEqual(status, 0, text)
        self.assertFalse((data / "g1_wbt" / "raw").exists())
        status, text = run(["-d", "g1_wbt", "--root", str(data), "--no-hf_dataset"], download=False)
        self.assertEqual(status, 1)
        self.assertIn("no longer exist", text)
        status, text = run(["-d", "g1_wbt", "--root", str(data), "--force"], download=False)
        self.assertEqual(status, 1)
        self.assertIn("nothing downloaded", text)


if __name__ == "__main__":
    unittest.main()
