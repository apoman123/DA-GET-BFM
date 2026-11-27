"""Build the processed-data cache used before physics pretraining.

Pass the same training JSON file to this script and ``train_physics_da_get.py``.
The data config, cache directory, and worker count come from that file, so a
later training run can reuse the cache without rebuilding it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.training import build_physics_datasets  # noqa: E402


def main(argv: Sequence[str] | None = None) -> None:
    """Build the exact cache that physics pretraining loads before training.

    Args:
        argv (Sequence[str] | None): Optional CLI arguments. The first argument
            is a pretraining JSON path; when omitted, use the selected large
            configuration. ``--workers`` optionally overrides its positive
            preprocessing worker count.

    Returns:
        None: Writes or reuses an Arrow cache and prints its dataset counts.

    Raises:
        FileNotFoundError: The pretraining or referenced data config is missing.
        json.JSONDecodeError: The pretraining or data config is invalid JSON.
        ValueError: The pretraining config has invalid data settings, requests
            an automatic cache rebuild, or cannot produce training data.
        OSError: The cache cannot be read or written.
        SystemExit: CLI arguments are invalid.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "training_config",
        nargs="?",
        type=Path,
        default=Path("configs/pretraining/selected_large.json"),
        help="Pretraining JSON passed to train_physics_da_get.py.",
    )
    parser.add_argument(
        "--workers", type=int, help="Override preprocess_workers in the JSON."
    )
    options = parser.parse_args(argv)
    config = json.loads(options.training_config.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("pretraining JSON must contain an object")
    data_config = config.get("data_config")
    if not isinstance(data_config, str) or not data_config:
        raise ValueError("pretraining JSON must contain a nonempty data_config path")
    workers = (
        options.workers
        if options.workers is not None
        else config.get("preprocess_workers")
    )
    if workers is not None and (not isinstance(workers, int) or workers < 1):
        raise ValueError("preprocess_workers must be a positive integer")
    cache_dir = config.get("cache_dir")
    if cache_dir is not None and not isinstance(cache_dir, str):
        raise ValueError("cache_dir must be a path string or null")
    if config.get("rebuild_cache", False):
        raise ValueError("set rebuild_cache to false so training reuses this cache")

    datasets = build_physics_datasets(
        data_config,
        cache_dir=cache_dir,
        workers=workers,
        progress=True,
    )
    print(
        f"Preprocessing ready: {len(datasets.records)} embodiments, "
        f"{len(datasets.train)} training samples, "
        f"{len(datasets.eval) if datasets.eval is not None else 0} evaluation samples"
    )


if __name__ == "__main__":
    main()
