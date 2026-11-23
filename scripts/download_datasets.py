"""Download public robot datasets into folders you choose, and convert them.

Every dataset of ``model.conversion.catalog.CATALOG`` (``--list`` prints them with sizes and
licenses) goes to ``<root>/<name>`` (``--root``, default ``data``) or to the folder given
with ``--path NAME=DIR``. The downloads land in ``<folder>/raw`` (``--raw_root`` moves them).
Unless ``--no-convert`` is given, the datasets are then converted as by
``scripts/convert_datasets.py``: converters write the table and ``source.json``, Unitree's
LeRobot sets get one source per repository, and, unless ``--no-hf_dataset`` is given, every
source is also saved as a Hugging Face dataset (``hf/``). ``<folder>/sources.json`` lists
the sources to paste into a data config. Files that are already present are not downloaded
again, and a finished conversion is not redone, so an interrupted run can be restarted.

The datasets belong to their creators and keep their licenses (``docs/data.md``); several
allow research use only. Robot models are fetched with ``scripts/download_urdfs.py``.

Examples::

    uv sync --extra convert --extra validation
    uv run python scripts/download_datasets.py --list
    uv run python scripts/download_datasets.py --all --root /mnt/data/bfm --dry_run
    uv run python scripts/download_datasets.py --dataset lafan1_g1,xwbc_g1_lafan
    uv run python scripts/download_datasets.py --dataset amass_g1 --path amass_g1=/mnt/big/amass_g1 --pattern "*Walk*" --limit 200
    uv run python scripts/download_datasets.py --dataset g1_wbt --no-convert --workers 16
    uv run python scripts/download_datasets.py --dataset xwbc_g1_amass --no-hf_dataset --delete_raw
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.conversion.catalog import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(download=True, description=__doc__.splitlines()[0]))
