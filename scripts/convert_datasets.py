"""Convert datasets that scripts/download_datasets.py downloaded, without the network.

Takes the same dataset names and folder options as ``scripts/download_datasets.py``
(``--dataset``/``--all``, ``--root``, ``--path NAME=DIR``, ``--raw_root``, ``--pattern``,
``--limit``) and converts the files found in each download folder:

* converted datasets go through ``model.conversion.datasets`` and get ``data/``, ``meta/``
  and ``source.json`` in their folder;
* Unitree's LeRobot sets get one source per repository (the ``huggingface`` source reads
  them as they are);
* with ``--hf_dataset`` (the default; ``--no-hf_dataset`` turns it off) every source is also
  saved as a Hugging Face dataset with ``datasets.Dataset.save_to_disk`` in ``hf/``, which
  the ``huggingface`` source opens memory-mapped.

``<folder>/sources.json`` lists the sources (the Hugging Face copies when they exist), ready
to paste into a data config. A conversion recorded in ``<folder>/catalog.json`` with the same
files and selection is not redone unless ``--force`` is given, so running this again only
adds what is missing, e.g. the Hugging Face copies. Converting one dataset with every option
of its converter is ``scripts/convert_dataset.py``.

Examples::

    uv run python scripts/convert_datasets.py --dataset amass_g1 --root /mnt/data/bfm
    uv run python scripts/convert_datasets.py --all --root /mnt/data/bfm --no-hf_dataset
    uv run python scripts/convert_datasets.py --dataset g1_wbt_smpl --pattern "*Inspire*" --force
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.conversion.catalog import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(download=False, description=__doc__.splitlines()[0]))
