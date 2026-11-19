"""Export a tiny pyro-sdis-like parquet file to the YOLO folder layout."""

import importlib.util
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/data/model_input/fetch_hf.py"


def test_export_split(tmp_path):
    spec = importlib.util.spec_from_file_location("fetch_hf", SCRIPT)
    fetch_hf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fetch_hf)

    parquet_file = tmp_path / "train-00000-of-00001.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "image": {"bytes": b"smoke", "path": "a.jpg"},
                    "annotations": "0 0.5 0.5 0.1 0.2",
                    "image_name": "a.jpg",
                },
                {
                    "image": {"bytes": b"empty", "path": "b.jpg"},
                    "annotations": "",
                    "image_name": "../b.jpg",
                },
            ]
        ),
        parquet_file,
    )

    output_dir = tmp_path / "out"
    assert fetch_hf.export_split([parquet_file], output_dir, "train") == 2

    assert (output_dir / "images/train/a.jpg").read_bytes() == b"smoke"
    assert (output_dir / "labels/train/a.txt").read_text() == "0 0.5 0.5 0.1 0.2"
    assert (output_dir / "images/train/b.jpg").read_bytes() == b"empty"
    assert (output_dir / "labels/train/b.txt").read_text() == ""
