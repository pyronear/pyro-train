"""
CLI script to fetch the public pyro-sdis dataset from Hugging Face and export
it to the folder layout expected by build.py:

    <output-dir>/images/{train,val}/<image_name>
    <output-dir>/labels/{train,val}/<image_stem>.txt
"""

import argparse
import logging
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import snapshot_download

SPLITS = ["train", "val"]


def make_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="pyronear/pyro-sdis")
    parser.add_argument("--revision", default="v2.0", help="git tag of the dataset")
    parser.add_argument(
        "--output-dir",
        help="path to save the exported dataset",
        default="./data/01_model_input/yolo_train_val",
        type=Path,
    )
    parser.add_argument(
        "-log",
        "--loglevel",
        default="warning",
        help="Provide logging level. Example --loglevel debug, default=warning",
    )
    return parser


def export_split(parquet_files: list[Path], output_dir: Path, split: str) -> int:
    """
    Write the images and YOLO labels stored in `parquet_files` under
    `output_dir`. Returns the number of exported images.
    """
    images_dir = output_dir / "images" / split
    labels_dir = output_dir / "labels" / split
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for parquet_file in parquet_files:
        # Small batches: each row embeds a full image.
        batches = pq.ParquetFile(parquet_file).iter_batches(
            batch_size=64, columns=["image", "annotations", "image_name"]
        )
        for batch in batches:
            for row in batch.to_pylist():
                # .name guards against path separators in the dataset column.
                name = Path(row["image_name"]).name
                (images_dir / name).write_bytes(row["image"]["bytes"])
                # Images without smoke keep an empty label file.
                (labels_dir / name).with_suffix(".txt").write_text(
                    row["annotations"] or ""
                )
                count += 1
    return count


if __name__ == "__main__":
    args = make_cli_parser().parse_args()
    logging.basicConfig(level=args.loglevel.upper())
    snapshot_dir = Path(
        snapshot_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            revision=args.revision,
            allow_patterns="data/*.parquet",
        )
    )
    for split in SPLITS:
        parquet_files = sorted((snapshot_dir / "data").glob(f"{split}-*.parquet"))
        if not parquet_files:
            raise SystemExit(f"No parquet file for split {split} in {snapshot_dir}")
        count = export_split(parquet_files, args.output_dir, split)
        logging.info(f"Exported {count} {split} images to {args.output_dir}")
