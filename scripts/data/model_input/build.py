"""
CLI script to generate model input for YOLO custom dataset training.
"""

import argparse
import logging
import os
import random
import shutil
from pathlib import Path

from pyro_train.data.utils import yaml_write


def make_cli_parser() -> argparse.ArgumentParser:
    """
    Make the CLI parser.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        help="path pointing to the raw dataset",
        default="./data/01_model_input/wildfire",
        type=Path,
    )
    parser.add_argument(
        "--output-dir",
        help="path to save the model_input",
        default="./data/01_model_input/wildfire",
        type=Path,
    )
    parser.add_argument(
        "--sampling-ratio",
        help="sampling ratio for the small version of the dataset",
        default=0.05,
        type=float,
    )
    parser.add_argument(
        "--random-seed",
        help="Random seed to sample the dataset fo r the small version",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--exclude-background",
        help="Exclude background images (empty label files) from the train split to train an ultra-sensitive model",
        action="store_true",
    )
    parser.add_argument(
        "-log",
        "--loglevel",
        default="warning",
        help="Provide logging level. Example --loglevel debug, default=warning",
    )
    return parser


def validate_parsed_args(args: dict) -> bool:
    """Returns whether the parsed args are valid."""
    if not args["input_dir"].exists():
        logging.error("Invalid --input-dir directory, it does not exist")
        return False
    else:
        return True


def write_data_yaml(yaml_filepath: Path) -> None:
    content = {
        "train": "./train/images",
        "val": "./val/images",
        "test": ".test/images",
        "nc": 1,
        "names": ["smoke"],
    }
    yaml_write(to=yaml_filepath, data=content)


def make_yolov8_folder_structure(dir: Path) -> None:
    """Creates the YOLOv8 data folder structure expected to train it on a
    custom dataset.

    $ tree -L 2 .
    .
    ├── test
    │   ├── images
    │   └── labels
    ├── train
    │   ├── images
    │   └── labels
    └── val
        ├── images
        └── labels
    """
    os.makedirs(dir / "datasets", exist_ok=True)

    for split in ["train", "val", "test"]:
        os.makedirs(dir / "datasets" / split / "images", exist_ok=True)
        os.makedirs(dir / "datasets" / split / "labels", exist_ok=True)


def copy_data(input_dir: Path, output_dir: Path) -> None:
    """
    Copy over all data from `input_dir` to `output_dir` using the YOLOv8
    folder structure conventions.
    """
    for split in ["train", "val"]:
        shutil.copytree(
            src=input_dir / "images" / split,
            dst=output_dir / split / "images",
            dirs_exist_ok=True,
        )
        shutil.copytree(
            src=input_dir / "labels" / split,
            dst=output_dir / split / "labels",
            dirs_exist_ok=True,
        )


def is_background(label_filepath: Path) -> bool:
    """
    Return whether the given label file corresponds to a background image,
    ie. a missing or empty label file (no smoke annotation).
    """
    return not label_filepath.exists() or label_filepath.stat().st_size == 0


def sample_dataset(
    input_dir: Path,
    output_dir: Path,
    sampling_ratio: float = 0.1,
    random_seed: int = 0,
    exclude_background: bool = False,
) -> list[dict]:
    """
    Return a downsampled list of images and labels for the given
    `sampling_ratio` and `random_seed`.

    Each element in the returned list has the following keys:
    - to: Path - where the image/label comes from
    - from: Path - where the image/label should be copied over - using the provided `output_dir`

    When `exclude_background` is set, background images (empty label files)
    are dropped from the train split to train an ultra-sensitive model.
    """
    assert 0 <= sampling_ratio <= 1.0, "sampling ratio should be between 0 and 1"

    result = []
    for split in ["train", "val"]:
        labels_split_dir = input_dir / "labels" / split
        images_split_dir = input_dir / "images" / split
        images_filepaths = list(images_split_dir.glob("*.jpg"))
        # Drop background images (empty labels) from the train split only, so
        # that the val split stays intact for a comparable evaluation.
        if exclude_background and split == "train":
            n_before = len(images_filepaths)
            images_filepaths = [
                fp
                for fp in images_filepaths
                if not is_background(labels_split_dir / f"{fp.stem}.txt")
            ]
            logging.info(
                f"Excluded {n_before - len(images_filepaths)} background images "
                f"from the {split} split ({len(images_filepaths)} remaining)"
            )
        n_images = len(images_filepaths)
        k = int(n_images * sampling_ratio)
        # For the val split we do not subsample as we want to eval on the same data as in the full
        downsampled_image_filepaths = images_filepaths
        if split == "train":
            downsampled_image_filepaths = random.Random(random_seed).sample(
                images_filepaths,
                k=k,
            )
        downsampled_label_filepaths = [
            labels_split_dir / f"{fp.stem}.txt"
            for fp in downsampled_image_filepaths
            if (labels_split_dir / f"{fp.stem}.txt").exists()
        ]
        copy_data_images = [
            {"type": "image", "from": fp, "to": output_dir / split / "images" / fp.name}
            for fp in downsampled_image_filepaths
        ]
        copy_data_labels = [
            {"type": "label", "from": fp, "to": output_dir / split / "labels" / fp.name}
            for fp in downsampled_label_filepaths
        ]
        result.extend(copy_data_images)
        result.extend(copy_data_labels)

    return result


def run_file_copy(copy_data: list[dict]) -> None:
    """
    Run the file copy instructions for each element of copy_data.

    Each element should contain a from and to key. If not present, a
    warning is printed and the element is skipped.
    """
    for e in copy_data:
        if "from" not in e or "to" not in e:
            logging.warning(
                f"Skipping file copy - Missing `from` key or `to` key from element: {e}"
            )
        else:
            os.makedirs(e["to"].parent, exist_ok=True)
            shutil.copy(src=e["from"], dst=e["to"])


if __name__ == "__main__":
    cli_parser = make_cli_parser()
    args = vars(cli_parser.parse_args())
    logging.basicConfig(level=args["loglevel"].upper())
    if not validate_parsed_args(args):
        logging.error(f"Could not validate the parsed args: {args}")
        exit(1)
    else:
        logging.info(args)
        input_dir = args["input_dir"]
        output_dir = args["output_dir"]
        sampling_ratio = args["sampling_ratio"]
        random_seed = args["random_seed"]

        logging.info(f"Creating dirs at {output_dir}")
        shutil.rmtree(output_dir, ignore_errors=True)
        make_yolov8_folder_structure(output_dir)

        logging.info(f"Creating {sampling_ratio*100:.0f}% sample at {output_dir}")
        run_file_copy(
            sample_dataset(
                input_dir=input_dir,
                output_dir=output_dir / "datasets",
                sampling_ratio=sampling_ratio,
                random_seed=random_seed,
                exclude_background=args["exclude_background"],
            )
        )
        write_data_yaml(output_dir / "datasets" / "data.yaml")

        exit(0)
