"""
CLI script to evaluate a trained YOLO model on a held-out test set.
"""

import argparse
import json
import logging
import os
from pathlib import Path

from pyro_train.data.utils import yaml_read, yaml_write
from pyro_train.model.yolo.train import load_pretrained_model
from pyro_train.utils import resolve_device


def resolve_data_yaml(data_yaml_path: Path) -> Path:
    """
    Rewrite a relative `path:` key in data.yaml to absolute so YOLO can find
    images regardless of the working directory.
    """
    content = yaml_read(data_yaml_path)
    if "path" not in content:
        return data_yaml_path
    dataset_path = Path(content["path"])
    if dataset_path.is_absolute():
        return data_yaml_path
    content["path"] = str((data_yaml_path.parent / dataset_path).resolve())
    patched_path = data_yaml_path.parent / "_data.yaml"
    yaml_write(to=patched_path, data=content)
    return patched_path


def make_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        help="directory containing the YOLO train run (must contain weights/best.pt)",
        default="./data/02_models/yolo/best",
        type=Path,
    )
    parser.add_argument(
        "--data",
        help="filepath to the data.yaml for the test dataset",
        default="./data/test/yolo_test/data.yaml",
        type=Path,
    )
    parser.add_argument(
        "--output-dir",
        help="directory to save evaluation results",
        default="./data/03_reporting/yolo/eval/",
        type=Path,
    )
    parser.add_argument(
        "--split",
        help="dataset split to evaluate on (test, val)",
        default="test",
        type=str,
    )
    parser.add_argument(
        "--device",
        help="device to use (cuda, mps, cpu). Defaults to best available.",
        default=None,
        type=str,
    )
    parser.add_argument(
        "-log",
        "--loglevel",
        default="warning",
        help="Provide logging level. Example --loglevel debug, default=warning.",
    )
    return parser


def validate_parsed_args(args: dict) -> bool:
    weights_path = args["model_dir"] / "weights" / "best.pt"
    if not weights_path.exists():
        logging.error(f"Model weights not found at {weights_path}")
        return False
    if not args["data"].exists():
        logging.error(f"--data file does not exist: {args['data']}")
        return False
    return True


if __name__ == "__main__":
    cli_parser = make_cli_parser()
    args = vars(cli_parser.parse_args())
    logging.basicConfig(level=args["loglevel"].upper())
    if not validate_parsed_args(args):
        logging.error(f"Could not validate the parsed args: {args}")
        exit(1)

    weights_path = args["model_dir"] / "weights" / "best.pt"
    logging.info(f"Loading model from {weights_path}")
    model = load_pretrained_model(str(weights_path))

    device = args["device"] or resolve_device()
    logging.info(f"Using device: {device}")

    data_yaml = resolve_data_yaml(args["data"])
    logging.info(f"Evaluating on {data_yaml} (split={args['split']})")

    metrics = model.val(
        data=str(data_yaml.absolute()),
        split=args["split"],
        device=device,
    )

    os.makedirs(args["output_dir"], exist_ok=True)

    results = {
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "model_dir": str(args["model_dir"]),
        "data": str(args["data"]),
        "split": args["split"],
    }

    output_path = args["output_dir"] / "eval_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logging.info(f"Results saved to {output_path}")
    print(
        f"mAP@50: {results['map50']:.4f}  |  "
        f"mAP@50-95: {results['map50_95']:.4f}  |  "
        f"P: {results['precision']:.4f}  |  "
        f"R: {results['recall']:.4f}"
    )
    exit(0)
