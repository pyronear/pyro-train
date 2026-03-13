"""
CLI script to train a YOLO model for the object detection task of
fire smokes.
"""

import argparse
import logging
import shutil
from pathlib import Path

from ultralytics import settings

from pyro_train.data.utils import yaml_read, yaml_write
from pyro_train.model.yolo.train import load_pretrained_model, train
from pyro_train.utils import resolve_device


def resolve_data_yaml(data_yaml_path: Path) -> Path:
    """
    Ultralytics resolves a relative `path:` key in data.yaml against its own
    datasets_dir, not the yaml file location.  When the key is present and
    relative, rewrite it to an absolute path in a sibling file so YOLO can
    find the images regardless of where the script is run from.
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
    """
    Make the CLI parser.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        help="filepath to the data_yaml config file for the dataset",
        default="./data/01_model_input/wildfire/small/datasets/data.yaml",
        type=Path,
    )
    parser.add_argument(
        "--output-dir",
        help="path to save the model_artifacts",
        default="./data/02_models/yolo/",
        type=Path,
    )
    parser.add_argument(
        "--experiment-name",
        help="experiment name",
        default="my_experiment",
        type=str,
    )
    parser.add_argument(
        "--config",
        help="Yaml configuration file to train the model on",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--device",
        help="Device to use for training (cuda, mps, cpu). Defaults to best available.",
        default=None,
        type=str,
    )
    parser.add_argument(
        "-log",
        "--loglevel",
        default="warning",
        help="Provide logging level. Example --loglevel debug, default=warning",
    )
    return parser


def validate_parsed_args(args: dict) -> bool:
    """
    Return whether the parsed args are valid.
    """
    if not args["data"].exists():
        logging.error("Invalid --data filepath does not exist")
        return False
    elif not args["config"].exists():
        logging.error("Invalid --config filepath does not exist")
        return False
    else:
        return True


if __name__ == "__main__":
    cli_parser = make_cli_parser()
    args = vars(cli_parser.parse_args())
    logging.basicConfig(level=args["loglevel"].upper())
    if not validate_parsed_args(args):
        logging.error(f"Could not validate the parsed args: {args}")
        exit(1)
    else:
        logging.info(args)
        params = yaml_read(args["config"])
        logging.info(f"Parsed run params: {params}")
        model_type = params["model_type"]
        logging.info(f"Loading model: {model_type}")
        model = load_pretrained_model(model_type)
        # Cleaning the train run directory
        shutil.rmtree(args["output_dir"] / args["experiment_name"], ignore_errors=True)

        # Update ultralytics settings to log with MLFlow
        settings.update({"mlflow": True})

        device = args["device"] or resolve_device()
        logging.info(f"Using device: {device}")

        train(
            model=model,
            data_yaml_path=resolve_data_yaml(args["data"]),
            params=params,
            device=device,
            project=str(args["output_dir"]),
            experiment_name=args["experiment_name"],
        )
        exit(0)
