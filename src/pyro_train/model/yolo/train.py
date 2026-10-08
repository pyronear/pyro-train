"""
Module to train the YOLO models.
"""

from pathlib import Path

from ultralytics import YOLO

from pyro_train.model.yolo.augment import install_camera_robustness_augmentations
from pyro_train.utils import resolve_device


def load_pretrained_model(model_str: str) -> YOLO:
    """
    Loads the pretrained `model`
    """
    return YOLO(model_str)


def train(
    model: YOLO,
    data_yaml_path: Path,
    params: dict,
    device: str | None = None,
    project: str = "data/04_models/yolo/",
    experiment_name: str = "train",
):
    """
    Main function for running a train run.
    """
    assert data_yaml_path.exists(), f"data_yaml_path does not exist, {data_yaml_path}"
    install_camera_robustness_augmentations()
    default_params = {
        # train parameters
        "batch": 16,
        "cos_lr": False,
        "epochs": 100,
        "imgsz": 640,
        "lr0": 0.01,
        "lrf": 0.01,
        "optimizer": "auto",
        "patience": 100,
        "warmup_epochs": 3,
        # eval parameters
        "box": 7.5,
        "cls": 0.5,
        "dfl": 1.5,
        "iou": 0.6,
        "single_cls": True,
        # data augmentation parameters
        "close_mosaic": 10,
        "degrees": 0.0,
        "fliplr": 0.5,
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "mixup": 0.0,
        "shear": 0.0,
        "translate": 0.1,
    }
    params = {**default_params, **params}
    params.pop("model_type", None)
    params.update(
        project=str(Path(project).resolve()),
        name=experiment_name,
        data=data_yaml_path.absolute(),
        device=device if device is not None else params.get("device", resolve_device()),
    )
    model.train(**params)
