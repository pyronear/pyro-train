"""
Camera-robustness Albumentations transforms for YOLO training.

Ultralytics' default Albumentations wrapper applies only Blur/MedianBlur/ToGray/
CLAHE at p=0.01 and leaves RandomBrightnessContrast/RandomGamma/ImageCompression
at p=0.0. We replace that list with transforms that simulate camera-sensor,
JPEG-compression, and resize artifacts, so the detector generalizes better
across heterogeneous camera feeds.

Installed by monkey-patching `ultralytics.data.augment.Albumentations.__init__`:
Ultralytics instantiates the wrapper once per dataset build with
`transforms=None`, so swapping in our list there is the least invasive hook.

Color-only jitter (hue/sat/val) is intentionally left to Ultralytics' native
`RandomHSV` (driven by the hsv_* hyperparameters) to avoid double-augmenting.
"""

from __future__ import annotations

import logging
from typing import Any

from ultralytics.data import augment as _ua

_PATCHED_FLAG = "_pyro_camera_patch_installed"


def build_camera_robustness_transforms() -> list[Any]:
    """
    Build the Albumentations transform list targeting camera/compression/resize
    robustness. All transforms here are pixel-level (non-spatial), so bounding
    boxes are untouched.
    """
    import albumentations as A

    return [
        A.ImageCompression(quality_range=(40, 95), p=0.35),
        A.Downscale(
            scale_range=(0.5, 0.9),
            interpolation_pair={"downscale": 0, "upscale": 1},
            p=0.25,
        ),
        # Gaussian noise range widened to match WARP paper (arXiv:2412.20006)
        # findings: a in [0.1, 0.4] where x' = (1-a)x + a*sigma*r.
        A.OneOf(
            [
                A.GaussNoise(std_range=(0.05, 0.20), p=1.0),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
            ],
            p=0.35,
        ),
        A.OneOf(
            [
                A.MotionBlur(blur_limit=(3, 7), p=1.0),
                A.Defocus(radius=(1, 3), alias_blur=(0.1, 0.3), p=1.0),
                A.Blur(blur_limit=(3, 5), p=1.0),
            ],
            p=0.15,
        ),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        A.RandomGamma(gamma_limit=(80, 120), p=0.2),
        A.CLAHE(clip_limit=(1.0, 3.0), tile_grid_size=(8, 8), p=0.1),
        A.Sharpen(alpha=(0.1, 0.3), lightness=(0.8, 1.2), p=0.1),
    ]


def install_camera_robustness_augmentations() -> None:
    """
    Monkey-patch Ultralytics' Albumentations wrapper so the built-in default
    transform list is replaced with camera-robustness transforms. Idempotent.
    """
    if getattr(_ua.Albumentations, _PATCHED_FLAG, False):
        return

    original_init = _ua.Albumentations.__init__

    def patched_init(self, p: float = 1.0, transforms: list | None = None) -> None:
        if transforms is None:
            try:
                transforms = build_camera_robustness_transforms()
            except ImportError:
                transforms = None
        original_init(self, p=p, transforms=transforms)

    _ua.Albumentations.__init__ = patched_init  # type: ignore[method-assign]
    setattr(_ua.Albumentations, _PATCHED_FLAG, True)
    logging.getLogger(__name__).info(
        "Installed camera-robustness Albumentations transforms"
    )
