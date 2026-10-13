"""Shared label loading and offline replay of the pinned engine's temporal scorer."""

import logging
import sys
from collections import deque
from pathlib import Path
from types import ModuleType

import numpy as np


def load_engine_class():
    """Load the engine without requiring its unused camera-controller SDK."""
    level = logging.getLogger().level
    try:
        try:
            from pyroengine.engine import Engine
        except ModuleNotFoundError as error:
            if error.name != "pyro_camera_api_client":
                raise
            client = ModuleType("pyro_camera_api_client.client")
            client.PyroCameraAPIClient = object
            package = ModuleType("pyro_camera_api_client")
            package.client = client
            sys.modules[package.__name__] = package
            sys.modules[client.__name__] = client
            try:
                from pyroengine.engine import Engine
            finally:
                del sys.modules[package.__name__]
                del sys.modules[client.__name__]
        return Engine
    finally:
        logging.getLogger().setLevel(level)


def parse_label_file(path: Path) -> np.ndarray:
    """Read normalized YOLO detections as [x1, y1, x2, y2, confidence]."""
    dets = []
    for line in path.read_text().splitlines():
        if line.strip():
            _, cx, cy, w, h, conf = (float(v) for v in line.split())
            dets.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, conf])
    return np.asarray(dets, dtype=np.float32).reshape(-1, 5)


def load_labels_dir(labels_dir: Path, min_frames: int = 0) -> dict:
    grouped = {}
    for category in ("wildfire", "fp"):
        cat_dir = labels_dir / category
        if not cat_dir.exists():
            continue
        for seq_dir in sorted(cat_dir.iterdir()):
            if seq_dir.is_dir():
                label_files = sorted((seq_dir / "labels").glob("*.txt"))
                if len(label_files) >= min_frames:
                    grouped[(category, seq_dir.name)] = [
                        parse_label_file(path) for path in label_files
                    ]
    return grouped


def reset_state(engine, cam_key: str = "-1") -> None:
    engine._states[cam_key] = {
        "last_predictions": deque(maxlen=engine.nb_consecutive_frames),
        "ongoing": False,
        "last_image_sent": None,
        "last_bbox_mask_fetch": None,
        "anchor_bbox": None,
        "anchor_ts": None,
        "miss_count": 0,
    }


class Replay:
    def __init__(self, nb_consecutive_frames: int, conf_thresh: float):
        Engine = load_engine_class()
        # Reuse temporal math without a detector, API clients, or image buffers.
        self.engine = Engine.__new__(Engine)
        self.engine.nb_consecutive_frames = nb_consecutive_frames
        self.engine.conf_thresh = conf_thresh
        self.engine._states = {}

    def trigger(self, frames) -> int | None:
        reset_state(self.engine)
        for index, pred in enumerate(frames):
            # Same width filter and empty-frame dtype as Engine.predict(fake_pred=...).
            preds = (
                pred[(pred[:, 2] - pred[:, 0]) < 0.4, :].reshape(-1, 5)
                if pred.size
                else np.empty((0, 5))
            )
            if self.engine._update_states(None, preds, "-1") > self.engine.conf_thresh:
                return index
        return None
