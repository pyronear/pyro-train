"""
Scan all sequences in a raw fp directory, predict on first N frames of sequences
with >= min_images, run the engine, and visualize alerting sequences in FiftyOne.

Predictions are cached in --labels-dir so subsequent runs skip inference.
Use --force to re-predict.

Expected input layout:
  <data-dir>/<sequence>/images/*.jpg

Usage:
    uv run python scan_raw_fp.py
    uv run python scan_raw_fp.py --data-dir /path/to/fp/data --force
"""

import argparse
import sys
from collections import deque
from pathlib import Path
from types import ModuleType

# ── Stub optional pyroengine dependencies ────────────────────────────────────
_Stub = type("_Stub", (), {"__init__": lambda *_, **__: None})
for _pkg, _attr, _cls in [
    ("pyro_camera_api_client", "client", "PyroCameraAPIClient"),
    ("pyroclient", "client", "PyroClient"),
]:
    _sub = ModuleType(f"{_pkg}.{_attr}")
    setattr(_sub, _cls, _Stub)
    _top = ModuleType(_pkg)
    setattr(_top, _attr, _sub)
    sys.modules.setdefault(_pkg, _top)
    sys.modules.setdefault(f"{_pkg}.{_attr}", _sub)

import logging  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
import pyroengine.engine  # noqa: E402
from pyroengine.engine import Engine  # noqa: E402

# Engine.__init__ builds a Classifier, which downloads a detector from
# HuggingFace. Nothing here runs it: _make_engine swaps engine.model for
# _PassthroughModel one line later, because this script replays predictions
# computed upstream. The download was always dead weight, and became fatal once
# the model repository stopped answering anonymously (HTTP 401).
pyroengine.engine.Classifier = _Stub
from tqdm import tqdm  # noqa: E402
from ultralytics import YOLO  # type: ignore[attr-defined]  # noqa: E402

import fiftyone as fo  # noqa: E402

logging.getLogger().setLevel(logging.WARNING)
logging.getLogger("pyroengine").setLevel(logging.WARNING)
_DUMMY_FRAME = Image.new("RGB", (1, 1))


# ── Device ────────────────────────────────────────────────────────────────────

def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ── Prediction helpers ────────────────────────────────────────────────────────

def predict_sequence(model: YOLO, images: list[Path], device: str, imgsz: int, labels_dir: Path) -> None:
    """Run YOLO on each image and write label files."""
    for img_path in images:
        label_path = labels_dir / img_path.with_suffix(".txt").name
        label_path.parent.mkdir(parents=True, exist_ok=True)
        results = model.predict(source=str(img_path), device=device, verbose=False,
                                iou=0.01, conf=0.05, imgsz=imgsz)
        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            xywhn = boxes.xywhn.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            lines = [f"0 {xywhn[i][0]:.6f} {xywhn[i][1]:.6f} {xywhn[i][2]:.6f} {xywhn[i][3]:.6f} {confs[i]:.6f}\n"
                     for i in range(len(xywhn))]
            label_path.write_text("".join(lines))
        else:
            label_path.write_text("")


def parse_label_file(path: Path) -> np.ndarray:
    dets = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        _, cx, cy, w, h, conf = (float(v) for v in line.split())
        dets.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, conf])
    return np.array(dets, dtype=np.float32) if dets else np.empty((0, 5), dtype=np.float32)


# ── Engine helpers ────────────────────────────────────────────────────────────

class _PassthroughModel:
    def post_process(self, fake_pred: np.ndarray, **_: object) -> np.ndarray:
        return fake_pred


def make_engine(nb_consecutive_frames: int, conf_thresh: float) -> Engine:
    cache = Path("/tmp/pyro_scan_raw_fp_cache")
    cache.mkdir(parents=True, exist_ok=True)
    engine = Engine(conf_thresh=conf_thresh, nb_consecutive_frames=nb_consecutive_frames,
                    cache_folder=str(cache))
    engine.model = _PassthroughModel()  # type: ignore[assignment]
    return engine


def reset_state(engine: Engine, cam_key: str) -> None:
    engine._states[cam_key] = {  # noqa: SLF001
        "last_predictions": deque(maxlen=engine.nb_consecutive_frames),
        "ongoing": False, "last_image_sent": None, "last_bbox_mask_fetch": None,
        "anchor_bbox": None, "anchor_ts": None, "miss_count": 0,
    }
    engine.occlusion_masks[cam_key] = (None, {}, 0)  # noqa: SLF001


def engine_alerts(engine: Engine, label_files: list[Path], cam_key: str) -> bool:
    reset_state(engine, cam_key)
    for lf in label_files:
        pred = parse_label_file(lf)
        if engine.predict(_DUMMY_FRAME, cam_id=cam_key, fake_pred=pred) > engine.conf_thresh:
            return True
    return False


# ── FiftyOne helpers ──────────────────────────────────────────────────────────

def load_detections(txt_path: Path) -> fo.Detections:
    dets = []
    if txt_path.exists():
        for line in txt_path.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            _, cx, cy, w, h, conf = (float(p) for p in parts)
            dets.append(fo.Detection(label="fire",
                                     bounding_box=[cx - w / 2, cy - h / 2, w, h],
                                     confidence=conf))
    return fo.Detections(detections=dets)


def build_fiftyone_dataset(alerting_seqs: list[tuple[str, list[Path], Path]],
                            dataset_name: str) -> fo.Dataset:
    if fo.dataset_exists(dataset_name):
        fo.delete_dataset(dataset_name)
    dataset = fo.Dataset(dataset_name)
    samples = []
    for seq_name, images, labels_dir in alerting_seqs:
        for img_path in images:
            sample = fo.Sample(filepath=str(img_path))
            sample["sequence"] = seq_name
            sample["predictions"] = load_detections(labels_dir / img_path.with_suffix(".txt").name)
            samples.append(sample)
    dataset.add_samples(samples)
    dataset.persistent = True
    return dataset


# ── Main ──────────────────────────────────────────────────────────────────────

def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Scan raw fp sequences, predict, and visualize alerts")
    parser.add_argument("--data-dir", type=Path,
                        default=Path("/Users/mateo/pyronear/vision/dataset/pyro-dataset/data/raw/fp/data"))
    parser.add_argument("--model-path", type=Path,
                        default=Path("data/02_models/yolo/best/weights/best.pt"))
    parser.add_argument("--labels-dir", type=Path, default=Path("predictions_labels_raw_fp"),
                        help="Cache directory for prediction label files")
    parser.add_argument("--min-images", type=int, default=8,
                        help="Min images in sequence to include (default: 8)")
    parser.add_argument("--max-frames", type=int, default=10,
                        help="Max frames to predict per sequence (default: 10)")
    parser.add_argument("--nb-consecutive-frames", type=int, default=5)
    parser.add_argument("--conf-thresh", type=float, default=0.2)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--force", action="store_true",
                        help="Re-run predictions even if cached labels exist")
    parser.add_argument("--port", type=int, default=5151)
    return parser


if __name__ == "__main__":
    args = make_parser().parse_args()

    if not args.data_dir.exists():
        raise SystemExit(f"Data dir not found: {args.data_dir}")
    if not args.model_path.exists():
        raise SystemExit(f"Model not found: {args.model_path}")

    device = args.device or pick_device()

    # ── Collect eligible sequences ────────────────────────────────────────────
    sequences = sorted(d for d in args.data_dir.iterdir() if d.is_dir())
    eligible = []
    for seq_dir in sequences:
        images = sorted((seq_dir / "images").glob("*.jpg")) + \
                 sorted((seq_dir / "images").glob("*.png"))
        if len(images) >= args.min_images:
            eligible.append((seq_dir.name, images[:args.max_frames]))

    print(f"Found {len(sequences)} sequences, {len(eligible)} with >= {args.min_images} images")

    # ── Predict (with caching) ────────────────────────────────────────────────
    model = None
    to_predict = [(name, imgs) for name, imgs in eligible
                  if args.force or not (args.labels_dir / name / imgs[0].with_suffix(".txt").name).exists()]

    if to_predict:
        print(f"\nRunning predictions on {len(to_predict)} sequences (device: {device})...")
        model = YOLO(str(args.model_path))
        for seq_name, images in tqdm(to_predict, desc="predicting"):
            predict_sequence(model, images, device, args.imgsz,
                             args.labels_dir / seq_name)
    else:
        print("All predictions cached — skipping inference (use --force to re-predict)")

    # ── Run engine, find alerting sequences ───────────────────────────────────
    print(f"\nRunning engine (nb_frames={args.nb_consecutive_frames}, conf={args.conf_thresh})...")
    engine = make_engine(args.nb_consecutive_frames, args.conf_thresh)
    alerting = []
    for seq_name, images in tqdm(eligible, desc="engine"):
        seq_labels_dir = args.labels_dir / seq_name
        label_files = [seq_labels_dir / img.with_suffix(".txt").name for img in images]
        if engine_alerts(engine, label_files, seq_name):
            alerting.append((seq_name, images, seq_labels_dir))

    print(f"\nAlerting sequences: {len(alerting)} / {len(eligible)} ({len(alerting)/len(eligible):.1%})")

    # ── Visualize in FiftyOne ─────────────────────────────────────────────────
    if not alerting:
        print("No alerting sequences — nothing to visualize.")
    else:
        dataset_name = "raw_fp_alerts"
        ds = build_fiftyone_dataset(alerting, dataset_name)
        print(f"\nLaunching FiftyOne with {len(ds)} frames → http://localhost:{args.port}")
        print("Press Ctrl+C to stop.")
        session = fo.launch_app(ds, port=args.port, auto=False)
        try:
            session.wait()
        except KeyboardInterrupt:
            pass
