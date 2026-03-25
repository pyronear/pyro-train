"""
Standalone script: run per-frame YOLO predictions on a sequential dataset and
save results as YOLO label files (one .txt per frame).

Note: MPS is disabled due to a bounding box corruption bug in ultralytics 8.4.21.

Dataset layout expected under --data-dir:
  <data-dir>/wildfire/<sequence>/images/*.jpg
  <data-dir>/fp/<sequence>/images/*.jpg

Output layout under --labels-dir:
  <labels-dir>/wildfire/<sequence>/labels/<frame>.txt
  <labels-dir>/fp/<sequence>/labels/<frame>.txt

Each .txt: one line per detection: 0 cx cy w h conf  (all normalized, class=0)
Empty file if no detections on that frame.
"""

import argparse
import logging
import shutil
from pathlib import Path

import torch
from tqdm import tqdm
from ultralytics import YOLO  # type: ignore[attr-defined]

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def all_detections(result) -> list[tuple[float, float, float, float, float]]:
    """Return list of (cx, cy, w, h, conf) sorted by conf desc."""
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []
    xywhn = boxes.xywhn.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    order = confs.argsort()[::-1]
    return [(float(xywhn[i][0]), float(xywhn[i][1]), float(xywhn[i][2]), float(xywhn[i][3]), float(confs[i])) for i in order]


def write_label_file(label_path: Path, dets: list[tuple[float, float, float, float, float]]) -> None:
    """Write detections in YOLO format: 0 cx cy w h conf per line. Empty if no detections."""
    label_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {conf:.6f}\n" for cx, cy, w, h, conf in dets]
    label_path.write_text("".join(lines))


def predict_category(
    model: YOLO,
    category_dir: Path,
    category: str,
    max_frames: int,
    device: str,
    labels_dir: Path,
) -> None:
    sequences = sorted(d for d in category_dir.iterdir() if d.is_dir())
    for seq_dir in tqdm(sequences, desc=category, unit="seq"):
        images_dir = seq_dir / "images"
        if not images_dir.exists():
            logging.warning(f"No images/ folder in {seq_dir}, skipping")
            continue
        frames = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))
        for frame_path in frames[:max_frames]:
            results = model.predict(source=str(frame_path), device=device, verbose=False, iou=0.01, conf=0.05, imgsz=1024)
            dets = all_detections(results[0])
            label_path = labels_dir / category / seq_dir.name / "labels" / frame_path.with_suffix(".txt").name
            write_label_file(label_path, dets)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Per-frame YOLO predictions → YOLO label files")
    parser.add_argument("--model-path", type=Path, required=True,
                        help="Path to YOLO .pt weights file")
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Root dir with wildfire/ and fp/ subfolders")
    parser.add_argument("--labels-dir", type=Path, default=Path("predictions_labels"),
                        help="Output directory for label files (default: predictions_labels/)")
    parser.add_argument("--max-frames", type=int, default=15)
    parser.add_argument("--device", type=str, default=None,
                        help="cuda / cpu (default: auto, MPS disabled)")
    return parser


if __name__ == "__main__":
    args = make_parser().parse_args()

    if not args.model_path.exists():
        raise SystemExit(f"Model not found: {args.model_path}")
    if not args.data_dir.exists():
        raise SystemExit(f"Data dir not found: {args.data_dir}")
    for split in ("wildfire", "fp"):
        if not (args.data_dir / split).exists():
            raise SystemExit(f"Expected subfolder '{split}' not found in {args.data_dir}")

    if args.labels_dir.exists():
        shutil.rmtree(args.labels_dir)

    device = args.device or pick_device()
    print(f"Device: {device}  →  {args.labels_dir}")

    model = YOLO(str(args.model_path))

    predict_category(model, args.data_dir / "wildfire", "wildfire", args.max_frames, device, args.labels_dir)
    predict_category(model, args.data_dir / "fp", "fp", args.max_frames, device, args.labels_dir)

    print("Done.")
