"""
Find the best conf threshold for frame-by-frame YOLO evaluation on the val split.

Runs inference once at a very low conf (to capture all detections), then sweeps
thresholds post-hoc to find the best F1 on val.

Output TSV columns: conf_thresh, tp, fn, fp, tn, precision, recall, f1, fpr
Sorted by f1 descending.

Usage:
    uv run python optimize_yolo_val.py
    uv run python optimize_yolo_val.py --model-path <model.pt> --output grid_search_yolo_val.tsv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from ultralytics import YOLO  # type: ignore[attr-defined]


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def has_gt(label_path: Path) -> bool:
    return label_path.exists() and label_path.stat().st_size > 0


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Grid-search conf threshold on YOLO val split")
    parser.add_argument("--model-path", type=Path,
                        default=Path("data/02_models/yolo/best/weights/best.pt"))
    parser.add_argument("--data-dir", type=Path,
                        default=Path("data/01_model_input/yolo_train_val"))
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--output", type=Path, default=Path("grid_search_yolo_val.tsv"))
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--device", type=str, default=None)
    return parser


if __name__ == "__main__":
    args = make_parser().parse_args()

    images_dir = args.data_dir / "images" / args.split
    labels_dir = args.data_dir / "labels" / args.split

    if not images_dir.exists():
        raise SystemExit(f"Images dir not found: {images_dir}")
    if not args.model_path.exists():
        raise SystemExit(f"Model not found: {args.model_path}")

    device = args.device or pick_device()
    print(f"Model: {args.model_path}  Split: {args.split}  Device: {device}")

    model = YOLO(str(args.model_path))
    images = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))

    # Run inference once at very low conf to capture all detections
    print(f"\nRunning inference on {len(images)} images at conf=0.01...")
    raw: list[tuple[bool, float | None]] = []  # (is_wildfire, max_conf_or_None)
    for img_path in tqdm(images, desc="predicting"):
        label_path = labels_dir / img_path.with_suffix(".txt").name
        is_wildfire = has_gt(label_path)
        results = model.predict(
            source=str(img_path),
            device=device,
            verbose=False,
            conf=0.01,
            imgsz=args.imgsz,
        )
        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            max_conf = float(boxes.conf.cpu().numpy().max())
        else:
            max_conf = None
        raw.append((is_wildfire, max_conf))

    # Sweep conf thresholds
    conf_values = [round(v, 2) for v in np.arange(0.05, 0.51, 0.05)]
    results = []
    for conf_thresh in conf_values:
        tp = fn = fp = tn = 0
        for is_wildfire, max_conf in raw:
            detected = max_conf is not None and max_conf >= conf_thresh
            if is_wildfire:
                if detected: tp += 1
                else: fn += 1
            else:
                if detected: fp += 1
                else: tn += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else None
        recall = tp / (tp + fn) if (tp + fn) > 0 else None
        f1 = (2 * precision * recall / (precision + recall)
              if precision and recall and (precision + recall) > 0 else None)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else None

        results.append({
            "conf_thresh": conf_thresh,
            "tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "precision": round(precision, 4) if precision is not None else None,
            "recall": round(recall, 4) if recall is not None else None,
            "f1": round(f1, 4) if f1 is not None else None,
            "fpr": round(fpr, 4) if fpr is not None else None,
        })
        print(
            f"conf={conf_thresh:.2f} | TP={tp} FN={fn} FP={fp} TN={tn} | "
            + (f"recall={recall:.1%} fpr={fpr:.1%} f1={f1:.3f}"
               if (recall and fpr is not None and f1) else "n/a")
        )

    df = pd.DataFrame(results).sort_values("f1", ascending=False)
    print("\n── Top 5 by F1 ──────────────────────────────────────────────────")
    print(df.head(5).to_string(index=False))

    df.to_csv(args.output, sep="\t", index=False)
    print(f"\nFull results → {args.output}")
