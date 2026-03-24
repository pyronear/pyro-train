"""
Run the engine with given params, find failing sequences (FN + FP),
and copy their image folders to an output directory.

FN: wildfire sequences the engine missed
FP: fp sequences the engine falsely alerted on
"""

import argparse
import json
import shutil
import sys
from collections import deque
from pathlib import Path
from types import ModuleType

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
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from pyroengine.engine import Engine  # noqa: E402

logging.getLogger().setLevel(logging.WARNING)
logging.getLogger("pyroengine").setLevel(logging.WARNING)

_DUMMY_FRAME = Image.new("RGB", (1, 1))


def _parse_label_file(path: Path) -> np.ndarray:
    dets = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        _, cx, cy, w, h, conf = (float(v) for v in line.split())
        dets.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, conf])
    return np.array(dets, dtype=np.float32) if dets else np.empty((0, 5), dtype=np.float32)


def load_labels_dir(
    labels_dir: Path, min_frames: int = 0
) -> dict[tuple[str, str], list[np.ndarray]]:
    grouped: dict[tuple[str, str], list[np.ndarray]] = {}
    for category in ("wildfire", "fp"):
        cat_dir = labels_dir / category
        if not cat_dir.exists():
            continue
        for seq_dir in sorted(cat_dir.iterdir()):
            if not seq_dir.is_dir():
                continue
            label_files = sorted((seq_dir / "labels").glob("*.txt"))
            if len(label_files) < min_frames:
                continue
            grouped[(category, seq_dir.name)] = [_parse_label_file(lf) for lf in label_files]
    return grouped


class _PassthroughModel:
    def post_process(self, fake_pred: np.ndarray, **_: object) -> np.ndarray:
        return fake_pred


def _make_engine(nb_consecutive_frames: int, conf_thresh: float) -> Engine:
    cache = Path("/tmp/pyro_failures_cache")
    cache.mkdir(parents=True, exist_ok=True)
    engine = Engine(
        conf_thresh=conf_thresh,
        nb_consecutive_frames=nb_consecutive_frames,
        cache_folder=str(cache),
    )
    engine.model = _PassthroughModel()  # type: ignore[assignment]
    return engine


def _reset_state(engine: Engine, cam_key: str) -> None:
    engine._states[cam_key] = {  # noqa: SLF001
        "last_predictions": deque(maxlen=engine.nb_consecutive_frames),
        "ongoing": False,
        "last_image_sent": None,
        "last_bbox_mask_fetch": None,
        "anchor_bbox": None,
        "anchor_ts": None,
        "miss_count": 0,
    }
    engine.occlusion_masks[cam_key] = (None, {}, 0)  # noqa: SLF001


def _evaluate_sequence(engine: Engine, frames: list[np.ndarray], cam_key: str) -> bool:
    _reset_state(engine, cam_key)
    for fake_pred in frames:
        if engine.predict(_DUMMY_FRAME, cam_id=cam_key, fake_pred=fake_pred) > engine.conf_thresh:
            return True
    return False


def _parse_gt_label_file(path: Path) -> list[tuple[float, float, float, float]]:
    """Parse YOLO GT label: class cx cy w h (no conf)."""
    boxes = []
    for line in path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        _, cx, cy, w, h = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        boxes.append((cx, cy, w, h))
    return boxes


def _draw_box(draw: "ImageDraw.ImageDraw", cx: float, cy: float, w: float, h: float,  # type: ignore[name-defined]
              img_w: int, img_h: int, color: str, label: str = "",
              font: object = None) -> None:
    x1 = int((cx - w / 2) * img_w)
    y1 = int((cy - h / 2) * img_h)
    x2 = int((cx + w / 2) * img_w)
    y2 = int((cy + h / 2) * img_h)
    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
    if label:
        draw.text((x1 + 2, y1 + 2), label, fill=color, font=font)  # type: ignore[arg-type]


def annotate_sequence(
    seq_dest: Path,
    category: str,
    sequence: str,
    data_dir: Path,
    labels_dir: Path,
) -> None:
    """Draw GT (green) and predicted (red) boxes on each frame and save as *_annotated.jpg."""
    images_dest = seq_dest / "images"
    if not images_dest.exists():
        return

    pred_labels_dir = labels_dir / category / sequence / "labels"
    gt_labels_dir = data_dir / category / sequence / "labels"

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)  # type: ignore[assignment]
    except OSError:
        font = ImageFont.load_default()  # type: ignore[assignment]

    annotated_dir = seq_dest / "annotated"
    annotated_dir.mkdir(parents=True, exist_ok=True)

    for img_path in sorted(images_dest.glob("*.jpg")) + sorted(images_dest.glob("*.png")):
        img = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        iw, ih = img.size

        # Ground truth (green) — only for wildfire sequences
        gt_file = gt_labels_dir / img_path.with_suffix(".txt").name
        if gt_file.exists():
            for cx, cy, w, h in _parse_gt_label_file(gt_file):
                _draw_box(draw, cx, cy, w, h, iw, ih, "lime", "GT", font)

        # Predictions (red)
        pred_file = pred_labels_dir / img_path.with_suffix(".txt").name
        if pred_file.exists():
            for det in _parse_label_file(pred_file):
                # det is [x1,y1,x2,y2,conf] — convert back to cx,cy,w,h
                x1, y1, x2, y2, conf = det
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                w = x2 - x1
                h = y2 - y1
                _draw_box(draw, cx, cy, w, h, iw, ih, "red", f"{conf:.2f}", font)

        img.save(annotated_dir / img_path.name)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels-dir", type=Path, default=Path("predictions_labels"))
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Root dir with wildfire/ and fp/ subfolders (source images to copy)")
    parser.add_argument("--output-dir", type=Path, default=Path("failures"))
    parser.add_argument("--nb-consecutive-frames", type=int, default=5)
    parser.add_argument("--conf-thresh", type=float, default=0.2)
    parser.add_argument("--min-frames", type=int, default=8)
    parser.add_argument("--output-json", type=Path, default=None,
                        help="If set, save evaluation metrics as JSON to this path")
    return parser


if __name__ == "__main__":
    args = make_parser().parse_args()

    grouped = load_labels_dir(args.labels_dir, min_frames=args.min_frames)

    engine = _make_engine(args.nb_consecutive_frames, args.conf_thresh)
    print(f"nb_consecutive_frames={args.nb_consecutive_frames}  conf_thresh={args.conf_thresh}")

    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)

    fn_dir = args.output_dir / "fn_wildfire"
    fp_dir = args.output_dir / "fp_alerted"
    fn_dir.mkdir(parents=True, exist_ok=True)
    fp_dir.mkdir(parents=True, exist_ok=True)

    fn_seqs, fp_seqs = [], []
    tp = fn = fp = tn = 0
    for (category, sequence), frames in grouped.items():
        alerted = _evaluate_sequence(engine, frames, sequence)
        if category == "wildfire":
            if alerted: tp += 1
            else:
                fn += 1
                fn_seqs.append(sequence)
        else:
            if alerted:
                fp += 1
                fp_seqs.append(sequence)
            else:
                tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    fpr = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else float("nan")

    print(f"\n{'':─<55}")
    print(f"  TP={tp}  FN={fn}  FP={fp}  TN={tn}")
    print(f"  Recall={recall:.1%}  FPR={fpr:.1%}  Precision={precision:.1%}  F1={f1:.3f}")
    print(f"{'':─<55}")

    def copy_seq(category: str, sequence: str, dest_dir: Path) -> None:
        src = args.data_dir / category / sequence
        if src.exists():
            shutil.copytree(src, dest_dir / sequence, dirs_exist_ok=True)
            annotate_sequence(dest_dir / sequence, category, sequence, args.data_dir, args.labels_dir)
        else:
            print(f"  WARNING: source not found: {src}")

    print(f"\nFN (missed wildfire): {len(fn_seqs)}")
    for seq in fn_seqs:
        print(f"  {seq}")
        copy_seq("wildfire", seq, fn_dir)

    print(f"\nFP (false alert): {len(fp_seqs)}")
    for seq in fp_seqs:
        print(f"  {seq}")
        copy_seq("fp", seq, fp_dir)

    print(f"\nCopied to {args.output_dir}/")

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        def _safe(v: float) -> float | None:
            import math
            return round(v, 4) if not math.isnan(v) else None

        metrics = {
            "nb_consecutive_frames": args.nb_consecutive_frames,
            "conf_thresh": args.conf_thresh,
            "tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "recall": _safe(recall),
            "fpr": _safe(fpr),
            "precision": _safe(precision),
            "f1": _safe(f1),
        }
        args.output_json.write_text(json.dumps(metrics, indent=2))
        print(f"Metrics → {args.output_json}")
