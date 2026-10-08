"""
Run the engine with given params, find failing sequences (FN + FP),
and copy their image folders to an output directory.

FN: wildfire sequences the engine missed
FP: fp sequences the engine falsely alerted on
"""

import argparse
import json
import math
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from pyro_train.model.sequential import Replay, load_labels_dir, parse_label_file


def wilson_ci95(successes: int, n: int) -> list[float] | None:
    """Wilson score 95% interval for a proportion; None when n == 0."""
    if n == 0:
        return None
    z = 1.96
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [round(max(center - half, 0.0), 4), round(min(center + half, 1.0), 4)]


def detected_within_frames(
    triggers: list[int | None], ks: tuple[int, ...] = (2, 3, 5)
) -> dict[str, float | None]:
    """Share of all wildfire sequences alerted within their first k frames.

    Missed wildfire counts as not detected, so a model is not rewarded for
    missing the hard ones. Same definition as temporal-model's eval.
    """
    return {
        str(k): round(sum(t is not None and t < k for t in triggers) / len(triggers), 4)
        if triggers
        else None
        for k in ks
    }


def _parse_gt_label_file(path: Path) -> list[tuple[float, float, float, float]]:
    """Parse YOLO GT label: class cx cy w h (no conf)."""
    boxes = []
    for line in path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        _, cx, cy, w, h = (
            float(parts[0]),
            float(parts[1]),
            float(parts[2]),
            float(parts[3]),
            float(parts[4]),
        )
        boxes.append((cx, cy, w, h))
    return boxes


def _draw_box(
    draw: "ImageDraw.ImageDraw",
    cx: float,
    cy: float,
    w: float,
    h: float,  # type: ignore[name-defined]
    img_w: int,
    img_h: int,
    color: str,
    label: str = "",
    font: object = None,
) -> None:
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

    for img_path in sorted(images_dest.glob("*.jpg")) + sorted(
        images_dest.glob("*.png")
    ):
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
            for det in parse_label_file(pred_file):
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
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Root dir with wildfire/ and fp/ subfolders (source images to copy)",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("failures"))
    parser.add_argument("--nb-consecutive-frames", type=int, default=5)
    parser.add_argument("--conf-thresh", type=float, default=0.2)
    parser.add_argument("--min-frames", type=int, default=0)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="If set, save evaluation metrics as JSON to this path",
    )
    parser.add_argument(
        "--output-predictions",
        type=Path,
        default=None,
        help="If set, write each sequence's decision here, to pair two models",
    )
    return parser


if __name__ == "__main__":
    args = make_parser().parse_args()

    grouped = load_labels_dir(args.labels_dir, min_frames=args.min_frames)

    replay = Replay(args.nb_consecutive_frames, args.conf_thresh)
    print(
        f"nb_consecutive_frames={args.nb_consecutive_frames}  conf_thresh={args.conf_thresh}"
    )

    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)

    fn_dir = args.output_dir / "fn_wildfire"
    fp_dir = args.output_dir / "fp_alerted"
    fn_dir.mkdir(parents=True, exist_ok=True)
    fp_dir.mkdir(parents=True, exist_ok=True)

    fn_seqs, fp_seqs = [], []
    tp = fn = fp = tn = 0
    predictions, wildfire_triggers = [], []
    for (category, sequence), frames in grouped.items():
        trigger = replay.trigger(frames)
        alerted = trigger is not None
        predictions.append(
            {
                "category": category,
                "sequence": sequence,
                "alerted": alerted,
                "trigger_frame": trigger,
            }
        )
        if category == "wildfire":
            wildfire_triggers.append(trigger)
            if alerted:
                tp += 1
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
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else float("nan")
    )

    print(f"\n{'':─<55}")
    print(f"  TP={tp}  FN={fn}  FP={fp}  TN={tn}")
    print(
        f"  Recall={recall:.1%}  FPR={fpr:.1%}  Precision={precision:.1%}  F1={f1:.3f}"
    )
    print(f"{'':─<55}")

    def copy_seq(category: str, sequence: str, dest_dir: Path) -> None:
        src = args.data_dir / category / sequence
        if src.exists():
            shutil.copytree(src, dest_dir / sequence, dirs_exist_ok=True)
            annotate_sequence(
                dest_dir / sequence, category, sequence, args.data_dir, args.labels_dir
            )
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
            return round(v, 4) if not math.isnan(v) else None

        # No precision or F1: the test smoke/FP ratio is arbitrary, so both
        # would move with it.
        metrics = {
            "nb_consecutive_frames": args.nb_consecutive_frames,
            "conf_thresh": args.conf_thresh,
            "num_sequences": len(grouped),
            "tp": tp,
            "fn": fn,
            "fp": fp,
            "tn": tn,
            "recall": _safe(recall),
            "fpr": _safe(fpr),
            "recall_ci95": wilson_ci95(tp, tp + fn),
            "fpr_ci95": wilson_ci95(fp, fp + tn),
            "detected_within_frames": detected_within_frames(wildfire_triggers),
        }
        args.output_json.write_text(json.dumps(metrics, indent=2))
        print(f"Metrics → {args.output_json}")

    if args.output_predictions is not None:
        args.output_predictions.parent.mkdir(parents=True, exist_ok=True)
        args.output_predictions.write_text(json.dumps(predictions, indent=2))
        print(f"Predictions → {args.output_predictions}")
