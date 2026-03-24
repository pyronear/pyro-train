"""
Visualize YOLO val failures (FP + FN) in FiftyOne.

Reads output from eval_yolo_val.py:
  <failures-dir>/fp/<image>.jpg  + .txt  (predicted boxes)
  <failures-dir>/fn/<image>.jpg  + .txt  (empty — model missed)

GT labels are loaded from the original yolo_train_val/labels/val/.

In the FiftyOne UI:
  predictions  — what the model detected  (shown in red by default)
  ground_truth — what the image actually contains (shown in green by default)

Usage:
    uv run python view_yolo_val_failures.py
    uv run python view_yolo_val_failures.py --split fp
    uv run python view_yolo_val_failures.py --failures-dir failures_val --port 5151
"""

import argparse
from pathlib import Path

import fiftyone as fo


def _load_pred_labels(txt_path: Path) -> fo.Detections:
    """Parse prediction label: class cx cy w h conf → fo.Detections."""
    dets = []
    if txt_path.exists():
        for line in txt_path.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            _, cx, cy, w, h, conf = (float(p) for p in parts)
            dets.append(fo.Detection(
                label="fire",
                bounding_box=[cx - w / 2, cy - h / 2, w, h],
                confidence=conf,
            ))
    return fo.Detections(detections=dets)


def _load_gt_labels(txt_path: Path) -> fo.Detections:
    """Parse GT label: class cx cy w h → fo.Detections."""
    dets = []
    if txt_path.exists():
        for line in txt_path.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            _, cx, cy, w, h = (float(p) for p in parts[:5])
            dets.append(fo.Detection(
                label="fire",
                bounding_box=[cx - w / 2, cy - h / 2, w, h],
            ))
    return fo.Detections(detections=dets)


def build_dataset(split_dir: Path, gt_labels_dir: Path, dataset_name: str) -> fo.Dataset:
    if fo.dataset_exists(dataset_name):
        fo.delete_dataset(dataset_name)
    dataset = fo.Dataset(dataset_name)

    images = sorted(split_dir.glob("*.jpg")) + sorted(split_dir.glob("*.png"))
    samples = []
    for img_path in images:
        sample = fo.Sample(filepath=str(img_path))
        sample["predictions"] = _load_pred_labels(split_dir / img_path.with_suffix(".txt").name)
        sample["ground_truth"] = _load_gt_labels(gt_labels_dir / img_path.with_suffix(".txt").name)
        samples.append(sample)

    dataset.add_samples(samples)
    dataset.persistent = True
    return dataset


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize YOLO val failures in FiftyOne")
    parser.add_argument("--failures-dir", type=Path, default=Path("failures_val"),
                        help="Output dir from eval_yolo_val.py (default: failures_val)")
    parser.add_argument("--gt-labels-dir", type=Path,
                        default=Path("data/01_model_input/yolo_train_val/labels/val"),
                        help="GT labels dir (default: yolo_train_val/labels/val)")
    parser.add_argument("--split", choices=["fp", "fn", "both"], default="both")
    parser.add_argument("--port-fp", type=int, default=5151)
    parser.add_argument("--port-fn", type=int, default=5152)
    return parser


if __name__ == "__main__":
    args = make_parser().parse_args()

    sessions = []

    if args.split in ("fp", "both"):
        fp_dir = args.failures_dir / "fp"
        if fp_dir.exists():
            ds = build_dataset(fp_dir, args.gt_labels_dir, "val_failures_fp")
            print(f"FP (false alerts): {len(ds)} images → http://localhost:{args.port_fp}")
            sessions.append(fo.launch_app(ds, port=args.port_fp, auto=False))
        else:
            print(f"WARNING: {fp_dir} not found — run eval_yolo_val.py first")

    if args.split in ("fn", "both"):
        fn_dir = args.failures_dir / "fn"
        if fn_dir.exists():
            ds = build_dataset(fn_dir, args.gt_labels_dir, "val_failures_fn")
            print(f"FN (missed wildfire): {len(ds)} images → http://localhost:{args.port_fn}")
            sessions.append(fo.launch_app(ds, port=args.port_fn, auto=False))
        else:
            print(f"WARNING: {fn_dir} not found — run eval_yolo_val.py first")

    if not sessions:
        raise SystemExit("No sessions started — check --failures-dir")

    print("\nPress Ctrl+C to stop.")
    try:
        for s in sessions:
            s.wait()
    except KeyboardInterrupt:
        pass
