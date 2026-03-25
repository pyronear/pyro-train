"""
Visualize failure sequences (FN wildfire + FP) in FiftyOne.

Uses original images and overlays predictions + ground truth as native
FiftyOne detections (no pre-drawn images).

  predictions  — from --labels-dir  (red in UI)
  ground_truth — from --data-dir labels/ subfolders (green in UI)

A black separator image is inserted between each sequence.

Usage:
    uv run python view_failures.py --data-dir data/01_model_input/sequential_train_val/val
    uv run python view_failures.py --data-dir ... --split fn_wildfire
"""

import argparse
import tempfile
from pathlib import Path

import fiftyone as fo
from PIL import Image


_SEPARATOR_SIZE = (1280, 720)


def _make_separator(tmp_dir: Path, name: str) -> Path:
    img = Image.new("RGB", _SEPARATOR_SIZE, color=(0, 0, 0))
    path = tmp_dir / f"_sep_{name}.jpg"
    img.save(path)
    return path


def _load_predictions(label_file: Path) -> fo.Detections:
    """Parse predictions label file: class cx cy w h conf → fo.Detections."""
    dets = []
    if label_file.exists():
        for line in label_file.read_text().splitlines():
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


def _load_ground_truth(label_file: Path) -> fo.Detections:
    """Parse GT label file: class cx cy w h → fo.Detections."""
    dets = []
    if label_file.exists():
        for line in label_file.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            _, cx, cy, w, h = (float(p) for p in parts[:5])
            dets.append(fo.Detection(
                label="fire",
                bounding_box=[cx - w / 2, cy - h / 2, w, h],
            ))
    return fo.Detections(detections=dets)


def build_dataset(
    split_dir: Path,
    category: str,
    dataset_name: str,
    tmp_dir: Path,
    labels_dir: Path,
    data_dir: Path,
) -> fo.Dataset:
    sequences = sorted(d for d in split_dir.iterdir() if d.is_dir())

    if fo.dataset_exists(dataset_name):
        fo.delete_dataset(dataset_name)
    dataset = fo.Dataset(dataset_name)

    samples = []
    for seq_dir in sequences:
        images_dir = seq_dir / "images"
        frames = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))
        if not frames:
            continue

        # black separator before each sequence
        sep_path = _make_separator(tmp_dir, seq_dir.name)
        sep_sample = fo.Sample(filepath=str(sep_path))
        sep_sample["sequence"] = f"--- {seq_dir.name} ---"
        sep_sample["is_separator"] = True
        samples.append(sep_sample)

        pred_labels_dir = labels_dir / category / seq_dir.name / "labels"
        gt_labels_dir = data_dir / category / seq_dir.name / "labels"

        for frame_path in frames[:args.max_frames]:
            sample = fo.Sample(filepath=str(frame_path))
            sample["sequence"] = seq_dir.name
            sample["is_separator"] = False

            stem = frame_path.stem
            sample["predictions"] = _load_predictions(pred_labels_dir / f"{stem}.txt")
            sample["ground_truth"] = _load_ground_truth(gt_labels_dir / f"{stem}.txt")

            samples.append(sample)

    dataset.add_samples(samples)
    dataset.persistent = True  # keep tags after script exits
    return dataset


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--failures-dir", type=Path, default=Path("failures"))
    parser.add_argument("--labels-dir", type=Path, default=Path("predictions_labels"),
                        help="Root dir with prediction label files")
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Root dir with wildfire/ and fp/ subfolders (GT labels + images)")
    parser.add_argument(
        "--split",
        choices=["fn_wildfire", "fp_alerted", "both"],
        default="both",
    )
    parser.add_argument("--port-fn", type=int, default=5151)
    parser.add_argument("--port-fp", type=int, default=5152)
    parser.add_argument("--max-frames", type=int, default=15)
    return parser


if __name__ == "__main__":
    args = make_parser().parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        sessions = []

        if args.split in ("fn_wildfire", "both"):
            fn_dir = args.failures_dir / "fn_wildfire"
            if fn_dir.exists():
                ds = build_dataset(fn_dir, "wildfire", "failures_fn_wildfire", tmp_dir,
                                   args.labels_dir, args.data_dir)
                print(f"FN wildfire: {len(ds)} frames")
                session = fo.launch_app(ds, port=args.port_fn, auto=False)
                sessions.append(session)
                print(f"  → http://localhost:{args.port_fn}")
            else:
                print(f"WARNING: {fn_dir} not found")

        if args.split in ("fp_alerted", "both"):
            fp_dir = args.failures_dir / "fp_alerted"
            if fp_dir.exists():
                ds = build_dataset(fp_dir, "fp", "failures_fp_alerted", tmp_dir,
                                   args.labels_dir, args.data_dir)
                print(f"FP alerted:  {len(ds)} frames")
                session = fo.launch_app(ds, port=args.port_fp, auto=False)
                sessions.append(session)
                print(f"  → http://localhost:{args.port_fp}")
            else:
                print(f"WARNING: {fp_dir} not found")

        if not sessions:
            print("No sessions started — check --failures-dir")
        else:
            print("\nPress Ctrl+C to stop.")
            try:
                for s in sessions:
                    s.wait()
            except KeyboardInterrupt:
                pass
