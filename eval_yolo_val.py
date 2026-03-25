"""
Run YOLO model on yolo_train_val/val images and copy failures.

Failures:
  - FP: background image (empty GT label) where model fired
  - FN: wildfire image (non-empty GT label) where model missed

Output layout:
  <output-dir>/fp/<image>      # false positives
  <output-dir>/fn/<image>      # false negatives (missed wildfire)
"""

import argparse
import shutil
from pathlib import Path

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
    """Return True if label file has at least one box."""
    return label_path.exists() and label_path.stat().st_size > 0


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate YOLO on val split, copy failures")
    parser.add_argument("--model-path", type=Path, default=Path("data/02_models/yolo/best/weights/best.pt"),
                        help="Path to YOLO .pt weights (default: mongoose.pt)")
    parser.add_argument("--data-dir", type=Path,
                        default=Path("data/01_model_input/yolo_train_val"),
                        help="Root dir with images/val and labels/val")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"],
                        help="Dataset split to evaluate (default: val)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory for failure images (default: failures_yolo_<split>)")
    parser.add_argument("--conf", type=float, default=0.2,
                        help="Confidence threshold (default: 0.2)")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--device", type=str, default=None)
    return parser


if __name__ == "__main__":
    args = make_parser().parse_args()

    output_dir = args.output_dir or Path(f"failures_yolo_{args.split}")
    images_dir = args.data_dir / "images" / args.split
    labels_dir = args.data_dir / "labels" / args.split

    if not images_dir.exists():
        raise SystemExit(f"Images dir not found: {images_dir}")
    if not labels_dir.exists():
        raise SystemExit(f"Labels dir not found: {labels_dir}")
    if not args.model_path.exists():
        raise SystemExit(f"Model not found: {args.model_path}")

    device = args.device or pick_device()
    print(f"Model: {args.model_path}  Split: {args.split}  Device: {device}  Conf: {args.conf}")

    if output_dir.exists():
        shutil.rmtree(output_dir)
    fp_dir = output_dir / "fp"
    fn_dir = output_dir / "fn"
    fp_dir.mkdir(parents=True)
    fn_dir.mkdir(parents=True)

    model = YOLO(str(args.model_path))

    images = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))
    fp_count = fn_count = 0

    for img_path in tqdm(images, desc="evaluating"):
        label_path = labels_dir / img_path.with_suffix(".txt").name
        is_wildfire = has_gt(label_path)

        results = model.predict(
            source=str(img_path),
            device=device,
            verbose=False,
            conf=args.conf,
            imgsz=args.imgsz,
        )
        boxes = results[0].boxes
        detected = boxes is not None and len(boxes) > 0

        if not is_wildfire and detected:
            dest_dir = fp_dir
            fp_count += 1
        elif is_wildfire and not detected:
            dest_dir = fn_dir
            fn_count += 1
        else:
            continue

        shutil.copy(img_path, dest_dir / img_path.name)
        # Save prediction labels alongside the image
        pred_txt = dest_dir / img_path.with_suffix(".txt").name
        if detected:
            xywhn = boxes.xywhn.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            lines = [f"0 {xywhn[i][0]:.6f} {xywhn[i][1]:.6f} {xywhn[i][2]:.6f} {xywhn[i][3]:.6f} {confs[i]:.6f}\n"
                     for i in range(len(xywhn))]
            pred_txt.write_text("".join(lines))
        else:
            pred_txt.write_text("")

    total = len(images)
    print(f"\nResults on {total} images:")
    print(f"  FP (false alerts on background): {fp_count}")
    print(f"  FN (missed wildfire):            {fn_count}")
    print(f"Failures copied to {output_dir}/")
