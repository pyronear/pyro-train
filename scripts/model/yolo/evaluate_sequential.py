"""
CLI script to evaluate a trained YOLO model (exported to ONNX) using the
pyro-engine sequential inference pipeline on a held-out sequential dataset.

Dataset layout expected under --data-dir:
  <data-dir>/wildfire/<sequence>/images/*.jpg   → positive class (fire)
  <data-dir>/fp/<sequence>/images/*.jpg         → negative class (no fire)

A sequence is considered *alerted* if the engine (with temporal aggregation over
nb_consecutive_frames) raises confidence above conf_thresh on any frame.

Metrics reported per sequence:
  wildfire → TP (alerted) / FN (not alerted)
  fp       → FP (alerted) / TN (not alerted)
"""

import argparse
import json
import logging
import ssl
import sys
from collections import deque
from pathlib import Path
from types import ModuleType

# macOS Python.org installs lack system CA certs.
ssl._create_default_https_context = ssl._create_unverified_context  # noqa: SLF001

# Stub API-client packages – not needed for offline inference.
# Must expose concrete names so `from <pkg> import <Class>` succeeds.
_Stub = type("_Stub", (), {"__init__": lambda s, *a, **k: None})

_cam_client = ModuleType("pyro_camera_api_client.client")
_cam_client.PyroCameraAPIClient = _Stub  # type: ignore[attr-defined]
_cam = ModuleType("pyro_camera_api_client")
_cam.client = _cam_client  # type: ignore[attr-defined]

_pyro_client_mod = ModuleType("pyroclient.client")
_pyro_client_mod.PyroClient = _Stub  # type: ignore[attr-defined]
_pyro = ModuleType("pyroclient")
_pyro.client = _pyro_client_mod  # type: ignore[attr-defined]

for _name, _mod in (
    ("pyro_camera_api_client", _cam),
    ("pyro_camera_api_client.client", _cam_client),
    ("pyroclient", _pyro),
    ("pyroclient.client", _pyro_client_mod),
):
    sys.modules.setdefault(_name, _mod)

import onnxruntime  # noqa: E402
from PIL import Image  # noqa: E402
from pyroengine.engine import Engine  # noqa: E402
from tqdm import tqdm  # noqa: E402

# pyroengine calls logging.basicConfig(force=True, level=INFO) at import time,
# resetting the root logger. Pin root to WARNING permanently so pyroengine's
# per-frame INFO spam is always suppressed, regardless of --loglevel.
logging.getLogger().setLevel(logging.WARNING)

# Named logger for this script – level is set from --loglevel in __main__.
logger = logging.getLogger("pyro_eval")
logger.addHandler(logging.StreamHandler())
logger.propagate = False


def reset_state(engine: Engine, cam_key: str) -> None:
    """Reset per-camera state so each sequence starts fresh."""
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


def evaluate_sequence(engine: Engine, images_dir: Path, cam_key: str, max_frames: int) -> bool:
    """Return True if engine raised an alert on any frame in the sequence."""
    reset_state(engine, cam_key)
    frames = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))
    for frame_path in frames[:max_frames]:
        conf = engine.predict(Image.open(frame_path), cam_id=cam_key)
        if conf > engine.conf_thresh:
            return True
    return False


def evaluate_category(engine: Engine, category_dir: Path, label: str, max_frames: int) -> list[dict]:
    sequences = sorted(d for d in category_dir.iterdir() if d.is_dir())
    records = []
    with tqdm(sequences, desc=label, unit="seq") as pbar:
        for seq_dir in pbar:
            images_dir = seq_dir / "images"
            if not images_dir.exists():
                logger.warning(f"No images/ folder in {seq_dir}, skipping")
                continue
            frames = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
            alerted = evaluate_sequence(engine, images_dir, seq_dir.name, max_frames)
            records.append({"sequence": seq_dir.name, "alerted": alerted, "n_frames": min(len(frames), max_frames)})
            pbar.set_postfix({"last": seq_dir.name[:30], "alert": alerted})
    return records


def make_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        help="path to the exported ONNX model (e.g. data/02_models/yolo-export/best/onnx/cpu/best.onnx)",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--data-dir",
        help="root directory containing wildfire/ and fp/ subfolders of sequences",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        help="directory to save evaluation results (eval_sequential_results.json)",
        default="./data/03_reporting/yolo/sequential/best/",
        type=Path,
    )
    parser.add_argument(
        "--conf-thresh",
        help="confidence threshold for alert (default: 0.15)",
        default=0.15,
        type=float,
    )
    parser.add_argument(
        "--nb-consecutive-frames",
        help="number of consecutive frames for temporal aggregation (default: 8)",
        default=8,
        type=int,
    )
    parser.add_argument(
        "--max-frames",
        help="maximum frames to evaluate per sequence (default: 15)",
        default=15,
        type=int,
    )
    parser.add_argument(
        "-log",
        "--loglevel",
        default="warning",
        help="logging level (debug, info, warning, error)",
    )
    return parser


def validate_parsed_args(args: dict) -> bool:
    if not args["model_path"].exists():
        logger.error(f"--model-path not found: {args['model_path']}")
        return False
    if not args["data_dir"].exists():
        logger.error(f"--data-dir not found: {args['data_dir']}")
        return False
    for split in ("wildfire", "fp"):
        if not (args["data_dir"] / split).exists():
            logger.error(f"Expected subfolder '{split}' not found in {args['data_dir']}")
            return False
    return True


if __name__ == "__main__":
    cli_parser = make_cli_parser()
    args = vars(cli_parser.parse_args())
    logger.setLevel(args["loglevel"].upper())

    if not validate_parsed_args(args):
        exit(1)

    args["output_dir"].mkdir(parents=True, exist_ok=True)

    engine = Engine(
        model_path=str(args["model_path"]),
        conf_thresh=args["conf_thresh"],
        nb_consecutive_frames=args["nb_consecutive_frames"],
        cache_folder=str(args["output_dir"]),
    )

    # Upgrade to CUDA execution provider if available (requires onnxruntime-gpu).
    available_providers = onnxruntime.get_available_providers()
    if "CUDAExecutionProvider" in available_providers:
        engine.model.ort_session = onnxruntime.InferenceSession(  # noqa: SLF001
            str(args["model_path"]),
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        logger.info("onnxruntime: using CUDAExecutionProvider")
    else:
        logger.info(f"onnxruntime: CUDA not available, using {available_providers[0]}")

    wf_records = evaluate_category(engine, args["data_dir"] / "wildfire", "wildfire", args["max_frames"])
    fp_records = evaluate_category(engine, args["data_dir"] / "fp", "fp     ", args["max_frames"])

    tp = sum(r["alerted"] for r in wf_records)
    fn = sum(not r["alerted"] for r in wf_records)
    fp = sum(r["alerted"] for r in fp_records)
    tn = sum(not r["alerted"] for r in fp_records)

    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    recall = tp / (tp + fn) if (tp + fn) > 0 else None
    f1 = 2 * precision * recall / (precision + recall) if (precision and recall and precision + recall > 0) else None
    fpr = fp / (fp + tn) if (fp + tn) > 0 else None

    results = {
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "conf_thresh": args["conf_thresh"],
        "nb_consecutive_frames": args["nb_consecutive_frames"],
        "max_frames": args["max_frames"],
        "model_path": str(args["model_path"]),
        "data_dir": str(args["data_dir"]),
        "wildfire_sequences": wf_records,
        "fp_sequences": fp_records,
    }

    output_path = args["output_dir"] / "eval_sequential_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    def _fmt(v: float | None, spec: str) -> str:
        return format(v, spec) if v is not None else "N/A"

    print(
        f"Sequential eval  |  "
        f"TP={tp}  FN={fn}  FP={fp}  TN={tn}  |  "
        f"Recall={_fmt(recall, '.1%')}  FPR={_fmt(fpr, '.1%')}  F1={_fmt(f1, '.3f')}"
    )
    exit(0)
