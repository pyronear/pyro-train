"""
Grid-search nb_consecutive_frames × conf_thresh using pre-computed label files
from predict_sequential.py.

Grid:
  nb_consecutive_frames : 4 → 8  (step 1)
  conf_thresh           : 0.05 → 0.40 (step 0.05)

Usage:
  uv run python optimize_sequential.py --labels-dir predictions_labels/
"""

import argparse
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
import ssl  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402
import pyroengine.engine  # noqa: E402
from pyroengine.engine import Engine  # noqa: E402

# Engine.__init__ builds a Classifier, which downloads a detector from
# HuggingFace. Nothing here runs it: _make_engine swaps engine.model for
# _PassthroughModel one line later, because this script replays predictions
# computed upstream. The download was always dead weight, and became fatal once
# the model repository stopped answering anonymously (HTTP 401).
pyroengine.engine.Classifier = _Stub

ssl._create_default_https_context = ssl._create_unverified_context  # noqa: SLF001
logging.getLogger().setLevel(logging.WARNING)
logging.getLogger("pyroengine").setLevel(logging.WARNING)

_DUMMY_FRAME = Image.new("RGB", (1, 1))


# ── Label file helpers ──────────────────────────────────────────────────────

def _parse_label_file(path: Path) -> np.ndarray:
    """Parse a YOLO label file → (N,5) array of [x1,y1,x2,y2,conf]."""
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
    """Load all sequences from a labels directory.

    Returns {(category, sequence): [frame_array, ...]} sorted by frame name.
    Each frame_array is (N,5) with [x1,y1,x2,y2,conf].
    """
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


# ── Engine helpers ──────────────────────────────────────────────────────────

class _PassthroughModel:
    def post_process(self, fake_pred: np.ndarray, **_: object) -> np.ndarray:
        return fake_pred


def _make_engine(nb_consecutive_frames: int, conf_thresh: float) -> Engine:
    cache = Path("/tmp/pyro_opt_cache")
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


# ── Grid search ─────────────────────────────────────────────────────────────

def run_grid(
    grouped: dict[tuple[str, str], list[np.ndarray]],
    nb_frames_values: list[int],
    conf_thresh_values: list[float],
) -> list[dict]:
    results = []
    total = len(nb_frames_values) * len(conf_thresh_values)
    done = 0
    for nb_frames in nb_frames_values:
        for conf_thresh in conf_thresh_values:
            engine = _make_engine(nb_frames, float(conf_thresh))
            tp = fn = fp = tn = 0
            for (category, sequence), frames in grouped.items():
                alerted = _evaluate_sequence(engine, frames, sequence)
                if category == "wildfire":
                    if alerted: tp += 1
                    else: fn += 1
                else:
                    if alerted: fp += 1
                    else: tn += 1

            precision = tp / (tp + fp) if (tp + fp) > 0 else None
            recall = tp / (tp + fn) if (tp + fn) > 0 else None
            f1 = (
                2 * precision * recall / (precision + recall)
                if precision and recall and (precision + recall) > 0
                else None
            )
            fpr = fp / (fp + tn) if (fp + tn) > 0 else None

            results.append({
                "nb_consecutive_frames": nb_frames,
                "conf_thresh": round(conf_thresh, 4),
                "tp": tp, "fn": fn, "fp": fp, "tn": tn,
                "precision": round(precision, 4) if precision is not None else None,
                "recall": round(recall, 4) if recall is not None else None,
                "f1": round(f1, 4) if f1 is not None else None,
                "fpr": round(fpr, 4) if fpr is not None else None,
            })
            done += 1
            print(
                f"[{done:>3}/{total}] nb_frames={nb_frames} conf={conf_thresh:.2f} | "
                f"TP={tp} FN={fn} FP={fp} TN={tn} | "
                f"recall={recall:.1%} fpr={fpr:.1%} f1={f1:.3f}"
                if (recall is not None and fpr is not None and f1 is not None)
                else f"[{done:>3}/{total}] nb_frames={nb_frames} conf={conf_thresh:.2f} | "
                f"TP={tp} FN={fn} FP={fp} TN={tn}"
            )
    return results


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Grid-search Engine params on pre-computed label files")
    parser.add_argument("--labels-dir", type=Path, default=Path("predictions_labels"),
                        help="Labels directory produced by predict_sequential.py")
    parser.add_argument("--output", type=Path, default=Path("grid_search_results.tsv"))
    parser.add_argument("--output-top", type=Path, default=None,
                        help="If set, save top-N rows to this file")
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--min-frames", type=int, default=8)
    return parser


if __name__ == "__main__":
    args = make_parser().parse_args()

    if not args.labels_dir.exists():
        raise SystemExit(f"Labels dir not found: {args.labels_dir}")

    grouped = load_labels_dir(args.labels_dir, min_frames=args.min_frames)

    wf = sum(1 for (cat, _) in grouped if cat == "wildfire")
    fp_count = sum(1 for (cat, _) in grouped if cat == "fp")
    print(f"Loaded {len(grouped)} sequences with ≥{args.min_frames} frames ({wf} wildfire, {fp_count} fp)\n")

    nb_frames_values = list(range(4, 9))
    conf_thresh_values = [round(v, 2) for v in np.arange(0.05, 0.41, 0.05)]

    results = run_grid(grouped, nb_frames_values, conf_thresh_values)

    results_df = pd.DataFrame(results).sort_values("f1", ascending=False)
    print(f"\n── Top {args.top_n} by F1 ──────────────────────────────────────────────")
    print(results_df.head(args.top_n).to_string(index=False))

    results_df.to_csv(args.output, sep="\t", index=False)
    print(f"\nFull results → {args.output}")

    if args.output_top is not None:
        args.output_top.parent.mkdir(parents=True, exist_ok=True)
        results_df.head(args.top_n).to_csv(args.output_top, sep="\t", index=False)
        print(f"Top {args.top_n} results → {args.output_top}")
