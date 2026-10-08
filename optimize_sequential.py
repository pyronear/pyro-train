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
from pathlib import Path

import numpy as np
import pandas as pd

from pyro_train.model.sequential import Replay, load_labels_dir


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
            replay = Replay(nb_frames, float(conf_thresh))
            tp = fn = fp = tn = 0
            for (category, _), frames in grouped.items():
                alerted = replay.trigger(frames) is not None
                if category == "wildfire":
                    tp += alerted
                    fn += not alerted
                else:
                    fp += alerted
                    tn += not alerted

            precision = tp / (tp + fp) if (tp + fp) > 0 else None
            recall = tp / (tp + fn) if (tp + fn) > 0 else None
            f1 = (
                2 * precision * recall / (precision + recall)
                if precision and recall and (precision + recall) > 0
                else None
            )
            fpr = fp / (fp + tn) if (fp + tn) > 0 else None
            # Youden's J: unlike F1, it does not move with the val smoke/FP ratio.
            youden = recall - fpr if recall is not None and fpr is not None else None

            results.append(
                {
                    "nb_consecutive_frames": nb_frames,
                    "conf_thresh": round(conf_thresh, 4),
                    "tp": tp,
                    "fn": fn,
                    "fp": fp,
                    "tn": tn,
                    "precision": round(precision, 4) if precision is not None else None,
                    "recall": round(recall, 4) if recall is not None else None,
                    "f1": round(f1, 4) if f1 is not None else None,
                    "fpr": round(fpr, 4) if fpr is not None else None,
                    "youden": youden,
                }
            )
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
    parser = argparse.ArgumentParser(
        description="Grid-search Engine params on pre-computed label files"
    )
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=Path("predictions_labels"),
        help="Labels directory produced by predict_sequential.py",
    )
    parser.add_argument("--output", type=Path, default=Path("grid_search_results.tsv"))
    parser.add_argument(
        "--output-top",
        type=Path,
        default=None,
        help="If set, save top-N rows to this file",
    )
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--min-frames", type=int, default=0)
    return parser


if __name__ == "__main__":
    args = make_parser().parse_args()

    if not args.labels_dir.exists():
        raise SystemExit(f"Labels dir not found: {args.labels_dir}")

    grouped = load_labels_dir(args.labels_dir, min_frames=args.min_frames)

    wf = sum(1 for (cat, _) in grouped if cat == "wildfire")
    fp_count = sum(1 for (cat, _) in grouped if cat == "fp")
    print(
        f"Loaded {len(grouped)} sequences with ≥{args.min_frames} frames ({wf} wildfire, {fp_count} fp)\n"
    )

    nb_frames_values = list(range(4, 9))
    conf_thresh_values = [round(v, 2) for v in np.arange(0.05, 0.41, 0.05)]

    results = run_grid(grouped, nb_frames_values, conf_thresh_values)

    # Ties go to the lower FPR, then the grid order, so the pick is deterministic.
    results_df = pd.DataFrame(results).sort_values(
        ["youden", "fpr", "nb_consecutive_frames", "conf_thresh"],
        ascending=[False, True, True, True],
    )
    results_df["youden"] = results_df["youden"].round(4)
    print(
        f"\n── Top {args.top_n} by Youden's J (recall − FPR) ─────────────────────────"
    )
    print(results_df.head(args.top_n).to_string(index=False))

    results_df.to_csv(args.output, sep="\t", index=False)
    print(f"\nFull results → {args.output}")

    if args.output_top is not None:
        args.output_top.parent.mkdir(parents=True, exist_ok=True)
        results_df.head(args.top_n).to_csv(args.output_top, sep="\t", index=False)
        print(f"Top {args.top_n} results → {args.output_top}")
