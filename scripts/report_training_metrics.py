"""Render the training-result PR body: the new model against main's, on one test set.

The train workflow scores both models in the same run, on the same test
sequences, so the comparison holds when the test set or the eval code changes.
Prints Markdown.

Sequential metrics are the prevalence-free ones (recall, FPR, detection delay
over all wildfire): the test smoke/FP ratio is arbitrary, so precision and F1
are not shown. The paired section counts the sequences the two models decide
differently and gives McNemar's exact p-value, for wildfire and fp separately.
"""

import argparse
import csv
import json
import math
from pathlib import Path

YOLO_ROWS = [
    ("mAP@50", "map50"),
    ("mAP@50-95", "map50_95"),
    ("Precision", "precision"),
    ("Recall", "recall"),
]

# (label, key, is_ratio, ci key)
SEQ_ROWS = [
    ("Recall", "recall", True, "recall_ci95"),
    ("FPR", "fpr", True, "fpr_ci95"),
    ("Missed wildfire (FN)", "fn", False, None),
    ("False alerts (FP)", "fp", False, None),
    ("Detected within 2 frames", ("detected_within_frames", "2"), True, None),
    ("Detected within 3 frames", ("detected_within_frames", "3"), True, None),
    ("Detected within 5 frames", ("detected_within_frames", "5"), True, None),
]


def get(metrics: dict, key):
    if isinstance(key, tuple):
        return (metrics.get(key[0]) or {}).get(key[1])
    return metrics.get(key)


def fmt(value, is_ratio: bool = True, ci=None) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    out = f"{value:.4f}" if is_ratio else f"{value:g}"
    return out if ci is None else f"{out} [{ci[0]:.3f}, {ci[1]:.3f}]"


def delta(new, old, is_ratio: bool = True) -> str:
    if new is None or old is None:
        return "n/a"
    d = new - old
    return f"{d:+.4f}" if is_ratio else f"{d:+g}"


def mcnemar_exact_p(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value on b vs c discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2**n
    return min(1.0, 2 * tail)


def paired(current: list[dict], baseline: list[dict]) -> dict[str, tuple[int, int]]:
    """Per category: (sequences the new model fixed, sequences it broke)."""
    cur = {(p["category"], p["sequence"]): p["alerted"] for p in current}
    base = {(p["category"], p["sequence"]): p["alerted"] for p in baseline}
    if cur.keys() != base.keys():
        diff = sorted(cur.keys() ^ base.keys())
        raise SystemExit(f"the two evaluations cover different sequences: {diff[:5]}")
    out = {"wildfire": [0, 0], "fp": [0, 0]}
    for (category, sequence), alerted in cur.items():
        # Correct means alerting on wildfire and staying silent on a false positive.
        want = category == "wildfire"
        cur_ok, base_ok = alerted == want, base[(category, sequence)] == want
        if cur_ok and not base_ok:
            out[category][0] += 1
        elif base_ok and not cur_ok:
            out[category][1] += 1
    return {category: (v[0], v[1]) for category, v in out.items()}


def yolo_section(cur: dict, base: dict) -> str:
    lines = [
        "## Image-level evaluation on test",
        "",
        "| Metric | main | current | Δ |",
        "|--------|------|---------|---|",
    ]
    for label, key in YOLO_ROWS:
        old, new = base.get(key), cur.get(key)
        lines.append(f"| {label} | {fmt(old)} | {fmt(new)} | {delta(new, old)} |")
    return "\n".join(lines)


def seq_section(cur: dict, base: dict) -> str:
    n_wf, n_fp = cur["tp"] + cur["fn"], cur["fp"] + cur["tn"]
    lines = [
        "## Sequential evaluation on test",
        "",
        f"Sequences: {cur['num_sequences']} ({n_wf} wildfire, {n_fp} fp). "
        "Both models are scored in this run, on the same sequences, each with "
        "its own engine params. Intervals are Wilson 95%.",
        "",
        f"Engine params: main nb_frames={base['nb_consecutive_frames']} "
        f"conf={base['conf_thresh']}, current nb_frames={cur['nb_consecutive_frames']} "
        f"conf={cur['conf_thresh']}",
        "",
        "| Metric | main | current | Δ |",
        "|--------|------|---------|---|",
    ]
    for label, key, is_ratio, ci in SEQ_ROWS:
        old, new = get(base, key), get(cur, key)
        old_cell = fmt(old, is_ratio, ci and base.get(ci))
        new_cell = fmt(new, is_ratio, ci and cur.get(ci))
        lines.append(
            f"| {label} | {old_cell} | {new_cell} | {delta(new, old, is_ratio)} |"
        )
    return "\n".join(lines)


def paired_section(pairs: dict[str, tuple[int, int]]) -> str:
    lines = [
        "## Paired comparison (same sequences)",
        "",
        "| | fixed by current | broken by current | McNemar exact p |",
        "|---|---|---|---|",
    ]
    names = {"wildfire": "wildfire (caught / missed)", "fp": "fp (silenced / alerted)"}
    for category, name in names.items():
        fixed, broken = pairs[category]
        lines.append(
            f"| {name} | {fixed} | {broken} | {mcnemar_exact_p(fixed, broken):.3f} |"
        )
    lines += [
        "",
        "Only the sequences the two models decide differently count. A large p means "
        "the difference is within noise, not that the models are equivalent.",
    ]
    return "\n".join(lines)


def grid_section(grid_path: Path, top_n: int = 10) -> str:
    with grid_path.open() as f:
        rows = list(csv.DictReader(f, delimiter="\t"))[:top_n]
    lines = [
        f"## Engine parameter search (top {top_n} on val, by Youden's J = recall − FPR)",
        "",
        "| nb_frames | conf | TP | FN | FP | TN | Recall | FPR | J |",
        "|-----------|------|----|----|----|----|--------|-----|---|",
    ]
    lines += [
        f"| {r['nb_consecutive_frames']} | {r['conf_thresh']} | {r['tp']} | {r['fn']} "
        f"| {r['fp']} | {r['tn']} | {r['recall']} | {r['fpr']} | {r['youden']} |"
        for r in rows
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("results", "seq-results", "predictions"):
        parser.add_argument(f"--{name}", type=Path, required=True)
        parser.add_argument(f"--baseline-{name}", type=Path, required=True)
    parser.add_argument("--grid", type=Path, required=True)
    parser.add_argument("--result-branch", default="")
    parser.add_argument("--baseline-ref", default="")
    parser.add_argument("--dataset-rev", default="")
    args = parser.parse_args()

    def load(path: Path):
        return json.loads(path.read_text())

    sections = [
        yolo_section(load(args.results), load(args.baseline_results)),
        seq_section(load(args.seq_results), load(args.baseline_seq_results)),
        paired_section(paired(load(args.predictions), load(args.baseline_predictions))),
        grid_section(args.grid),
        f"**Branch:** `{args.result_branch}` | **Baseline:** main @ `{args.baseline_ref}` "
        f"| **Test dataset:** pyronear/pyro-dataset @ {args.dataset_rev}",
    ]
    print("\n\n".join(sections))


if __name__ == "__main__":
    main()
