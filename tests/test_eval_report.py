"""Check the test-set metrics and the PR report built from them."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(path: str):
    spec = importlib.util.spec_from_file_location(Path(path).stem, ROOT / path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


copy_failures = load("copy_failures.py")
report = load("scripts/report_training_metrics.py")


def test_wilson_ci95():
    assert copy_failures.wilson_ci95(0, 0) is None
    low, high = copy_failures.wilson_ci95(9, 10)
    assert low < 0.9 < high
    assert copy_failures.wilson_ci95(10, 10)[1] == 1.0


def test_detected_within_frames_counts_misses():
    # Trigger frames are zero-based; None is a missed wildfire.
    within = copy_failures.detected_within_frames([0, 1, 2, 4, None])
    assert within == {"2": 0.4, "3": 0.6, "5": 0.8}
    assert copy_failures.detected_within_frames([]) == {"2": None, "3": None, "5": None}


def test_mcnemar_exact_p():
    assert report.mcnemar_exact_p(0, 0) == 1.0
    assert report.mcnemar_exact_p(5, 5) == 1.0
    assert report.mcnemar_exact_p(10, 0) == pytest.approx(2 / 2**10)


def test_paired_counts_correctness_per_category():
    def preds(alerts):
        return [
            {"category": c, "sequence": s, "alerted": a} for (c, s), a in alerts.items()
        ]

    base = {
        ("wildfire", "a"): False,
        ("wildfire", "b"): True,
        ("fp", "a"): True,
        ("fp", "c"): False,
    }
    cur = {
        ("wildfire", "a"): True,
        ("wildfire", "b"): True,
        ("fp", "a"): False,
        ("fp", "c"): True,
    }
    assert report.paired(preds(cur), preds(base)) == {"wildfire": (1, 0), "fp": (1, 1)}

    with pytest.raises(SystemExit):
        report.paired(preds(cur), preds({("wildfire", "a"): False}))
