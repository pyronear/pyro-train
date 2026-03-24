"""
List sequences that have at least one frame tagged in FiftyOne.

Usage:
    uv run python list_tagged_failures.py
    uv run python list_tagged_failures.py --tag wrong --dataset failures_fn_wildfire
"""

import argparse

import fiftyone as fo


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="wrong", help="FiftyOne tag to filter on (default: wrong)")
    parser.add_argument(
        "--dataset",
        default=None,
        help="Dataset name to query (default: both failures_fn_wildfire and failures_fp_alerted)",
    )
    return parser


if __name__ == "__main__":
    args = make_parser().parse_args()

    dataset_names = (
        [args.dataset]
        if args.dataset
        else ["failures_fn_wildfire", "failures_fp_alerted"]
    )

    for name in dataset_names:
        if not fo.dataset_exists(name):
            print(f"{name}: not found (run view_failures.py first)")
            continue

        ds = fo.load_dataset(name)
        tagged = ds.match_tags(args.tag)
        sequences = sorted({s["sequence"] for s in tagged
                            if "is_separator" not in s.field_names or not s["is_separator"]})

        print(f"\n{name} — {len(sequences)} sequence(s) tagged '{args.tag}':")
        for seq in sequences:
            print(f"  {seq}")
