from pathlib import Path

from scripts.data.model_input.build import is_background, sample_dataset


def make_split(root: Path, split: str, images: dict[str, str | None]) -> None:
    """Lay out one split. A None label means the file is not written at all."""
    (root / "images" / split).mkdir(parents=True, exist_ok=True)
    (root / "labels" / split).mkdir(parents=True, exist_ok=True)
    for stem, label in images.items():
        (root / "images" / split / f"{stem}.jpg").write_bytes(b"")
        if label is not None:
            (root / "labels" / split / f"{stem}.txt").write_text(label)


def sampled_stems(result: list[dict], split: str) -> set[str]:
    return {
        entry["from"].stem
        for entry in result
        if entry["type"] == "image" and entry["to"].parent.parent.name == split
    }


def test_is_background_on_empty_missing_and_annotated_labels(tmp_path):
    empty = tmp_path / "empty.txt"
    empty.write_text("")
    annotated = tmp_path / "annotated.txt"
    annotated.write_text("0 0.5 0.5 0.1 0.1\n")

    assert is_background(empty)
    assert is_background(tmp_path / "missing.txt")
    assert not is_background(annotated)


def test_exclude_background_drops_them_from_train(tmp_path):
    make_split(
        tmp_path,
        "train",
        {"smoke": "0 0.5 0.5 0.1 0.1\n", "empty": "", "missing": None},
    )
    make_split(tmp_path, "val", {"smoke_val": "0 0.5 0.5 0.1 0.1\n"})

    result = sample_dataset(
        input_dir=tmp_path,
        output_dir=tmp_path / "out",
        sampling_ratio=1,
        exclude_background=True,
    )

    assert sampled_stems(result, "train") == {"smoke"}


def test_val_keeps_its_background_images(tmp_path):
    """The evaluation has to stay comparable to a model trained with them."""
    make_split(tmp_path, "train", {"smoke": "0 0.5 0.5 0.1 0.1\n"})
    make_split(tmp_path, "val", {"smoke_val": "0 0.5 0.5 0.1 0.1\n", "empty_val": ""})

    result = sample_dataset(
        input_dir=tmp_path,
        output_dir=tmp_path / "out",
        sampling_ratio=1,
        exclude_background=True,
    )

    assert sampled_stems(result, "val") == {"smoke_val", "empty_val"}


def test_background_is_kept_by_default(tmp_path):
    make_split(tmp_path, "train", {"smoke": "0 0.5 0.5 0.1 0.1\n", "empty": ""})
    make_split(tmp_path, "val", {"smoke_val": "0 0.5 0.5 0.1 0.1\n"})

    result = sample_dataset(
        input_dir=tmp_path,
        output_dir=tmp_path / "out",
        sampling_ratio=1,
    )

    assert sampled_stems(result, "train") == {"smoke", "empty"}
