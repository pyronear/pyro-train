"""Small regression checks for replay, training settings, and dataset references."""

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from pyro_train.model.sequential import Replay, load_engine_class, reset_state
from tests.test_eval_report import load


def test_replay_matches_engine(tmp_path, monkeypatch):
    Engine = load_engine_class()
    monkeypatch.setattr(
        "pyroengine.engine.Classifier",
        lambda **_: SimpleNamespace(post_process=lambda pred, **_: pred),
    )
    empty = np.empty((0, 5), dtype=np.float32)
    hit = np.array([[0.1, 0.1, 0.2, 0.2, 0.8]], dtype=np.float32)
    wide = np.array([[0, 0, 0.5, 0.2, 0.9]], dtype=np.float32)
    sequences = [[], [empty] * 15, [hit] * 15, [wide] * 15, [hit, empty] * 8]
    for nb in (4, 6, 8):
        for threshold in (0.05, 0.25, 0.4):
            engine = Engine(
                nb_consecutive_frames=nb,
                conf_thresh=threshold,
                cache_folder=str(tmp_path),
            )
            replay = Replay(nb, threshold)
            for frames in sequences:
                reset_state(engine)
                expected = None
                for index, pred in enumerate(frames):
                    if (
                        engine.predict(Image.new("RGB", (1, 1)), fake_pred=pred)
                        > threshold
                    ):
                        expected = index
                        break
                assert replay.trigger(frames) == expected
                assert len(replay.engine._states) == 1


def test_training_forwards_settings_and_resolves_device(tmp_path, monkeypatch):
    from pyro_train.model.yolo import train as training

    data = tmp_path / "data.yaml"
    data.touch()
    seen = {}
    monkeypatch.setattr(
        training, "install_camera_robustness_augmentations", lambda: None
    )
    monkeypatch.setattr(training, "resolve_device", lambda: "cuda")
    model = SimpleNamespace(train=lambda **kwargs: seen.update(kwargs))
    params = {"model_type": "yolo11s.pt", "workers": 2, "cache": "disk"}
    training.train(model, data, params)
    assert seen["device"] == "cuda" and "model_type" not in seen
    assert seen["workers"] == 2 and seen["cache"] == "disk"
    training.train(model, data, {**params, "device": 0}, device="cpu")
    assert seen["device"] == "cpu" and "model_type" in params


def test_dataset_full_reference_and_subsample(tmp_path):
    from pyro_train.data.utils import yaml_read
    from ultralytics.data.base import BaseDataset
    from ultralytics.data.utils import img2label_paths

    source, output = tmp_path / "source", tmp_path / "output"
    for split in ("train", "val"):
        (source / "images" / split).mkdir(parents=True)
        (source / "labels" / split).mkdir(parents=True)
        for index in range(4):
            Image.new("RGB", (32, 32)).save(source / "images" / split / f"{index}.jpg")
            (source / "labels" / split / f"{index}.txt").write_text("0 .5 .5 .2 .2\n")
        Image.new("RGB", (32, 32)).save(source / "images" / split / "extra.png")
    script = load("scripts/data/model_input/build.py")
    command = [sys.executable, script.__file__, "--input-dir", str(source)]
    command += ["--output-dir", str(output)]
    subprocess.run([*command, "--sampling-ratio", "1"], check=True)
    config = yaml_read(output / "datasets" / "data.yaml")
    files = BaseDataset.get_img_files(
        SimpleNamespace(prefix="", fraction=1), config["train"]
    )
    assert len(files) == 4 and all(path.endswith(".jpg") for path in files)
    assert all(Path(path).exists() for path in img2label_paths(files))
    assert not list(output.rglob("*.jpg"))
    subprocess.run([*command, "--sampling-ratio", "0.5"], check=True)
    assert len(list((output / "datasets/train/images").glob("*.jpg"))) == 2
    assert len(list((output / "datasets/val/images").glob("*.jpg"))) == 4
    assert not script.validate_parsed_args({"input_dir": source, "output_dir": source})


@pytest.mark.parametrize("outcome", [False, True, ValueError])
def test_direct_evaluation_releases_state(tmp_path, monkeypatch, outcome):
    evaluator = load("scripts/model/yolo/evaluate_sequential.py")
    engine = SimpleNamespace(
        _states={}, occlusion_masks={}, nb_consecutive_frames=4, conf_thresh=0.25
    )
    Image.new("RGB", (32, 32)).save(tmp_path / "frame.jpg")

    def predict(image, **_):
        if outcome is ValueError:
            raise ValueError("failed inference")
        return float(outcome)

    engine.predict = predict
    if outcome is ValueError:
        with pytest.raises(ValueError):
            evaluator.evaluate_sequence(engine, tmp_path, "sequence", 15)
    else:
        assert evaluator.evaluate_sequence(engine, tmp_path, "sequence", 15) is outcome
    assert engine._states == engine.occlusion_masks == {}
