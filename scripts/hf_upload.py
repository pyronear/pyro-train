"""
Upload a trained Pyronear YOLO model to Hugging Face Hub.

Creates a model repo named `{hf-org}/{model_type}_{release_name}_{version}`
and uploads:
  - best.pt          (PyTorch weights)
  - best.onnx        (ONNX export, cpu)
  - ncnn/            (NCNN export, cpu)
  - manifest.yaml    (training manifest)
  - README.md        (auto-generated model card)

Usage:
    export HF_TOKEN=hf_...
    uv run python scripts/hf_upload.py \
        --version v5.2.0 \
        --release-name "nimble narwhal" \
        --hf-org pyronear
"""

import argparse
import logging
import os
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, ModelCard, ModelCardData

from pyro_train.data.utils import yaml_read


def make_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=str, required=True, help="e.g. v5.2.0")
    parser.add_argument("--release-name", type=str, required=True, help="e.g. 'nimble narwhal'")
    parser.add_argument("--hf-org", type=str, default="pyronear", help="HuggingFace organisation")
    parser.add_argument("--manifest", type=Path, default=Path("./data/03_reporting/yolo/best/manifest.yaml"))
    parser.add_argument("--model-pt", type=Path, default=Path("./data/02_models/yolo/best/weights/best.pt"))
    parser.add_argument("--exports-dir", type=Path, default=Path("./data/02_models/yolo-export/best/"))
    parser.add_argument("--output-dir", type=Path, default=None, help="If set, copy staged files here instead of uploading")
    parser.add_argument("-log", "--loglevel", default="info")
    return parser


def get_repo_id(hf_org: str, version: str, release_name: str, manifest: dict) -> str:
    model_type = manifest["model"]["model_type"].split(".")[0]
    slug = release_name.replace(" ", "-")
    return f"{hf_org}/{model_type}_{slug}_{version}"


def make_model_card(repo_id: str, version: str, release_name: str, manifest: dict) -> ModelCard:
    model_type = manifest["model"]["model_type"].split(".")[0]
    train_args = manifest.get("train_run_args", {})
    epochs = train_args.get("epochs", "?")
    imgsz = train_args.get("imgsz", "?")
    optimizer = train_args.get("optimizer", "?")
    data_md5 = manifest["data"]["dvc"]["md5"]
    weights_sha256 = manifest["model"]["weights"]["sha256"]

    card_data = ModelCardData(
        license="apache-2.0",
        tags=["wildfire", "fire-detection", "yolo", "object-detection", "pyronear"],
        datasets=["pyronear/pyro-dataset"],
    )

    content = f"""---
{card_data.to_yaml()}
---

# {repo_id}

Pyronear YOLO model for early wildfire smoke detection.

**Release name:** {release_name.title()}
**Version:** {version}

## Model details

| Field | Value |
|---|---|
| Architecture | {model_type} |
| Image size | {imgsz} |
| Epochs | {epochs} |
| Optimizer | {optimizer} |
| Weights SHA-256 | `{weights_sha256[:16]}...` |
| Training data MD5 | `{data_md5[:16]}...` |

## Files

| File | Description |
|---|---|
| `best.pt` | PyTorch weights |
| `best.onnx` | ONNX export (cpu) |
| `ncnn_cpu.zip` | NCNN export (cpu), unzip before use |
| `manifest.yaml` | Full training manifest |

## Usage

### PyTorch (ultralytics)

```python
from ultralytics import YOLO

model = YOLO("best.pt")
results = model.predict("image.jpg", imgsz={imgsz}, conf=0.2, iou=0.01)
for r in results:
    print(r.boxes)  # bounding boxes + confidences
```

### ONNX (onnxruntime)

```python
from huggingface_hub import hf_hub_download
import onnxruntime as ort
import numpy as np
from PIL import Image

path = hf_hub_download(repo_id="{repo_id}", filename="best.onnx")
session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])

img = Image.open("image.jpg").resize(({imgsz}, {imgsz}))
x = np.array(img).transpose(2, 0, 1)[None].astype(np.float32) / 255.0
outputs = session.run(None, {{session.get_inputs()[0].name: x}})
```

### NCNN

```bash
# Unzip first
unzip ncnn_cpu.zip -d ncnn_cpu/
```

### Download with huggingface_hub

```python
from huggingface_hub import snapshot_download

local_dir = snapshot_download(repo_id="{repo_id}")
```

### Pyronear engine (sequential smoke detection)

```python
from pyroengine.engine import Engine

engine = Engine(
    conf_thresh=0.20,
    nb_consecutive_frames=5,
)
# feed frames one by one — engine.predict() returns a score
score = engine.predict(pil_image, cam_id="camera_01")
if score > engine.conf_thresh:
    print("Smoke detected!")
```

## About Pyronear

[Pyronear](https://pyronear.org) builds open-source tools for early wildfire detection.
"""
    return ModelCard(content)


if __name__ == "__main__":
    args = make_cli_parser().parse_args()
    logging.basicConfig(level=args.loglevel.upper())
    logger = logging.getLogger(__name__)

    assert args.manifest.exists(), f"Manifest not found: {args.manifest}"
    assert args.model_pt.exists(), f"Model weights not found: {args.model_pt}"

    manifest = yaml_read(args.manifest)
    repo_id = get_repo_id(args.hf_org, args.version, args.release_name, manifest)
    logger.info(f"Target repo: https://huggingface.co/{repo_id}")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)

        # manifest.yaml
        shutil.copy(args.manifest, tmp_dir / "manifest.yaml")

        # best.pt
        shutil.copy(args.model_pt, tmp_dir / "best.pt")

        # ONNX cpu
        onnx_src = args.exports_dir / "onnx" / "cpu" / "best.onnx"
        if onnx_src.exists():
            shutil.copy(onnx_src, tmp_dir / "best.onnx")
        else:
            logger.warning(f"ONNX export not found: {onnx_src}")

        # NCNN cpu — zipped
        ncnn_src = args.exports_dir / "ncnn" / "cpu"
        if ncnn_src.exists():
            shutil.make_archive(str(tmp_dir / "ncnn_cpu"), "zip", root_dir=ncnn_src, base_dir=".")
        else:
            logger.warning(f"NCNN export not found: {ncnn_src}")

        # README / model card
        card = make_model_card(repo_id, args.version, args.release_name, manifest)
        (tmp_dir / "README.md").write_text(str(card))

        if args.output_dir is not None:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            shutil.copytree(str(tmp_dir), str(args.output_dir), dirs_exist_ok=True)
            print(f"\nFiles saved locally: {args.output_dir}")
        else:
            HF_TOKEN = os.getenv("HF_TOKEN")
            api = HfApi(token=HF_TOKEN or None)  # None → use cached login
            api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
            logger.info(f"Repo ready: {repo_id}")
            logger.info("Uploading files...")
            api.upload_folder(
                folder_path=str(tmp_dir),
                repo_id=repo_id,
                repo_type="model",
                commit_message=f"Upload {args.release_name.title()} {args.version}",
            )
            print(f"\nModel uploaded: https://huggingface.co/{repo_id}")
