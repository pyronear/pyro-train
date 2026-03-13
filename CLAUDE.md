# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

End-to-end ML training pipeline for wildfire detection using YOLO models. Handles data preparation, model training, hyperparameter optimization, model export (ONNX/NCNN), and GitHub releases.

## Setup

```bash
uv sync                    # Install dependencies into virtual environment
uv run pre-commit install  # Install git hooks
```

Data is managed via DVC with an S3 remote (`s3://pyronear-ml/dvc`). Requires AWS credentials under the `pyronear` profile.

## Common Commands

```bash
# Development
uv run ruff check .        # Lint
uv run ruff format .       # Format
uv run mypy src/           # Type check
uv run pytest              # Run tests (or: make run_test_suite)

# DVC Pipeline
dvc dag                    # View pipeline DAG
dvc repro                  # Run full pipeline

# MLFlow
make mlflow_start          # Start experiment tracking UI at localhost:5000
make mlflow_stop

# Hyperparameter search
make run_yolo_wide_hyperparameter_search   # 50 iterations, fast
make run_yolo_narrow_hyperparameter_search # 5 iterations, deep

# Benchmark
make run_yolo_benchmark    # Generate benchmark CSV from trained models
```

## Architecture

### Data Flow (DVC Pipeline in `dvc.yaml`)

```
01_raw/ (wildfire dataset)
  → build_model_input → 03_model_input/ (YOLO format, 5% sample)
  → train_yolo_baseline_small / train_yolo_baseline / train_yolo_best
  → build_manifest_yolo_best → 06_reporting/
  → export_yolo_best (ONNX + NCNN, cpu/mps matrix) → 04_models/yolo-export/
```

### Code Structure

- **`src/pyro_train/`** — Library code:
  - `model/yolo/train.py` — Core training function with default hyperparameters
  - `model/yolo/utils.py` — YOLO version/size enums
  - `model/yolo/hyperparameters/space.py` — Hyperparameter space parsing
  - `data/` — YAML utilities for dataset configs
  - `utils.py` — Shared utilities (file hashing)

- **`scripts/model/yolo/`** — CLI entry points:
  - `train.py` — Training CLI
  - `export.py` — Export to ONNX/NCNN
  - `hyperparameter_search.py` — Random search
  - `benchmark.py` — Benchmark trained models
  - `build_manifest.py` — Build model metadata
  - `configs/` — Training configs (`baseline.yaml`, `best.yaml`)
  - `spaces/` — Hyperparameter search spaces (`wide.yaml`, `narrow.yaml`, `default.yaml`)

- **`scripts/release.py`** — GitHub release automation (requires `GITHUB_ACCESS_TOKEN`)

### Key Design Decisions

- **DVC** manages data versioning and pipeline reproducibility; never commit data files directly.
- **MLFlow** tracks all training experiments; experiments are gitignored but tracked via DVC.
- Model releases follow an adjective+animal naming convention with matching initials (e.g., "dazzling dragonfly").
- Exports target both ONNX and NCNN for edge/mobile deployment.
- Pre-commit hooks prevent direct commits to `main`; always use pull requests.
