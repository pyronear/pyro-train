# Pyronear: Machine Learning Pipeline for Wildfire Detection 🔥

Machine Learning Training Pipeline for Wildfire Detection.

[<img src="./docs/assets/images/ml_space.png" />](https://www.earthtoolsmaker.org/spaces/early_forest_fire_detection/)

![Pipeline Overview](./docs/assets/images/pipeline.png)

## Data Pipeline

The whole repository is organized as a data pipeline that can be run to
train the models and export them to the appropriate formats.

The Data pipeline is organized with a [dvc.yaml](./dvc.yaml) file.

### DVC Stages

This section list and describes all the DVC stages that are defined in the
[dvc.yaml](./dvc.yaml) file:

- __fetch_model_input__: Download the `yolo_train_val` dataset from [pyro-dataset](https://github.com/pyronear/pyro-dataset) at `v3.0.0`.
- __train_yolo_best__: Train the best YOLO model on the full dataset.
- __build_manifest_yolo_best__: Build the manifest.yaml file to attach with the model.
- __fetch_sequential_val__: Download the sequential val dataset from [pyro-dataset](https://github.com/pyronear/pyro-dataset) at `v3.0.0`.
- __predict_sequential_val__: Run per-frame YOLO predictions on the sequential val set and save label files.
- __optimize_sequential_val__: Grid-search engine parameters (nb_consecutive_frames × conf_thresh) on val predictions and save top-20 results.
- __export_yolo_best__: Export the best YOLO model to ONNX and NCNN formats.

## Setup

### 🐍 Python dependencies

Install `uv` with `pipx`:

```sh
pipx install uv
```

Create a virtualenv and install the dependencies with `uv`:

```sh
uv sync
```

Activate the `uv` virutalenv:

```sh
source .venv/bin/activate
```

### Data Dependencies

The dataset is fetched automatically from [pyro-dataset](https://github.com/pyronear/pyro-dataset)
as the first stage of the pipeline — no AWS credentials needed. Just run:

```sh
uv run dvc repro fetch_model_input
```

For model artifacts (weights, exports), you need access to the Pyronear S3 remote,
reserved for Pyronear members. On request, you will be provided with AWS credentials.

Pull all other DVC-tracked files:

```sh
uv run dvc pull
```

![Random batch sample from the dataset](./docs/assets/images/batch.jpg)

##### Setup S3 access

Create the following file `~/.aws/config`:

```toml
[profile pyronear]
region = eu-west-3
```

Add your credentials in the file `~/.aws/credentials` - replace `XXX`
with your access key id and your secret access key:

```toml
[pyronear]
aws_access_key_id = XXX
aws_secret_access_key = XXX
```

Make sure you use the AWS `pyronear` profile:

```bash
export AWS_PROFILE=pyronear
```

## Project structure and conventions

The project is organized following mostly the [cookie-cutter-datascience
guideline](https://drivendata.github.io/cookiecutter-data-science/#directory-structure).

### Data

All the data lives in the `data` folder and follows some [data engineering
conventions](https://docs.kedro.org/en/stable/faq/faq.html#what-is-data-engineering-convention).

### Library Code

The library code is available under the `src/pyro_train/` folder.

### Scripts

The scripts live in the `scripts` folder, they are
commonly CLI interfaces to the library
code.

## DVC

DVC is used to track and define data pipelines and make them
reproducible. See `dvc.yaml`.

To get an overview of the pipeline DAG:

```sh
uv run dvc dag
```

To run the full pipeline:

```sh
uv run dvc repro
```

## MLFlow

An MLFlow server is running when running ML experiments to track
hyperparameters and performances and to streamline model
selection.

To start the mlflow UI server, run the following command:

```sh
make mlflow_start
```

To stop the mlflow UI server, run the following command:

```sh
make mlflow_stop
```

To browse the different runs, open your browser and navigate to the URL:
[http://localhost:5000](http://localhost:5000)

## Test Suite

Run the test suite with the following commmand:

```sh
make run_test_suite
```

## Contribute to the project

### New ML experiments

Follow the steps:

1. Work on a separate git branch: `git checkout -b "<user>/<experiment-name>"`
2. Modify and iterate on the code, then run `dvc repro`. It will rerun
   parts of the pipeline that have been updated.
3. Commit your changes and open a Pull Request to get your changes
   approved and merged.

### Run Random Hyperparameter Search

We use random hyperparameter search to find the best set of hyperparameters for
our models.

#### Wide & Fast

The initial stage is to optimize for exploration of all hyperparameter ranges.
A [wide.yaml](./scripts/model/yolo/spaces/wide.yaml) hyperparamter config file
is available for performing this type of search.

It is good practice to run this search on a small subset of the full dataset to
make quickly iterate over many different combinations of hyperparameters.

Run the wide and fast hyperparameter search with:

```sh
make run_yolo_wide_hyperparameter_search
```

#### Narrow & Deep


The second stage of the hyperparameter search is to run some more narrow and
local searches on identified combinations of good parameters from stage 1.
A [narrow.yaml](./scripts/model/yolo/spaces/narrow.yaml) hyperparameter config
file is available for this type of search.

It is good practice to run this search on the full dataset to get the actual
model performances of the randomly drawn sets of hyperparameters.

Run the narrow and deep hyperparameter search with:

```sh
make run_yolo_narrow_hyperparameter_search
```

#### Custom

Adapt and run this command to launch a specific hyperparamater space search:

```sh
uv run python ./scripts/model/yolo/hyperparameter_search.py \
   --data ./data/03_model_input/yolo_train_val/data.yaml \
   --output-dir ./data/04_models/yolo/ \
   --experiment-name "random_hyperparameter_search" \
   --filepath-space-yaml ./scripts/model/yolo/spaces/default.yaml \
   --n 5 \
   --loglevel "info"
```

One can adapt the hyperparameter space to search by adding a new `space.yaml`
file based on the [default.yaml](./scripts/model/yolo/spaces/default.yaml)

```yaml
model_type:
  type: array
  array_type: str
  values:
    - yolo11n.pt
    - yolo11s.pt
    - yolo12n.pt
    - yolo12s.pt
epochs:
  type: space
  space_type: int
  space_config:
    type: linear
    start: 50
    stop: 70
    num: 10
patience:
  type: space
  space_type: int
  space_config:
    type: linear
    start: 10
    stop: 50
    num: 10
batch:
  type: array
  array_type: int
  values:
    - 16
    - 32
    - 64
...
```

### Generate a benchmark CSV file

```sh
make run_yolo_benchmark
```

## Error Analysis

### Frame-based (YOLO) — `yolo_train_val`

#### 1. Fetch data

```sh
uv run dvc repro fetch_model_input
```

#### 2. Run model and collect failures

```sh
uv run python eval_yolo_val.py --model-path ./data/02_models/yolo/best/weights/best.pt
```

Defaults: `--conf 0.2`, `--imgsz 1024`, device auto-detected.
Failures are saved to `failures_val/fp/` (false positives) and `failures_val/fn/` (false negatives).

#### 3. Visualize in FiftyOne

```sh
uv run python view_yolo_val_failures.py
# FP → http://localhost:5151  |  FN → http://localhost:5152
```

---

### Sequential (Engine) — `sequential_train_val`

Runs predictions on train/val/test, optimizes engine parameters on val, then collects failures.

#### Option A — Using CI outputs (no re-training needed)

CI already runs `predict_sequential_val` and `optimize_sequential_val`. Pull those results and use the best params directly.

**1. Pull CI outputs from DVC:**

```sh
dvc pull data/03_reporting/sequential/
dvc pull data/01_model_input/sequential_train_val/val
```

**2. Read the best engine params from the CI grid search:**

```sh
tail -n +2 data/03_reporting/sequential/grid_search_val.tsv | head -1
# columns: nb_consecutive_frames  conf_thresh  precision  recall  f1  ...
```

**3. Collect val failures** (replace `NB_FRAMES` and `CONF` with values from step 2):

```sh
uv run python copy_failures.py \
  --labels-dir ./data/03_reporting/sequential/predictions_labels_val \
  --data-dir ./data/01_model_input/sequential_train_val/val \
  --output-dir failures_val_seq \
  --nb-consecutive-frames NB_FRAMES \
  --conf-thresh CONF
```

**4. Visualize in FiftyOne:**

```sh
uv run python view_failures.py \
  --failures-dir failures_val_seq \
  --labels-dir ./data/03_reporting/sequential/predictions_labels_val \
  --data-dir ./data/01_model_input/sequential_train_val/val
# FN (missed wildfire) → http://localhost:5151
# FP (false alerts)    → http://localhost:5152
```

#### Option A (test set) — Using CI params, local predictions

The test set is not part of the CI pipeline, so predictions must be run locally. The best engine params from CI can still be reused.

**1. Fetch test data:**

```sh
SSL_CERT_FILE=$(uv run python -c "import certifi; print(certifi.where())") \
uv run dvc get https://github.com/pyronear/pyro-dataset \
  data/processed/sequential_test --rev v3.0.0 \
  --out ./data/test/sequential_test
```

**2. Pull CI grid search results:**

```sh
dvc pull data/03_reporting/sequential/grid_search_val.tsv
```

**3. Read best params:**

```sh
tail -n +2 data/03_reporting/sequential/grid_search_val.tsv | head -1
# columns: nb_consecutive_frames  conf_thresh  precision  recall  f1  ...
```

**4. Run predictions on test set:**

```sh
uv run python predict_sequential.py \
  --model-path ./data/02_models/yolo/best/weights/best.pt \
  --data-dir ./data/test/sequential_test/test \
  --labels-dir predictions_labels_test
```

**5. Collect test failures** (replace `NB_FRAMES` and `CONF` with values from step 3):

```sh
uv run python copy_failures.py \
  --labels-dir predictions_labels_test \
  --data-dir ./data/test/sequential_test/test \
  --output-dir failures_test \
  --nb-consecutive-frames NB_FRAMES \
  --conf-thresh CONF
```

**6. Visualize in FiftyOne:**

```sh
bash view_seq_failures.sh test
# FN (missed wildfire) → http://localhost:5151
# FP (false alerts)    → http://localhost:5152
```

#### Option B — Full local run (re-predict + optimize)

#### 1. Fetch data

```sh
uv run dvc repro fetch_sequential_val
# or fetch all sets manually via dvc get / fetch_data.sh
```

#### 2. Run full sequential analysis

```sh
bash run_seq_analysis.sh [model_path]
# default model: ./data/02_models/yolo/best/weights/best.pt
```

This will:
1. Run per-frame YOLO predictions on train/val/test (cached — skips if labels already exist)
2. Grid-search `nb_consecutive_frames` (4–8) × `conf_thresh` (0.05–0.40) on val
3. Apply best params to train/val/test and copy failures to `failures_train/`, `failures_val_seq/`, `failures_test/`

#### 3. Visualize in FiftyOne

```sh
bash view_seq_failures.sh val    # or: train / test
```

Opens a FiftyOne session at [http://localhost:5151](http://localhost:5151).
Each sequence is separated by a black frame. Shows:
- `predictions` (red) — YOLO detections per frame
- `ground_truth` (green) — GT labels (wildfire sequences only)

Two datasets are loaded:
- `failures_fn_wildfire` — missed wildfire sequences (false negatives)
- `failures_fp_alerted` — sequences that triggered a false alert

## Known Issues

### MPS bounding box corruption on macOS 14+ (ultralytics 8.4.21)

Ultralytics 8.4.21 contains a bug where MPS inference produces corrupted bounding box
X-coordinates (`cx` and `w`) while Y-coordinates remain correct. Root cause: in-place
`clamp_()` operations in `clip_boxes()` are broken by Apple's Metal backend on macOS 14+.
See [ultralytics#23140](https://github.com/ultralytics/ultralytics/issues/23140).

The fix was merged upstream but the version check `MACOS_VERSION.startswith("14.")` does
not match macOS versions beyond 14.x (e.g. macOS 26 / Tahoe). Apply this one-line patch
to `.venv/lib/python3.12/site-packages/ultralytics/utils/__init__.py`:

```python
# Before
NOT_MACOS14 = not (MACOS and MACOS_VERSION.startswith("14."))

# After
NOT_MACOS14 = not (MACOS and int((MACOS_VERSION or "0").split(".")[0]) >= 14)
```

## 🌎 Release a new Model to the world

The script to upload a model to Hugging Face is located in `./scripts/hf_upload.py`.

### 1. Login to Hugging Face

```sh
uv run huggingface-cli login
```

### 2. Upload the model

```sh
uv run python scripts/hf_upload.py \
  --version v6.0.0 \
  --release-name "nimble narwhal" \
  --hf-org pyronear
```

This will create `pyronear/{model_type}_{release-name}_{version}` on Hugging Face and upload:
- `best.pt` — PyTorch weights
- `best.onnx` — ONNX export (cpu)
- `ncnn_cpu.zip` — NCNN export (cpu)
- `manifest.yaml` — training manifest
- `README.md` — auto-generated model card

### 3. (Optional) Export locally without uploading

```sh
uv run python scripts/hf_upload.py \
  --version v6.0.0 \
  --release-name "nimble narwhal" \
  --hf-org pyronear \
  --output-dir ./hf_export/
```

__Note__: The naming convention is an adjective paired with an animal name
starting with the same letter, in alphabetical order across releases
(eg. `nimble narwhal`, `outstanding octopus`, `powerful panther`, ...).
