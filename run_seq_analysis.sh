#!/bin/bash
# Full sequential analysis pipeline:
#   1. Predict all sets (train, val, test)
#   2. Optimize nb_consecutive_frames + conf_thresh on val predictions
#   3. Copy failures for all sets using best params
#
# Usage: bash run_seq_analysis.sh [model_path]

set -e

MODEL=${1:-./data/02_models/yolo/best/weights/best.pt}
MIN_FRAMES=0  # min label files per sequence (0 = keep all)

echo "=== Sequential analysis with model: $MODEL ==="

# ── 1. Predictions ────────────────────────────────────────────────────────────

echo ""
if [ -d "predictions_labels_train" ]; then
  echo ">>> [1/3] Skipping train predictions (predictions_labels_train already exists)"
else
  echo ">>> [1/3] Predicting train..."
  uv run python predict_sequential.py \
    --model-path "$MODEL" \
    --data-dir ./data/01_model_input/sequential_train_val/train \
    --labels-dir predictions_labels_train
fi

echo ""
if [ -d "predictions_labels_val" ]; then
  echo ">>> [2/3] Skipping val predictions (predictions_labels_val already exists)"
else
  echo ">>> [2/3] Predicting val..."
  uv run python predict_sequential.py \
    --model-path "$MODEL" \
    --data-dir ./data/01_model_input/sequential_train_val/val \
    --labels-dir predictions_labels_val
fi

echo ""
if [ -d "predictions_labels_test" ]; then
  echo ">>> [3/3] Skipping test predictions (predictions_labels_test already exists)"
else
  echo ">>> [3/3] Predicting test..."
  uv run python predict_sequential.py \
    --model-path "$MODEL" \
    --data-dir ./data/test/sequential_test/test \
    --labels-dir predictions_labels_test
fi

# ── 2. Optimize on val ────────────────────────────────────────────────────────

echo ""
echo ">>> Optimizing engine params on val predictions..."
uv run python optimize_sequential.py \
  --labels-dir predictions_labels_val \
  --min-frames $MIN_FRAMES \
  --output grid_search_val.tsv

# Extract best params (TSV sorted by f1 desc, skip header)
BEST=$(tail -n +2 grid_search_val.tsv | head -1)
NB_FRAMES=$(echo "$BEST" | cut -f1)
CONF_THRESH=$(echo "$BEST" | cut -f2)

echo ""
echo ">>> Best params: nb_consecutive_frames=$NB_FRAMES  conf_thresh=$CONF_THRESH"

# ── 3. Copy failures with best params ────────────────────────────────────────

echo ""
echo ">>> Collecting train failures..."
uv run python copy_failures.py \
  --labels-dir predictions_labels_train \
  --data-dir ./data/01_model_input/sequential_train_val/train \
  --output-dir failures_train \
  --nb-consecutive-frames "$NB_FRAMES" \
  --conf-thresh "$CONF_THRESH" \
  --min-frames $MIN_FRAMES

echo ""
echo ">>> Collecting val failures..."
uv run python copy_failures.py \
  --labels-dir predictions_labels_val \
  --data-dir ./data/01_model_input/sequential_train_val/val \
  --output-dir failures_val_seq \
  --nb-consecutive-frames "$NB_FRAMES" \
  --conf-thresh "$CONF_THRESH" \
  --min-frames $MIN_FRAMES

echo ""
echo ">>> Collecting test failures..."
uv run python copy_failures.py \
  --labels-dir predictions_labels_test \
  --data-dir ./data/test/sequential_test/test \
  --output-dir failures_test \
  --nb-consecutive-frames "$NB_FRAMES" \
  --conf-thresh "$CONF_THRESH" \
  --min-frames $MIN_FRAMES

echo ""
echo "=== Done. Best params: nb_frames=$NB_FRAMES  conf_thresh=$CONF_THRESH"
echo ""
echo "To visualize:"
echo "  bash view_seq_failures.sh train"
echo "  bash view_seq_failures.sh val"
echo "  bash view_seq_failures.sh test"
