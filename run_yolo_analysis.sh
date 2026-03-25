#!/bin/bash
# Full frame-by-frame YOLO analysis pipeline:
#   1. Optimize conf threshold on val split
#   2. Evaluate train + val with best conf and copy failures
#
# Usage: bash run_yolo_analysis.sh [model_path]

set -e

MODEL=${1:-./data/02_models/yolo/best/weights/best.pt}
DATA_DIR="./data/01_model_input/yolo_train_val"

echo "=== YOLO frame-by-frame analysis with model: $MODEL ==="

# ── 1. Optimize conf threshold on val ────────────────────────────────────────

echo ""
echo ">>> [1/3] Optimizing conf threshold on val split..."
uv run python optimize_yolo_val.py \
  --model-path "$MODEL" \
  --data-dir "$DATA_DIR" \
  --split val \
  --output grid_search_yolo_val.tsv

# Extract best conf threshold (TSV sorted by f1 desc, skip header)
CONF_THRESH=$(tail -n +2 grid_search_yolo_val.tsv | head -1 | cut -f1)

echo ""
echo ">>> Best conf threshold: $CONF_THRESH"

# ── 2. Evaluate train ─────────────────────────────────────────────────────────

echo ""
echo ">>> [2/3] Evaluating train split..."
uv run python eval_yolo_val.py \
  --model-path "$MODEL" \
  --data-dir "$DATA_DIR" \
  --split train \
  --conf "$CONF_THRESH" \
  --output-dir failures_yolo_train

# ── 3. Evaluate val ───────────────────────────────────────────────────────────

echo ""
echo ">>> [3/3] Evaluating val split..."
uv run python eval_yolo_val.py \
  --model-path "$MODEL" \
  --data-dir "$DATA_DIR" \
  --split val \
  --conf "$CONF_THRESH" \
  --output-dir failures_yolo_val

echo ""
echo "=== Done. Best conf threshold: $CONF_THRESH"
echo ""
echo "To visualize:"
echo "  bash view_yolo_failures.sh val"
echo "  bash view_yolo_failures.sh train"
